"""Bac à sable Docker pour les commandes shell de l'agent.

Avec ``--docker``, chaque projet (``projet_<id>``) dispose de son propre
conteneur. Le workspace y est monté au même chemin que sur l'hôte : les
chemins vus par le LLM restent valables des deux côtés. Un venv persistant
(``<workspace>/.venv``) est créé au premier lancement puis réutilisé ; il est
en tête du ``PATH``, donc ``pip install`` et ``python3`` l'utilisent
directement (plus d'erreur ``externally-managed-environment``).

Cycle de vie (``ensure``) :

1. construit l'image (``docker/Dockerfile``) si elle n'existe pas ;
2. réutilise le conteneur s'il tourne, le redémarre s'il est arrêté,
   le crée sinon ;
3. crée le venv s'il est absent.
"""

import hashlib
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .logger import get_logger

logger = get_logger("sandbox")

DEFAULT_IMAGE = "agent-langchain-sandbox:latest"
DOCKERFILE_DIR = Path(__file__).resolve().parent.parent / "docker"
VENV_DIRNAME = ".venv"

# Appel de pip en début de commande (ou après ; && || | ( $( ou un saut de ligne),
# avec sudo/env optionnels : pip, pip3, pip3.12, /usr/bin/pip3, python3 -m pip...
# Le lookahead après « pip » écarte pipx, pip-tools, etc.
_PIP_CALL = re.compile(
    r"(?P<pre>(?:^|[;&|(\n]|\$\()\s*)"
    r"(?:sudo(?:\s+-\S+)*\s+)?(?:env\s+)?"
    r"(?P<exe>(?:\S*/)?(?:pip(?:3(?:\.\d+)?)?|python(?:3(?:\.\d+)?)?\s+-m\s+pip))"
    r"(?=\s|$)"
    r"(?P<args>[^;&|\n)]*)"
)
# Options incompatibles avec un venv (--user y échoue) ou inutiles dedans.
_PIP_BAD_FLAGS = re.compile(r"\s--(?:user|break-system-packages)(?=\s|$)")


def inject_venv_pip(command: str, venv_dir: Path) -> str:
    """Remplace chaque appel à pip par le pip du venv du projet.

    Le ``PATH`` du conteneur pointe déjà sur le venv, mais le LLM écrit
    parfois ``sudo pip``, ``/usr/bin/pip3``, ``python3 -m pip ... --user`` :
    ces formes contourneraient le venv ou échoueraient. Chaque appel devient
    ``<venv>/bin/python -m pip`` et ``--user`` / ``--break-system-packages``
    sont retirés.

    Args:
        command: Commande Bash générée par l'agent.
        venv_dir: Venv du projet (chemin vu dans le conteneur).

    Returns:
        La commande réécrite (identique si elle n'appelle pas pip).
    """

    venv_pip = f"{venv_dir}/bin/python -m pip"

    def _replace(match: re.Match) -> str:
        args = _PIP_BAD_FLAGS.sub("", match.group("args"))
        return f"{match.group('pre')}{venv_pip}{args}"

    return _PIP_CALL.sub(_replace, command)


class SandboxError(RuntimeError):
    """Erreur de préparation ou d'utilisation du conteneur."""


def _docker(*args: str, timeout: int = 120, check: bool = True) -> subprocess.CompletedProcess:
    """Lance une commande ``docker`` et journalise son résultat.

    Args:
        *args: Arguments passés à ``docker``.
        timeout: Délai maximal en secondes.
        check: Lève ``SandboxError`` si le code retour est non nul.

    Returns:
        Le résultat de ``subprocess.run``.

    Raises:
        SandboxError: Docker est absent ou la commande a échoué (``check``).
    """

    cmd = ["docker", *args]
    logger.debug("docker: %s", " ".join(cmd))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise SandboxError("Docker n'est pas installé ou absent du PATH.") from exc
    except subprocess.TimeoutExpired as exc:
        raise SandboxError(f"Timeout docker après {timeout}s : {' '.join(cmd)}") from exc

    logger.debug("  -> code=%s stdout=%r stderr=%r", result.returncode,
                 result.stdout[-500:], result.stderr[-500:])

    if check and result.returncode != 0:
        raise SandboxError(f"Échec de `{' '.join(cmd)}` : {result.stderr.strip()[-1000:]}")

    return result


@dataclass(frozen=True)
class DockerSandbox:
    """Conteneur Docker dédié à un projet.

    Attributes:
        project_dir: Workspace du projet (monté au même chemin dans le conteneur).
        image: Image Docker utilisée.
    """

    project_dir: Path
    image: str = DEFAULT_IMAGE

    @property
    def name(self) -> str:
        """Nom du conteneur : ``agent_<projet>_<hash du chemin>``.

        Le hash évite qu'un workspace de test et le workspace réel partagent
        le même conteneur.
        """

        digest = hashlib.sha1(str(self.project_dir).encode()).hexdigest()[:8]
        return f"agent_{self.project_dir.name}_{digest}"

    @property
    def venv_dir(self) -> Path:
        """Chemin du venv persistant du projet."""

        return self.project_dir / VENV_DIRNAME

    def _state(self) -> str | None:
        """Retourne l'état du conteneur (``running``, ``exited``...) ou None s'il n'existe pas."""

        result = _docker("container", "inspect", "-f", "{{.State.Status}}", self.name, check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    def _ensure_image(self) -> None:
        """Construit l'image si elle n'existe pas encore."""

        if _docker("image", "inspect", self.image, check=False).returncode == 0:
            logger.debug("Image %s présente", self.image)
            return

        if self.image != DEFAULT_IMAGE:
            logger.info("Image %s absente : docker pull", self.image)
            _docker("pull", self.image, timeout=900)
            return

        logger.info("Construction de l'image %s (première fois, quelques minutes)", self.image)
        _docker(
            "build",
            "--build-arg", f"UID={os.getuid()}",
            "--build-arg", f"GID={os.getgid()}",
            "-t", self.image,
            str(DOCKERFILE_DIR),
            timeout=1800,
        )

    def _create(self) -> None:
        """Crée et démarre le conteneur du projet."""

        workdir = str(self.project_dir)
        venv = str(self.venv_dir)
        _docker(
            "run", "-d",
            "--name", self.name,
            "--network", "host",          # pip/curl + accès à SearXNG/Ollama via localhost
            "--user", f"{os.getuid()}:{os.getgid()}",
            "-v", f"{workdir}:{workdir}",
            "-w", workdir,
            "-e", f"VIRTUAL_ENV={venv}",
            "-e", f"PATH={venv}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "-e", "PIP_DISABLE_PIP_VERSION_CHECK=1",
            "--label", "agent_langchain=1",
            self.image,
            "sleep", "infinity",
        )

    def ensure(self) -> "DockerSandbox":
        """Prépare le conteneur et le venv (idempotent).

        Returns:
            L'instance elle-même, prête à exécuter des commandes.

        Raises:
            SandboxError: Docker indisponible ou préparation impossible.
        """

        self.project_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_image()

        state = self._state()
        if state == "running":
            logger.info("Conteneur %s réutilisé", self.name)
        elif state is not None:
            logger.info("Conteneur %s (%s) redémarré", self.name, state)
            _docker("start", self.name)
        else:
            logger.info("Conteneur %s créé (image %s)", self.name, self.image)
            self._create()

        # pyvenv.cfg et non bin/python : ce lien pointe vers le Python du conteneur,
        # absent de l'hôte, donc Path.exists() le voit cassé.
        if not (self.venv_dir / "pyvenv.cfg").is_file():
            logger.info("Création du venv %s", self.venv_dir)
            result = _docker("exec", self.name, "python3", "-m", "venv", str(self.venv_dir),
                             timeout=300, check=False)
            if result.returncode != 0:
                raise SandboxError(f"Création du venv impossible : {result.stderr.strip()}")
        else:
            logger.info("Venv %s réutilisé", self.venv_dir)

        return self

    def reset(self) -> None:
        """Supprime le conteneur et le venv (utilisé par ``--reset``)."""

        if self._state() is not None:
            logger.info("Suppression du conteneur %s", self.name)
            _docker("rm", "-f", self.name)

        if self.venv_dir.exists():
            logger.info("Suppression du venv %s", self.venv_dir)
            # Le venv a été écrit avec l'UID de l'hôte : suppression directe possible.
            shutil.rmtree(self.venv_dir)

    def inject(self, command: str) -> str:
        """Réécrit les appels à pip de ``command`` vers le venv du projet.

        Args:
            command: Commande Bash générée par l'agent.

        Returns:
            La commande à exécuter dans le conteneur.
        """

        injected = inject_venv_pip(command, self.venv_dir)
        if injected != command:
            logger.info("pip redirigé vers le venv : %s", injected)
        return injected

    def wrap(self, command: str, cwd: Path, timeout: int) -> list[str]:
        """Construit la commande ``docker exec`` qui exécute ``command`` dans le conteneur.

        ``timeout`` est appliqué *dans* le conteneur : sinon tuer le client
        ``docker exec`` laisserait le processus tourner.

        Args:
            command: Commande Bash générée par l'agent.
            cwd: Répertoire de travail (chemin identique hôte/conteneur).
            timeout: Délai maximal en secondes.

        Returns:
            La liste d'arguments à passer à ``subprocess.run``.
        """

        return [
            "docker", "exec", "-w", str(cwd), self.name,
            "timeout", "-k", "5", str(timeout), "bash", "-c", command,
        ]
