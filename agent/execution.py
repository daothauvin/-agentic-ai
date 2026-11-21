"""Exécution contrôlée des commandes Linux."""

import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .logger import get_logger, preview

if TYPE_CHECKING:
    from .sandbox import DockerSandbox

logger = get_logger("shell")


def compact(text: str, limit: int = 16000) -> str:
    """Limite la taille d'un texte avant stockage ou transmission au LLM."""

    text = text or ""

    if len(text) <= limit:
        return text

    return text[:limit] + "\n...[tronqué]..."


# Séparateurs Bash : chaque segment est analysé comme une commande simple.
_SEPARATORS = {";", "&&", "||", "|", "&", "(", ")", "\n"}
# Préfixes ignorés devant le gestionnaire de paquets.
_PREFIXES = {"sudo", "env", "nohup", "time"}
# Options suivies d'une valeur (la valeur n'est pas un nom de paquet).
_VALUE_OPTIONS = {
    "pip": {"-r", "--requirement", "-c", "--constraint", "-i", "--index-url",
            "--extra-index-url", "-f", "--find-links", "-t", "--target",
            "--prefix", "--root", "--trusted-host", "--python", "--platform"},
    "apt": {"-o", "-t", "--target-release", "-c", "--config-file"},
    "npm": {"--prefix", "--registry", "--tag"},
    "cargo": {"--root", "--version", "--git", "--branch", "--tag", "--rev",
              "--path", "--index", "--registry"},
}
_PIP_EXE = re.compile(r"pip(?:3(?:\.\d+)?)?")
_PYTHON_EXE = re.compile(r"python(?:3(?:\.\d+)?)?")


def _strip_prefixes(words: list[str]) -> list[str]:
    """Retire ``sudo`` (et ses options), ``env VAR=val``, ``nohup``... en tête.

    Args:
        words: Mots d'une commande simple.

    Returns:
        Les mots à partir du programme réellement appelé.
    """

    i = 0
    while i < len(words):
        word = words[i]
        if word in _PREFIXES:
            i += 1
            # Options de sudo/env (-H, -E, -u root...). -u/-g prennent une valeur.
            while i < len(words) and words[i].startswith("-"):
                i += 2 if words[i] in ("-u", "-g") else 1
        elif "=" in word and not word.startswith("-") and word.split("=", 1)[0].isidentifier():
            i += 1  # affectation VAR=valeur
        else:
            break
    return words[i:]


def _parse_simple(words: list[str]) -> tuple[str, list[str]] | None:
    """Analyse une commande simple (sans séparateur Bash).

    Args:
        words: Mots de la commande.

    Returns:
        ``(gestionnaire, paquets)`` ou None si ce n'est pas une installation.
    """

    words = _strip_prefixes(words)
    if not words:
        return None

    exe = words[0].rsplit("/", 1)[-1]
    if _PIP_EXE.fullmatch(exe):
        manager, rest = "pip", words[1:]
    elif _PYTHON_EXE.fullmatch(exe) and words[1:3] == ["-m", "pip"]:
        manager, rest = "pip", words[3:]
    elif exe in ("apt", "apt-get"):
        manager, rest = "apt", words[1:]
    elif exe in ("npm", "cargo"):
        manager, rest = exe, words[1:]
    else:
        return None

    # Options globales avant la sous-commande (apt-get -y install x).
    while rest and rest[0].startswith("-"):
        rest = rest[1:]
    if not rest or rest[0] not in (("install", "i") if manager == "npm" else ("install",)):
        return None

    packages = []
    args = rest[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in _VALUE_OPTIONS[manager]:
            i += 2
            continue
        if not arg.startswith("-"):
            packages.append(arg)
        i += 1

    return (manager, packages) if packages else None


def parse_install_command(command: str) -> tuple[str, list[str]] | None:
    """Détecte une installation de paquets dans une commande Bash.

    Gère ``sudo``/``env`` en préfixe, ``pip3``, ``/usr/bin/pip3.12``,
    ``python3 -m pip``, ``apt-get -y install`` et les commandes enchaînées
    (``&&``, ``;``, ``|``) : seuls les mots du segment d'installation sont
    retenus (``pip install x && python3 t.py`` donne ``["x"]``).

    Args:
        command: Commande Bash.

    Returns:
        ``(gestionnaire, paquets)`` pour le premier gestionnaire trouvé
        (paquets de tous ses segments fusionnés), ou None.
    """

    try:
        # Les sauts de ligne séparent des commandes : shlex les traite comme des espaces.
        lexer = shlex.shlex(command.replace("\n", " ; "), posix=True, punctuation_chars=";&|()")
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:
        logger.debug("parse_install_command: commande non analysable: %r", command)
        return None

    segments, current = [], []
    for token in tokens:
        if token in _SEPARATORS or set(token) <= set(";&|()"):
            segments.append(current)
            current = []
        else:
            current.append(token)
    segments.append(current)

    found: tuple[str, list[str]] | None = None
    for segment in segments:
        parsed = _parse_simple(segment)
        if parsed is None:
            continue
        if found is None:
            found = parsed
        elif parsed[0] == found[0]:
            found[1].extend(p for p in parsed[1] if p not in found[1])

    return found


def run_command(
    command: str,
    cwd: Path,
    timeout: int,
    sandbox: "DockerSandbox | None" = None,
) -> dict[str, Any]:
    """Exécute une commande Bash et retourne stdout, stderr et le code retour.

    Args:
        command: Commande Bash à exécuter.
        cwd: Répertoire de travail.
        timeout: Délai maximal en secondes.
        sandbox: Conteneur Docker où exécuter la commande (``--docker``) ;
            None = exécution directe sur l'hôte.

    Returns:
        Dictionnaire ``command``, ``ok``, ``returncode``, ``stdout``,
        ``stderr``, ``duration_s`` (+ ``package_manager``/``package`` si
        la commande est une installation).
    """

    logger.info("%s$ %s", "[docker] " if sandbox else "", command)
    start = time.perf_counter()
    installation = parse_install_command(command)
    executed = command

    try:
        if sandbox is not None:
            # Injection : tout appel à pip passe par le venv du conteneur.
            executed = sandbox.inject(command)
            argv = sandbox.wrap(executed, cwd, timeout)
            logger.debug("docker exec: %s", argv[:6])
            # Marge : le timeout réel est appliqué dans le conteneur.
            process = subprocess.run(
                argv,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout + 15,
            )
            if process.returncode in (124, 137):
                process.stderr = (process.stderr or "") + f"\nTimeout après {timeout}s."
        else:
            process = subprocess.run(
                command,
                shell=True,
                cwd=str(cwd),
                executable="/bin/bash",
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )

        result = {
            "command": command,
            "ok": process.returncode == 0,
            "returncode": process.returncode,
            "stdout": compact(process.stdout),
            "stderr": compact(process.stderr),
        }

    except subprocess.TimeoutExpired as exc:
        result = {
            "command": command,
            "ok": False,
            "returncode": -2,
            # Même avec text=True, TimeoutExpired.stdout peut être en bytes.
            "stdout": compact(
                exc.stdout.decode(errors="replace")
                if isinstance(exc.stdout, bytes)
                else exc.stdout or ""
            ),
            "stderr": f"Timeout après {timeout}s.",
        }

    except Exception as exc:
        result = {
            "command": command,
            "ok": False,
            "returncode": -3,
            "stdout": "",
            "stderr": repr(exc),
        }

    if sandbox is not None and executed != command:
        result["executed_command"] = executed

    if installation:
        result["package_manager"] = installation[0]
        result["package"] = installation[1]

    result["duration_s"] = round(time.perf_counter() - start, 2)
    logger.info(
        "  -> code=%s en %.1fs | stdout: %s | stderr: %s",
        result["returncode"],
        result["duration_s"],
        preview(result["stdout"], 150),
        preview(result["stderr"], 150),
    )

    if result["stdout"]:
        logger.debug("stdout complet:\n%s", result["stdout"])

    if result["stderr"]:
        logger.debug("stderr complet:\n%s", result["stderr"])

    return result
