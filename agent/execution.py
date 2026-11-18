"""Exécution contrôlée des commandes Linux."""

import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from .logger import get_logger, preview

logger = get_logger("shell")


def compact(text: str, limit: int = 16000) -> str:
    """Limite la taille d'un texte avant stockage ou transmission au LLM."""

    text = text or ""

    if len(text) <= limit:
        return text

    return text[:limit] + "\n...[tronqué]..."


def parse_install_command(command: str) -> tuple[str, list[str]] | None:
    """Détecte un gestionnaire de paquets dans une commande simple."""

    parts = shlex.split(command)

    if not parts:
        return None

    if parts[0] in ("pip", "pip3") and len(parts) >= 3:
        if parts[1] == "install":
            return "pip", [x for x in parts[2:] if not x.startswith("-")]

    if parts[0] in ("apt", "apt-get") and len(parts) >= 3:
        if parts[1] == "install":
            return "apt", [x for x in parts[2:] if not x.startswith("-")]

    if len(parts) >= 4 and parts[0] == "sudo":
        if parts[1] in ("apt", "apt-get") and parts[2] == "install":
            return "apt", [x for x in parts[3:] if not x.startswith("-")]

    if parts[0] == "npm" and len(parts) >= 3:
        if parts[1] == "install":
            return "npm", [x for x in parts[2:] if not x.startswith("-")]

    if parts[0] == "cargo" and len(parts) >= 3:
        if parts[1] == "install":
            return "cargo", [x for x in parts[2:] if not x.startswith("-")]

    return None


def run_command(command: str, cwd: Path, timeout: int) -> dict[str, Any]:
    """Exécute une commande Bash et retourne stdout, stderr et le code retour."""

    logger.info("$ %s", command)
    start = time.perf_counter()
    installation = parse_install_command(command)

    try:
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
