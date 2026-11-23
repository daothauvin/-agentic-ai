"""Configuration du logging et outils de chronométrage de l'agent.

Les logs sont écrits à la fois sur la console et dans un fichier du workspace.
Chaque enregistrement est vidé immédiatement sur disque : même si l'exécution
est interrompue (timeout, Ctrl+C, crash), le fichier contient tout ce qui
s'est passé jusque-là.
"""

import functools
import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

LOGGER_NAME = "agent"
FORMAT = "%(asctime)s.%(msecs)03d %(levelname)-7s [%(name)s] %(message)s"
DATEFMT = "%H:%M:%S"


def get_logger(name: str | None = None) -> logging.Logger:
    """Retourne le logger de l'agent ou un de ses sous-loggers."""

    return logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)


class _FlushingFileHandler(logging.FileHandler):
    """FileHandler qui force l'écriture sur disque après chaque log."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


def setup_logging(log_dir: Path, console_level: str = "INFO") -> Path:
    """Configure la console (niveau choisi), un fichier complet et un fichier RAG (DEBUG).

    Returns:
        Chemin du fichier de log de cette exécution.
    """

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}.log"

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(FORMAT, DATEFMT)

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(console_level.upper())
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = _FlushingFileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Fichier dédié au RAG (chunking, embeddings, retrieval), toujours en DEBUG.
    rag_file = log_dir / log_file.name.replace("run_", "rag_", 1)
    rag_handler = _FlushingFileHandler(rag_file, encoding="utf-8")
    rag_handler.setLevel(logging.DEBUG)
    rag_handler.setFormatter(formatter)
    rag_handler.addFilter(logging.Filter(f"{LOGGER_NAME}.rag"))
    logger.addHandler(rag_handler)

    # Liens stables vers les derniers logs pour un `tail -f` facile.
    for name, target in (("latest.log", log_file), ("latest_rag.log", rag_file)):
        latest = log_dir / name
        try:
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            latest.symlink_to(target.name)
        except OSError:
            pass

    return log_file


def preview(value: Any, limit: int = 300) -> str:
    """Résumé court d'une valeur pour les logs INFO."""

    text = str(value).replace("\n", " ⏎ ")
    return text if len(text) <= limit else text[:limit] + f"… [{len(text)} car.]"


def log_codegen(
    logger: logging.Logger,
    stage: str,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
) -> None:
    """Logue en DEBUG l'entrée et la sortie complètes d'une génération de code.

    Le bloc est délimité par des bannières ``CODEGEN <stage>`` faciles à
    retrouver (``grep -n CODEGEN latest.log``).

    Args:
        logger: Logger à utiliser.
        stage: Nom de l'étape (ex. ``construct_code_for_package``).
        inputs: Entrées envoyées au LLM (prompt système, prompt utilisateur…).
        outputs: Sorties (réponse brute, code nettoyé, chemin du fichier…).
    """

    lines = [f"===== CODEGEN {stage} — ENTRÉE ====="]
    for key, value in inputs.items():
        lines.append(f"--- {key} ---\n{value}")
    lines.append(f"===== CODEGEN {stage} — SORTIE =====")
    for key, value in outputs.items():
        lines.append(f"--- {key} ---\n{value}")
    lines.append(f"===== FIN CODEGEN {stage} =====")
    logger.debug("\n".join(lines))


@contextmanager
def timed(label: str, logger: logging.Logger | None = None) -> Iterator[None]:
    """Chronomètre un bloc de code et logue sa durée."""

    logger = logger or get_logger("timing")
    logger.info("▶ %s", label)
    start = time.perf_counter()
    try:
        yield
    except BaseException:
        logger.error("✖ %s échoue après %.1fs", label, time.perf_counter() - start)
        raise
    logger.info("✔ %s terminé en %.1fs", label, time.perf_counter() - start)


def timed_node(func: Callable) -> Callable:
    """Décorateur pour les nœuds LangGraph : logue le début, la fin et la durée."""

    logger = get_logger("node")

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with timed(f"NŒUD {func.__name__}", logger):
            return func(*args, **kwargs)

    return wrapper
