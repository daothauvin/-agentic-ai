"""Interface en ligne de commande de l'agent."""

import argparse
import os
from pathlib import Path

from .config import AgentConfig
from .graph import AgentGraph
from dotenv import load_dotenv
from .logger import get_logger, setup_logging

logger = get_logger("cli")

def build_parser() -> argparse.ArgumentParser:
    """Construit le parser CLI et documente toutes les options."""

    parser = argparse.ArgumentParser(
        description="Agent Linux autonome basé sur LangChain + LangGraph."
    )
    parser.add_argument("request", nargs="?", help="Demande à exécuter.")
    parser.add_argument("--id", type=int, default=None, help="Identifiant de l'IA.")
    parser.add_argument( "--reset", action="store_true", help="Réinitialisation de l'IA avant utilisation." )
    parser.add_argument( "--savegraph", nargs="?", const="graph.png", default=None, help="Sauvegarde le graphe dans un fichier. Nom du fichier optionnel." )

    parser.add_argument(
        "--model",
        default=os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),
        help="Modèle Ollama utilisé par LangChain.",
    )
    parser.add_argument(
        "--workspace",
        default=os.getenv("AI_WORKSPACE", "AI_Workspace"),
        help="Racine des workspaces des agents.",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"),
        help="URL du serveur Ollama.",
    )
    parser.add_argument(
        "--prompts-dir",
        default=os.getenv("AI_PROMPTS_DIR", "agent/prompts"),
        help="Répertoire contenant les prompts texte.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout maximal d'une commande shell.",
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Reprend l'exécution depuis ce checkpoint_id (thread = --id). La demande est alors ignorée.",
    )

    parser.add_argument(
        "--log-level",
        default=os.getenv("AI_LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Niveau de log console (le fichier de log est toujours en DEBUG).",
    )

    return parser


def main() -> int:
    """Parse les arguments et démarre l'agent."""

    parser = build_parser()
    args = parser.parse_args()
    load_dotenv()

    ai_id = args.id

    if ai_id is None:
        ai_id = int(input("Numéro de l'IA [1] > ").strip() or "1")

    if ai_id < 1:
        print("Le numéro d'IA doit être >= 1.")
        return 2

    request = args.request or ("" if args.checkpoint else input("Demande utilisateur > ").strip())

    if not request and not args.checkpoint:
        print("Demande vide.")
        return 2

    config = AgentConfig(
        ollama_url=args.ollama_url,
        model=args.model,
        ai_id=ai_id,
        workspace_root=Path(args.workspace).expanduser().resolve(),
        command_timeout=args.timeout,
        prompts_dir=Path(args.prompts_dir).expanduser().resolve()
    )

    log_file = setup_logging(config.project_dir / ".agent" / "logs", args.log_level)
    logger.info("Logs complets (DEBUG) : %s", log_file)
    logger.info("Logs RAG (DEBUG) : %s", log_file.with_name(log_file.name.replace("run_", "rag_", 1)))

    try:
        model = AgentGraph(config)
        if args.reset:
            model.tools.reset()

        if args.savegraph:
            img = model.graph.get_graph().draw_mermaid_png()
            with open("graph.png", "wb") as png:
                png.write(img)
            logger.info("Graphe sauvegardé")
        from typing import Any, TypedDict
        class AgentState(TypedDict, total=False):
            """État transmis entre les nœuds du graphe LangGraph."""

            request: str
            environment: dict[str, Any]
            queue: list[dict[str, Any]]
            completed: dict[str, Any]
            current_task: dict[str, Any] | None
            current_plan: dict[str, Any]
            execution: list[dict[str, Any]]
            evaluation: dict[str, Any]
            resolution_plan: dict[str, Any]
            generated_tools: list[dict[str, Any]]
            final_success: bool
            fatal_error: str | None
        #state = AgentState()
        #state["current_task"] = {'id': 'user_request_image_download_1', 'title': 'Rechercher une image sur internet', 'input': "Recherche d'une image", 'output': "URL d'une image", 'depends_on': [], 'resolution_depth': 0}
        # {'id': 'user_request_image_download_1', 'title': 'Rechercher une image de chien sur internet', 'input': "Recherche d'une image de chien", 'output': "URL d'une image de chien", 'depends_on': [], 'resolution_depth': 0}
        #model.search_and_install_package(state)
        #raise

        
        return model.run(request, checkpoint_id=args.checkpoint)
    except KeyboardInterrupt:
        logger.warning("Exécution interrompue par l'utilisateur.")
        return 130
    except Exception as exc:
        logger.exception(
            "ERREUR: %s — Vérifie Ollama, le modèle et les dépendances Python.",
            exc,
        )
        return 4
