"""Interface en ligne de commande de l'agent."""

import argparse
import os
from pathlib import Path

from .config import AgentConfig
from .graph import AgentGraph


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

    return parser


def main() -> int:
    """Parse les arguments et démarre l'agent."""

    parser = build_parser()
    args = parser.parse_args()
    

    ai_id = args.id

    if ai_id is None:
        ai_id = int(input("Numéro de l'IA [1] > ").strip() or "1")

    if ai_id < 1:
        print("Le numéro d'IA doit être >= 1.")
        return 2

    request = args.request or input("Demande utilisateur > ").strip()

    if not request:
        print("Demande vide.")
        return 2

    config = AgentConfig(
        ollama_url=args.ollama_url,
        model=args.model,
        ai_id=ai_id,
        workspace_root=Path(args.workspace).expanduser().resolve(),
        command_timeout=args.timeout,
        prompts_dir=Path(args.prompts_dir).expanduser().resolve(),
    )

    try:
        model = AgentGraph(config)
        if args.reset:
            model.tools.reset()

        if args.savegraph:
            img = model.graph.get_graph().draw_mermaid_png()
            with open("graph.png", "wb") as png:
                png.write(img)
            print("INFO: Graph Saved")
        return model.run(request)
    except Exception as exc:
        print(
            f"\nERREUR: {exc}\n"
            "Vérifie Ollama, le modèle et les dépendances Python."
        )
        return 4
