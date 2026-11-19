"""Configuration centrale de l'agent."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AgentConfig:
    """Configuration immuable de l'agent Linux.

    Attributes:
        ollama_url: URL du serveur Ollama.
        model: Nom du modèle local utilisé par LangChain.
        ai_id: Identifiant de l'agent et de son workspace.
        workspace_root: Répertoire racine des workspaces.
        command_timeout: Timeout maximal d'une commande shell.
        max_tasks: Nombre maximal de tâches dans un plan.
        max_resolution_depth: Nombre maximal de cycles de réparation.
        prompts_dir: Répertoire contenant les prompts externes.
        max_packages_analyzed: Limite temporaire du nombre de packages analysés.
        rag_multi_level: RAG multi-niveau (sections + phrases), voir agent/rag.py.
        rag_install_k: Nombre de chunks donnés au LLM pour l'installation.
        rag_usage_k: Nombre de chunks par requête d'usage (3 requêtes).
    """

    ollama_url: str
    model: str
    ai_id: int
    workspace_root: Path
    command_timeout: int = 300
    max_tasks: int = 30
    max_resolution_depth: int = 5
    # LIMITE TEMPORAIRE : nombre max de packages PyPI analysés par recherche.
    max_packages_analyzed: int = 3
    prompts_dir: Path = Path("agent/prompts")
    searxng_url: str = "http://localhost:8080/search"
    # Moteurs SearXNG (séparés par des virgules), vide = moteurs par défaut de l'instance.
    # DuckDuckGo, Brave, Google, Qwant se font bloquer (captcha / 403 / 429), Yahoo
    # renvoie une « parsing error » (SearXNG 2026.9) et Bing ignore `site:pypi.org`.
    # Plusieurs moteurs respectant `site:` : si l'un est suspendu (429), les autres répondent.
    searxng_engines: str = "startpage,presearch,yandex,seznam,marginalia"
    # Options de la recherche de packages (voir AMELIORATIONS.md, section 9).
    search_multi_query: bool = True       # 1. plusieurs requêtes courtes
    search_name_guess: bool = True        # 2. noms proposés par le LLM, vérifiés sur PyPI
    search_prerank: bool = True           # 3. classement LLM sur nom + résumé avant indexation
    search_metadata_filter: bool = True   # 4. filtre sur les métadonnées PyPI
    search_min_monthly_downloads: int = 5000  # 4. seuil de popularité (pypistats)
    search_normalize: bool = True         # 5. URL normalisées + dédoublonnage
    # RAG : indexe aussi chaque phrase / ligne de code (mesuré : install 81 % -> 100 %).
    rag_multi_level: bool = True
    # top-1 : commande extraite 16/16 contre 15/16 en top-2, 7/16 en top-5 (bench/rag_bench.py --k).
    rag_install_k: int = 1
    rag_usage_k: int = 3
    search_use_request: bool = False      # 7. demande utilisateur transmise à la recherche (sans gain mesuré)
    
    @property
    def project_dir(self) -> Path:
        """Retourne le workspace dédié à cet agent."""

        return self.workspace_root / f"projet_{self.ai_id}"
