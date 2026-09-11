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
    """

    ollama_url: str
    model: str
    ai_id: int
    workspace_root: Path
    command_timeout: int = 300
    max_tasks: int = 30
    max_resolution_depth: int = 5
    prompts_dir: Path = Path("agent/prompts")

    @property
    def project_dir(self) -> Path:
        """Retourne le workspace dédié à cet agent."""

        return self.workspace_root / f"projet_{self.ai_id}"
