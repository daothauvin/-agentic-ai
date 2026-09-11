"""Gestion des outils générés dynamiquement par l'agent."""

import json
from pathlib import Path
from typing import Any
from .models import GeneratedTool


class ToolManager:
    """Stocke, enregistre et référence les outils générés par l'agent."""

    def __init__(self, project_dir: Path) -> None:
        """Crée les répertoires internes et le registre JSON."""

        self.agent_dir = project_dir / ".agent"
        self.tools_dir = self.agent_dir / "tools"
        self.registry = self.agent_dir / "tools.json"

        self.agent_dir.mkdir(parents=True, exist_ok=True)
        self.tools_dir.mkdir(parents=True, exist_ok=True)

    def installed_registry(self) -> dict[str, Any]:
        """Charge le registre des outils et packages connus."""

        if not self.registry.exists():
            return {}

        try:
            return json.loads(self.registry.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def save(self, data: dict[str, Any]) -> None:
        """Sauvegarde le registre JSON avec une indentation lisible."""

        self.registry.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def register_generated_tool(self, tool: GeneratedTool) -> Path:
        """Écrit un outil Python et l'ajoute au registre local."""

        filename = Path(tool.filename).name

        if not filename.endswith(".py"):
            filename += ".py"

        target = self.tools_dir / filename
        target.write_text(tool.code.rstrip() + "\n", encoding="utf-8")

        registry = self.installed_registry()
        registry[tool.name] = {
            "type": "generated",
            "description": tool.description,
            "path": str(target),
            "packages": tool.packages,
        }
        self.save(registry)

        return target

    def record_packages(self, packages: list[str], manager: str) -> None:
        """Enregistre les packages installés par l'agent."""

        registry = self.installed_registry()

        for package in packages:
            registry[package] = {
                "type": "package",
                "manager": manager,
                "status": "installed",
            }

        self.save(registry)
    def reset(self) -> None: 
        """Supprime tous les outils générés et réinitialise le registre.""" 
        if self.tools_dir.exists(): 
            for path in self.tools_dir.iterdir(): 
                if path.is_file() or path.is_symlink(): 
                    path.unlink()  
            self.save({})