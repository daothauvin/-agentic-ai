"""Chargement des prompts depuis des fichiers texte."""

from pathlib import Path


class PromptLoader:
    """Charge et met en cache les prompts externes de l'agent."""

    def __init__(self, prompts_dir: Path) -> None:
        """Initialise le chargeur avec le répertoire des prompts."""

        self.prompts_dir = prompts_dir
        self._cache: dict[str, str] = {}

    def load(self, name: str) -> str:
        """Charge un prompt `.txt` par son nom logique.

        Args:
            name: Nom du fichier sans extension.

        Returns:
            Contenu UTF-8 nettoyé du prompt.

        Raises:
            FileNotFoundError: Si le prompt demandé n'existe pas.
        """

        if name not in self._cache:
            path = self.prompts_dir / f"{name}.txt"
            self._cache[name] = path.read_text(encoding="utf-8").strip()

        return self._cache[name]

    def render(self, name: str, **values: object) -> str:
        """Charge un prompt puis remplace ses variables `{nom}`."""
        print(self.load(name))
        return self.load(name).format(**values)
