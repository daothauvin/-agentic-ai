"""Registre persistant des paquets installés **et vérifiés** par l'agent.

Un paquet n'entre dans ce registre qu'après avoir passé la boucle
``execute_code_for_package`` -> ``verify_code_package`` : il a été installé,
utilisé (via un outil Python généré ou directement en ligne de commande) et
le vérificateur LLM a validé le résultat.

Le registre sert ensuite à deux endroits :

* ``find_installed_packages`` (avant ``prepare_task``) : retrouve par RAG les
  paquets utiles à la tâche courante pour les décrire au planificateur ;
* ``ToolSearchService.rank_candidates`` : les paquets installés pertinents
  sont ajoutés aux candidats du classement LLM.

Stockage (dans ``<workspace>/.agent/``) :

* ``packages.json`` : une fiche complète par paquet (voir :class:`PackageRecord`) ;
* ``packages_embeddings.json`` : cache des embeddings bge-m3 des fiches,
  invalidé dès que le texte indexé d'une fiche change.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
from pydantic import BaseModel, Field

from .logger import get_logger

logger = get_logger("package_registry")

Embedder = Callable[[list[str]], np.ndarray]


class VerifiedUsage(BaseModel):
    """Une utilisation réelle et validée du paquet.

    Attributes:
        tool_name: Nom de l'outil (spécification ``GeneratedSpeTool``).
        tool_description: Ce que fait l'outil.
        tool_input: Entrée attendue par l'outil.
        tool_output: Sortie produite par l'outil.
        mode: ``"python"`` (outil Python généré) ou ``"cli"`` (paquet utilisé
            directement en ligne de commande, sans code).
        tool_file: Chemin de l'outil Python généré (mode ``python``).
        command: Commande shell exécutée et validée (exemple d'utilisation).
        output_preview: Début de la sortie de la commande validée.
        verified_at: Date ISO de la vérification.
    """

    tool_name: str = ""
    tool_description: str = ""
    tool_input: str = ""
    tool_output: str = ""
    mode: str = "python"
    tool_file: str | None = None
    command: str = ""
    output_preview: str = ""
    verified_at: str = ""


class PackageRecord(BaseModel):
    """Fiche complète d'un paquet vérifié.

    Attributes:
        name: Nom PyPI normalisé (PEP 503).
        title: Nom affiché.
        url: Page PyPI du paquet.
        version: Version installée (PyPI) si connue.
        summary: Résumé PyPI.
        description: Description courte rédigée par le LLM.
        capabilities: Mots-clés / tâches que le paquet sait faire.
        keywords: Mots-clés PyPI.
        homepage: Page d'accueil du projet.
        manager: Gestionnaire d'installation (``pip``, ``apt``...).
        install_commands: Commandes d'installation qui ont fonctionné.
        python_use: Utilisable en Python.
        shell_use: Utilisable en ligne de commande.
        requires_api_key: Clé d'API nécessaire.
        paid: Usage payant.
        usage_context: Extraits de documentation (RAG) sur l'usage.
        usages: Utilisations vérifiées (la plus récente en dernier).
        registered_at: Date ISO de la première sauvegarde.
        updated_at: Date ISO de la dernière mise à jour.
    """

    name: str
    title: str = ""
    url: str = ""
    version: str = ""
    summary: str = ""
    description: str = ""
    capabilities: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    homepage: str = ""
    manager: str = "pip"
    install_commands: list[str] = Field(default_factory=list)
    python_use: bool = False
    shell_use: bool = False
    requires_api_key: bool = False
    paid: bool = False
    usage_context: str = ""
    usages: list[VerifiedUsage] = Field(default_factory=list)
    registered_at: str = ""
    updated_at: str = ""

    def index_text(self) -> str:
        """Texte indexé par le RAG (nom, descriptions, capacités, outils vérifiés).

        Returns:
            Texte court et dense décrivant ce que sait faire le paquet.
        """

        parts = [f"{self.name}: {self.description or self.summary}"]
        if self.summary and self.summary != self.description:
            parts.append(self.summary)
        if self.capabilities:
            parts.append("Capabilities: " + ", ".join(self.capabilities))
        for usage in self.usages:
            parts.append(f"Verified tool: {usage.tool_description}".strip())
        return "\n".join(parts)

    def prompt_card(self, max_examples: int = 2) -> dict:
        """Résumé compact de la fiche pour un prompt LLM.

        Args:
            max_examples: Nombre maximal d'exemples vérifiés repris.

        Returns:
            Dictionnaire sérialisable en JSON.
        """

        return {
            "name": self.name,
            "description": self.description or self.summary,
            "capabilities": self.capabilities,
            "mode": "cli" if self.shell_use else "python",
            "verified_examples": [
                {
                    "tool": usage.tool_description,
                    "input": usage.tool_input,
                    "output": usage.tool_output,
                    "tool_file": usage.tool_file,
                    "command": usage.command,
                }
                for usage in self.usages[-max_examples:]
            ],
        }


def _now() -> str:
    """Date courante au format ISO (seconde).

    Returns:
        Date ISO.
    """

    return datetime.now().isoformat(timespec="seconds")


def _hash(text: str) -> str:
    """Empreinte courte d'un texte (invalidation du cache d'embeddings).

    Args:
        text: Texte à hacher.

    Returns:
        Empreinte SHA-1 hexadécimale.
    """

    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class PackageRegistry:
    """Registre JSON des paquets vérifiés, interrogeable par similarité.

    Args:
        agent_dir: Répertoire ``.agent`` du workspace.
        embedder: Fonction texte -> embeddings normalisés L2 (bge-m3). Si
            ``None``, la recherche retombe sur un score lexical.
    """

    def __init__(self, agent_dir: Path, embedder: Embedder | None = None) -> None:
        self.agent_dir = agent_dir
        self.path = agent_dir / "packages.json"
        self.vectors_path = agent_dir / "packages_embeddings.json"
        self.embedder = embedder
        self.agent_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Lecture / écriture
    # ------------------------------------------------------------------

    def load(self) -> dict[str, PackageRecord]:
        """Charge toutes les fiches.

        Returns:
            Dictionnaire nom -> fiche (vide si le fichier est absent ou invalide).
        """

        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return {name: PackageRecord.model_validate(data) for name, data in raw.items()}
        except (json.JSONDecodeError, ValueError) as error:
            logger.error("Registre de paquets illisible (%s): %s", self.path, error)
            return {}

    def save_all(self, records: dict[str, PackageRecord]) -> None:
        """Écrit toutes les fiches sur disque.

        Args:
            records: Dictionnaire nom -> fiche.
        """

        self.path.write_text(
            json.dumps(
                {name: record.model_dump(mode="json") for name, record in records.items()},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def get(self, name: str) -> PackageRecord | None:
        """Retourne la fiche d'un paquet.

        Args:
            name: Nom du paquet.

        Returns:
            La fiche, ou ``None`` si le paquet n'est pas enregistré.
        """

        return self.load().get(name)

    def names(self) -> list[str]:
        """Noms des paquets enregistrés.

        Returns:
            Liste des noms.
        """

        return list(self.load())

    def add(self, record: PackageRecord) -> PackageRecord:
        """Ajoute ou met à jour un paquet (les usages vérifiés sont cumulés).

        Args:
            record: Fiche à enregistrer (en général avec un seul usage).

        Returns:
            La fiche finale enregistrée.
        """

        records = self.load()
        previous = records.get(record.name)
        now = _now()
        if previous is not None:
            merged = previous.model_copy(update={
                key: value
                for key, value in record.model_dump().items()
                if value not in ("", [], None) and key not in ("usages", "registered_at")
            })
            known = {(u.tool_name, u.command) for u in previous.usages}
            merged.usages = previous.usages + [
                u for u in record.usages if (u.tool_name, u.command) not in known
            ]
            merged.capabilities = list(dict.fromkeys(previous.capabilities + record.capabilities))
            record = merged
            logger.info("Paquet mis à jour dans le registre: %s (%d usage(s))",
                        record.name, len(record.usages))
        else:
            record.registered_at = record.registered_at or now
            logger.info("Nouveau paquet enregistré: %s", record.name)
        record.updated_at = now
        records[record.name] = record
        self.save_all(records)
        logger.debug("Fiche enregistrée:\n%s", record.model_dump_json(indent=2))
        return record

    def remove(self, name: str) -> bool:
        """Supprime un paquet du registre.

        Args:
            name: Nom du paquet.

        Returns:
            ``True`` si le paquet existait.
        """

        records = self.load()
        if records.pop(name, None) is None:
            return False
        self.save_all(records)
        return True

    def reset(self) -> None:
        """Vide le registre et son cache d'embeddings."""

        self.save_all({})
        if self.vectors_path.exists():
            self.vectors_path.unlink()

    # ------------------------------------------------------------------
    # Recherche
    # ------------------------------------------------------------------

    def _vectors(self, records: dict[str, PackageRecord]) -> dict[str, np.ndarray]:
        """Embeddings des fiches, recalculés seulement si leur texte a changé.

        Args:
            records: Fiches à encoder.

        Returns:
            Dictionnaire nom -> vecteur normalisé.
        """

        cache: dict = {}
        if self.vectors_path.exists():
            try:
                cache = json.loads(self.vectors_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                cache = {}

        texts = {name: record.index_text() for name, record in records.items()}
        stale = [name for name, text in texts.items()
                 if cache.get(name, {}).get("hash") != _hash(text)]
        if stale:
            logger.debug("Embeddings du registre à (re)calculer: %s", stale)
            vectors = self.embedder([texts[name] for name in stale])
            for name, vector in zip(stale, vectors):
                cache[name] = {"hash": _hash(texts[name]), "vector": vector.tolist()}
            cache = {name: value for name, value in cache.items() if name in records}
            self.vectors_path.write_text(json.dumps(cache), encoding="utf-8")
        return {name: np.asarray(cache[name]["vector"], dtype="float32") for name in records}

    @staticmethod
    def _lexical_score(query: str, text: str) -> float:
        """Score de repli sans embeddings : proportion de mots de la requête présents.

        Args:
            query: Requête.
            text: Texte de la fiche.

        Returns:
            Score entre 0 et 1.
        """

        words = {w for w in query.lower().split() if len(w) > 2}
        if not words:
            return 0.0
        haystack = text.lower()
        return sum(w in haystack for w in words) / len(words)

    def search(self, query: str, k: int = 3, min_score: float = 0.0) -> list[tuple[PackageRecord, float]]:
        """Paquets du registre les plus proches d'une requête.

        Args:
            query: Description de la tâche ou de l'outil recherché.
            k: Nombre maximal de résultats.
            min_score: Similarité cosinus minimale (ou score lexical en repli).

        Returns:
            Liste (fiche, score) triée par score décroissant.
        """

        records = self.load()
        if not records or not query.strip():
            return []

        if self.embedder is None:
            scored = [(r, self._lexical_score(query, r.index_text())) for r in records.values()]
        else:
            try:
                vectors = self._vectors(records)
                query_vector = self.embedder([query])[0]
                scored = [(records[name], float(vector @ query_vector))
                          for name, vector in vectors.items()]
            except Exception as error:  # Ollama injoignable : on ne bloque pas l'agent.
                logger.warning("Embeddings du registre indisponibles (%s) : score lexical", error)
                scored = [(r, self._lexical_score(query, r.index_text())) for r in records.values()]

        scored.sort(key=lambda item: -item[1])
        logger.debug("Registre — scores pour %r: %s", query,
                     [(r.name, round(s, 3)) for r, s in scored[:max(k, 5)]])
        return [(record, score) for record, score in scored[:k] if score >= min_score]
