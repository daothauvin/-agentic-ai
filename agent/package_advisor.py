"""Exploitation du registre de paquets vérifiés par le LLM.

* :meth:`PackageAdvisor.describe` : rédige la description d'un paquet au
  moment de sa sauvegarde (``register_tools``) ;
* :meth:`PackageAdvisor.hints_for_task` : avant ``prepare_task``, retrouve par
  RAG les paquets installés proches de la tâche, puis demande au LLM, paquet par
  paquet, s'il convient et avec quelle commande l'utiliser ;
* :meth:`PackageAdvisor.format_hints` : bloc de texte injecté dans le prompt
  du planificateur de tâche.
"""

from __future__ import annotations

import json
from typing import Any

from .logger import get_logger, preview
from .models import PackageDescription, PackageMatch
from .package_registry import PackageRecord, PackageRegistry

logger = get_logger("package_advisor")


def task_query(task: dict[str, Any] | None) -> str:
    """Requête RAG construite depuis une tâche du planificateur.

    Args:
        task: Tâche (``title``, ``input``, ``output``).

    Returns:
        Texte de requête (vide si pas de tâche).
    """

    task = task or {}
    parts = [task.get("title", "")]
    if task.get("input"):
        parts.append(f"Input: {task['input']}")
    if task.get("output"):
        parts.append(f"Output: {task['output']}")
    return ". ".join(p for p in parts if p)


def tool_query(tool_spec: dict[str, Any] | None) -> str:
    """Requête RAG construite depuis une spécification d'outil.

    Args:
        tool_spec: Spécification ``GeneratedSpeTool`` (dict).

    Returns:
        Texte de requête.
    """

    tool_spec = tool_spec or {}
    parts = [tool_spec.get("description", "") or tool_spec.get("name", "")]
    if tool_spec.get("input"):
        parts.append(f"Input: {tool_spec['input']}")
    if tool_spec.get("output"):
        parts.append(f"Output: {tool_spec['output']}")
    return ". ".join(p for p in parts if p)


class PackageAdvisor:
    """Fait le lien entre le registre de paquets et le LLM.

    Args:
        llm: Adaptateur ``LocalLLM``.
        prompts: Chargeur de prompts.
        registry: Registre des paquets vérifiés.
        config: Configuration de l'agent.
    """

    def __init__(self, llm, prompts, registry: PackageRegistry, config) -> None:
        self.llm = llm
        self.prompts = prompts
        self.registry = registry
        self.config = config
        #: Détail de la dernière recherche (candidats, scores, décisions) pour les benchs.
        self.last_search: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Sauvegarde
    # ------------------------------------------------------------------

    def describe(self, record: PackageRecord) -> PackageDescription:
        """Rédige la description et les capacités d'un paquet vérifié.

        Args:
            record: Fiche du paquet (métadonnées, doc, usage vérifié).

        Returns:
            Description structurée. En cas d'échec du LLM, le résumé PyPI.
        """

        usage = record.usages[-1] if record.usages else None
        payload = {
            "name": record.name,
            "summary": record.summary,
            "keywords": record.keywords,
            "verified_tool": usage.model_dump() if usage else None,
            "documentation": (record.usage_context or "")[:3000],
        }
        try:
            result = self.llm.structured(
                PackageDescription,
                self.prompts.load("package_describer"),
                json.dumps(payload, ensure_ascii=False, indent=2),
            )
        except Exception as error:
            logger.warning("Description LLM impossible pour %s: %s", record.name, error)
            return PackageDescription(description=record.summary, capabilities=[])
        logger.info("Description de %s: %s | %s", record.name, result.description, result.capabilities)
        return result

    # ------------------------------------------------------------------
    # Recherche pour une tâche
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: int | None = None,
                 min_score: float | None = None) -> list[tuple[PackageRecord, float]]:
        """Paquets du registre proches d'une requête (RAG seul, sans LLM).

        Args:
            query: Texte de la tâche / de l'outil.
            k: Nombre de résultats (défaut ``config.registry_search_k``).
            min_score: Score minimal (défaut ``config.registry_min_score``).

        Returns:
            Liste (fiche, score).
        """

        return self.registry.search(
            query,
            k=self.config.registry_search_k if k is None else k,
            min_score=self.config.registry_min_score if min_score is None else min_score,
        )

    def match(self, task: dict[str, Any], candidates: list[PackageRecord]) -> list[PackageMatch]:
        """Demande au LLM, pour chaque candidat, s'il convient à la tâche.

        Args:
            task: Tâche courante.
            candidates: Fiches candidates.

        Returns:
            Une décision par candidat (même ordre).
        """

        if not candidates:
            return []
        user_task = {key: task.get(key) for key in ("title", "input", "output") if task.get(key)}
        prompts = [
            f"TACHE:\n{json.dumps(user_task, ensure_ascii=False, indent=2)}\n\n"
            f"PAQUET:\n{json.dumps(record.prompt_card(), ensure_ascii=False, indent=2)}"
            for record in candidates
        ]
        for prompt in prompts:
            logger.debug("USER PROMPT (package_matcher):\n%s", prompt)
        try:
            return self.llm.structured_batch(PackageMatch, self.prompts.load("package_matcher"), prompts)
        except Exception as error:
            logger.warning("Vérification LLM des paquets installés impossible: %s", error)
            return [PackageMatch(usable=False) for _ in candidates]

    def hints_for_task(self, task: dict[str, Any] | None) -> list[dict[str, Any]]:
        """Paquets installés utilisables pour la tâche, avec description et exemple.

        Args:
            task: Tâche courante.

        Returns:
            Liste de dictionnaires ``name``, ``score``, ``description``,
            ``example``, ``mode``, ``tool_file`` (vide si aucun ne convient).
        """

        query = task_query(task)
        found = self.retrieve(query)
        self.last_search = {"query": query, "retrieved": [(r.name, round(s, 3)) for r, s in found]}
        logger.info("Paquets installés proches de la tâche: %s", self.last_search["retrieved"])
        if not found:
            self.last_search["decisions"] = []
            return []

        decisions = self.match(task or {}, [record for record, _ in found])
        hints = []
        for (record, score), decision in zip(found, decisions):
            logger.info("Paquet %s (score %.3f) utilisable=%s | %s",
                        record.name, score, decision.usable, preview(decision.example, 200))
            if not decision.usable:
                continue
            usage = record.usages[-1] if record.usages else None
            hints.append({
                "name": record.name,
                "score": round(score, 3),
                "description": decision.description or record.description or record.summary,
                "example": decision.example or (usage.command if usage else ""),
                "mode": "cli" if record.shell_use else "python",
                "tool_file": usage.tool_file if usage else None,
            })
        self.last_search["decisions"] = [
            {"name": record.name, **decision.model_dump()} for (record, _), decision in zip(found, decisions)
        ]
        return hints

    @staticmethod
    def format_hints(hints: list[dict[str, Any]]) -> str:
        """Bloc de prompt décrivant les paquets utilisables.

        Args:
            hints: Sortie de :meth:`hints_for_task`.

        Returns:
            Texte pour le prompt du planificateur (``Aucun`` si la liste est vide).
        """

        if not hints:
            return "Aucun"
        lines = []
        for hint in hints:
            lines.append(f"- {hint['name']} : {hint['description']}")
            if hint.get("tool_file"):
                lines.append(f"  Outil Python déjà écrit : {hint['tool_file']}")
            if hint.get("example"):
                lines.append(f"  Exemple pour cette tâche : {hint['example']}")
        return "\n".join(lines)
