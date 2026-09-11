"""Modèles Pydantic et état partagé du graphe LangGraph."""

from typing import Any, TypedDict

from pydantic import BaseModel, Field


class Task(BaseModel):
    """Décrit une tâche atomique et vérifiable."""

    id: str = Field(description="Identifiant unique de la tâche.")
    title: str = Field(description="Titre court et lisible.")
    input: str = Field(description="Les ressources nécessaires à la réalisation de la tâche.")
    output: str = Field(description="Sortie attendue.")
    depends_on: list[str] = Field(default_factory=list)
    resolution_depth: int = 0


class GeneratedTool(BaseModel):
    """Décrit un outil que l'agent peut générer lui-même."""

    name: str = Field(description="Nom logique de l'outil.")
    description: str = Field(description="Description de sa responsabilité.")
    filename: str = Field(description="Nom du fichier Python à générer.")
    install_manager: str | None = Field(
        default=None,
        description="Gestionnaire de paquets: apt, pip, npm ou cargo.",
    )
    packages: list[str] = Field(default_factory=list)
    code: str = Field(description="Code source Python exécutable.")


class PlanOutput(BaseModel):
    """Sortie structurée du planificateur principal."""

    tasks: list[Task]


class TaskPlanOutput(BaseModel):
    """Sortie structurée pour préparer l'exécution d'une tâche."""

    tools: list[GeneratedTool] = Field(default_factory=list)
    commands: list[str] = Field(default_factory=list)
    expected: str


class EvaluationOutput(BaseModel):
    """Résultat de la vérification d'une tâche."""

    success: bool
    reason: str


class ResolutionOutput(BaseModel):
    """Plan de réparation produit après un échec."""

    resolvable: bool
    reason: str
    tasks: list[Task] = Field(default_factory=list)


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
