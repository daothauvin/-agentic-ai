"""Modèles Pydantic et état partagé du graphe LangGraph."""

from typing import Any, TypedDict

from pydantic import BaseModel, Field


class GeneratedTool(BaseModel):
    """Décrit un outil que l'agent peut générer lui-même."""

    name: str = Field(description="Nom logique de l'outil.")
    description: str = Field(description="Description de sa responsabilité.")
    filename: str = Field(description="Nom du fichier Python à générer.")
    packages: list[str] = Field(default_factory=list)
    code: str = Field(description="Code source Python exécutable.")
    input: str = Field(description="Entrée du code python.")
    output: str = Field(description="Sortie du code python.")


class Task(BaseModel):
    """Décrit une tâche atomique et vérifiable."""

    id: str = Field(description="Identifiant unique de la tâche.")
    title: str = Field(description="Titre court et lisible.")
    input: str = Field(description="Les ressources nécessaires à la réalisation de la tâche.")
    output: str = Field(description="Sortie attendue.")
    depends_on: list[str] = Field(default_factory=list)
    resolution_depth: int = 0
    
class GeneratedSpeTool(BaseModel):
    """Décrit la spécification d'un outil que l'agent peut générer lui-même."""
    
    name: str = Field(description="Nom logique de l'outil.")
    description: str = Field(description="Description de sa responsabilité.")
    input: str = Field(description="Entrée du code python.")
    output: str = Field(description="Sortie du code python.")


class IDs(BaseModel):
    """Décrit les IDs des nodes du tree."""
    
    node_list: list[str]

class PackageInstall(BaseModel):
    install_commands: list[str] = Field(default_factory=list, description="Commands to install the package")
    pip_install: bool
    apt_get_install: bool
    requires_API_key: bool 
    paid: bool 
    
class PackageUse(BaseModel):
    """Décrit un outil que l'agent peut générer lui-même."""
    utility: bool = Field(description="Utility of the package for the task.")
    python_use: bool = Field(description="If Python Usage.")
    shell_use: bool = Field(description="If Shell Usage.")
    requires_API_key: bool 
    paid: bool 


class Package(BaseModel):
    """Décrit un paquet que l'agent peut installer."""
    title: str = Field(description="Title of the package.")
    url: str = Field(description="Found URL of the package.")
    content: str = Field(description="Small Description of the package.")
    package_use: PackageUse | None = Field(description = "Usage Information of the package")
    package_install: PackageInstall | None = Field(description = "Install Information of the package")
    usage_context: str | None = Field(description = "Context To Use the Package")
    tree: list | None

class ExecutionOutput(BaseModel):
    command: str
    ok: bool
    returncode: int
    stdout: str
    stderr: str




class PlanOutput(BaseModel):
    """Sortie structurée du planificateur principal."""

    tasks: list[Task]


class TaskPlanOutput(BaseModel):
    """Sortie structurée pour préparer l'exécution d'une tâche."""
    commands: list[str] = Field(default_factory=list)
    expected: str

class ToolsSpePlanOutput(BaseModel):
    tools: list[GeneratedSpeTool] = Field(default_factory=list)

class ToolsPlanOutput(BaseModel):
    tools: list[GeneratedTool] = Field(default_factory=list)
    
class PackageResearch(BaseModel):
    """Sortie structurée pour préparer la recherche de package."""

    internet_research: str


class PackageQueries(BaseModel):
    """Plusieurs requêtes de recherche courtes (mots-clés)."""

    queries: list[str] = Field(description="3 short keyword search queries (3 to 6 words each).")


class PackageNames(BaseModel):
    """Noms de packages PyPI proposés par le LLM (vérifiés ensuite sur PyPI)."""

    names: list[str] = Field(description="Exact PyPI package names.")


class PackageRanking(BaseModel):
    """Classement des candidats par pertinence."""

    ranking: list[int] = Field(description="Candidate numbers, most relevant first. Irrelevant candidates are omitted.")


class EvaluationOutput(BaseModel):
    """Résultat de la vérification d'une tâche."""

    success: bool
    reason: str

class CodeOutput(BaseModel):
    """Code généré."""

    code: str

class ToolEvaluationOutput(BaseModel):
    """Résultat de la vérification d'un outil."""

    success: bool
    code: str


class PackageDescription(BaseModel):
    """Description d'un paquet vérifié, rédigée au moment de sa sauvegarde."""

    description: str = Field(description="One or two sentences: what the package does.")
    capabilities: list[str] = Field(
        default_factory=list,
        description="3 to 8 short tasks the package can perform (e.g. 'download youtube video').",
    )


class PackageMatch(BaseModel):
    """Décision : un paquet déjà installé peut-il servir à la tâche ?"""

    # La description vient en premier : le petit modèle décide mieux après l'avoir écrite.
    description: str = Field(default="", description="Short description of what the package does.")
    usable: bool = Field(description="True if the package can perform the task.")
    example: str = Field(default="", description="Shell command using the package for this exact task.")


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
    current_tool: GeneratedTool | None
    tool_success: bool | None
    current_packages: list[Package] | None = Field(default_factory=list)
    eligible_package: bool | None
    installed_package: bool | None
    # Sauvegarde des paquets (voir agent/package_registry.py).
    package_hints: list[dict[str, Any]]
    tool_mode: str | None
    tool_command: str | None
    tool_attempts: int
    tool_retry: bool
    tool_searches: dict[str, int]
    
    
    
