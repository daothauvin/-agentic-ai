"""Construction du graphe LangGraph et logique des nœuds."""

import json
import os
import sys
from pathlib import Path
from typing import Any
from PIL import Image

from langgraph.graph import END, START, StateGraph
from .config import AgentConfig
from .execution import compact, run_command
from .llm import LocalLLM
from .models import (
    AgentState,
    EvaluationOutput,
    GeneratedTool,
    PlanOutput,
    ResolutionOutput,
    TaskPlanOutput,
)
from .prompts import PromptLoader
from .tools import ToolManager


MERMAID_ARCHITECTURE = """graph TD
    START([START]) --> INSPECT[inspect_environment]
    INSPECT --> PLAN[plan]
    PLAN --> SELECT[select_task]
    SELECT -->|tâche prête| PREPARE[prepare_task]
    SELECT -->|plus de tâche| FINISH[finish]
    PREPARE --> INSTALL[install_and_register_tools]
    INSTALL --> EXECUTE[execute_commands]
    EXECUTE --> EVALUATE[evaluate]
    EVALUATE -->|success| RECORD[record_success]
    RECORD --> SELECT
    EVALUATE -->|failure| RESOLVE[resolve_failure]
    RESOLVE -->|resolvable| SELECT
    RESOLVE -->|not resolvable| FINISH
    FINISH --> END([END])
"""


class AgentGraph:
    """Implémente les nœuds métier et le StateGraph de l'agent."""

    def __init__(self, config: AgentConfig) -> None:
        """Initialise le LLM, le gestionnaire d'outils et les prompts."""

        self.config = config
        self.llm = LocalLLM(config)
        self.tools = ToolManager(config.project_dir)
        
        self.prompts = PromptLoader(config.prompts_dir)
        self.graph = self._build_graph()
        

    def environment(self) -> dict[str, Any]:
        """Collecte les informations système utiles à la planification."""

        uname = run_command(
            "uname -a",
            self.config.project_dir,
            20,
        )

        return {
            "cwd": str(self.config.project_dir),
            "os": compact(uname["stdout"], 2000),
            "python": sys.version.split()[0],
            "user": os.getenv("USER", ""),
            "ai_id": self.config.ai_id,
            "packages_and_tools": self.tools.installed_registry(),
        }

    def inspect_environment(self, state: AgentState) -> dict[str, Any]:
        """Nœud initial: inspecte le workspace et le système."""

        return {"environment": self.environment()}

    def plan(self, state: AgentState) -> dict[str, Any]:
        """Transforme la demande utilisateur en tâches atomiques."""

        planner_prompt = self.prompts.render(
            "planner",
            environment=json.dumps(
                state["environment"],
                ensure_ascii=False,
                indent=2,
            ),
        )
        
        user_prompt = state["request"]

        result = self.llm.structured(
            PlanOutput,
            planner_prompt,
            user_prompt,
        )

        tasks = [
            task.model_dump()
            for task in result.tasks[: self.config.max_tasks]
        ]

        if not tasks:
            raise RuntimeError("Le planificateur n'a créé aucune tâche.")

        return {
            "queue": tasks,
            "execution": [],
        }

    def select_task(self, state: AgentState) -> dict[str, Any]:
        """Sélectionne la première tâche dont les dépendances sont satisfaites."""

        queue = list(state.get("queue", []))
        completed = state.get("completed", {})
        current = None

        for index, task in enumerate(queue):
            dependencies = task.get("depends_on", [])

            if all(dependency in completed for dependency in dependencies):
                current = queue.pop(index)
                break

        if current is None and queue:
            return {
                "fatal_error": "Dépendances circulaires ou invalides.",
                "queue": [],
                "current_task": None,
            }

        return {
            "queue": queue,
            "current_task": current,
            "execution": [],
        }

    def prepare_task(self, state: AgentState) -> dict[str, Any]:
        """Demande au LLM un plan d'exécution et les outils manquants."""

        task = state["current_task"]

        if task is None:
            raise RuntimeError("Aucune tâche courante à préparer.")

        dependencies = {
            dependency: state.get("completed", {}).get(dependency)
            for dependency in task.get("depends_on", [])
        }

        taskplanner_prompt = self.prompts.render(
            "task_planner",
            environment=json.dumps(
                state["environment"],
                ensure_ascii=False,
                indent=2,
            ),
            dependencies = dependencies
        )

        user_prompt = json.dumps(task, ensure_ascii=False, indent=2)

        result = self.llm.structured(
            TaskPlanOutput,
            taskplanner_prompt,
            user_prompt,
        )

        return {
            "current_plan": result.model_dump(),
            "generated_tools": [
                tool.model_dump()
                for tool in result.tools
            ],
            "execution": [],
        }

    def install_and_register_tools(self, state: AgentState) -> dict[str, Any]:
        """Enregistre les outils générés et installe leurs dépendances."""

        plan = state.get("current_plan", {})
        execution = list(state.get("execution", []))

        for raw_tool in plan.get("tools", []):
            tool = GeneratedTool.model_validate(raw_tool)
            self.tools.register_generated_tool(tool)

            if not tool.packages or not tool.install_manager:
                continue

            package_args = " ".join(
                shlex_quote(package)
                for package in tool.packages
            )

            commands = {
                "pip": f"python3 -m pip install {package_args}",
                "apt": (
                    "sudo apt-get update && "
                    f"sudo apt-get install -y {package_args}"
                ),
                "npm": f"npm install {package_args}",
                "cargo": f"cargo install {package_args}",
            }

            command = commands.get(tool.install_manager)

            if not command:
                continue

            result = run_command(
                command,
                self.config.project_dir,
                self.config.command_timeout,
            )
            execution.append(result)

            if result.get("ok"):
                self.tools.record_packages(
                    tool.packages,
                    tool.install_manager,
                )

        return {"execution": execution}

    def execute_commands(self, state: AgentState) -> dict[str, Any]:
        """Exécute une commande par étape et conserve chaque résultat."""

        execution = list(state.get("execution", []))

        for command in state.get("current_plan", {}).get("commands", []):
            result = run_command(
                command,
                self.config.project_dir,
                self.config.command_timeout,
            )
            execution.append(result)

            if result.get("package"):
                self.tools.record_packages(
                    result["package"],
                    result.get("package_manager", "unknown"),
                )

        return {"execution": execution}

    def evaluate(self, state: AgentState) -> dict[str, Any]:
        """Vérifie si les résultats correspondent réellement à l'objectif."""

        task=json.dumps(
                        state.get("current_task", {}),
                        ensure_ascii=False,
                        indent=2,
                    )
        execution=json.dumps(
                        state.get("execution", []),
                        ensure_ascii=False,
                        indent=2,
                    )
        user_prompt = f'TACHE: {task} \n EXECUTION: {execution}'
        

        result = self.llm.structured(
            EvaluationOutput,
            self.prompts.load("evaluator"),
            user_prompt,
        )


        return {"evaluation": result.model_dump()}

    def record_success(self, state: AgentState) -> dict[str, Any]:
        """Enregistre une tâche réussie dans l'état partagé."""

        completed = dict(state.get("completed", {}))
        task = state.get("current_task") or {}
        task_id = task.get("id", "unknown")

        completed[task_id] = {
            "task": task,
            "execution": state.get("execution", []),
            "evaluation": state.get("evaluation", {}),
        }

        return {
            "completed": completed,
            "current_task": None,
            "execution": [],
        }

    def resolve_failure(self, state: AgentState) -> dict[str, Any]:
        """Construit un plan de réparation et programme une nouvelle tentative."""

        task = state.get("current_task") or {}
        depth = int(task.get("resolution_depth", 0))

        if depth >= self.config.max_resolution_depth:
            return {
                "final_success": False,
                "fatal_error": (
                    "Nombre maximal de niveaux de résolution atteint."
                ),
            }

        user_prompt = self.prompts.render(
            "resolver",
            task=json.dumps(task, ensure_ascii=False, indent=2),
            execution=json.dumps(
                state.get("execution", []),
                ensure_ascii=False,
                indent=2,
            ),
            evaluation=json.dumps(
                state.get("evaluation", {}),
                ensure_ascii=False,
                indent=2,
            ),
        )

        result = self.llm.structured(
            ResolutionOutput,
            self.prompts.load("resolver"),
            user_prompt,
        )
        resolution = result.model_dump()

        if not resolution["resolvable"]:
            return {
                "resolution_plan": resolution,
                "final_success": False,
                "fatal_error": resolution["reason"],
            }

        repair_tasks = [
            repair_task.model_dump()
            for repair_task in result.tasks
        ]
        retry = dict(task)
        retry["resolution_depth"] = depth + 1
        retry["depends_on"] = [task["id"] for task in repair_tasks]

        for repair_task in repair_tasks:
            repair_task["resolution_depth"] = depth + 1

        return {
            "resolution_plan": resolution,
            "queue": repair_tasks + [retry] + list(state.get("queue", [])),
            "current_task": None,
            "execution": [],
        }

    def finish(self, state: AgentState) -> dict[str, Any]:
        """Détermine si le graphe termine sur un succès ou un échec."""

        success = (
            not state.get("fatal_error")
            and not state.get("queue")
            and not state.get("current_task")
        )

        return {"final_success": bool(success)}

    @staticmethod
    def route_after_select(state: AgentState) -> str:
        """Route vers la préparation ou la fin après sélection d'une tâche."""

        if state.get("fatal_error"):
            return "finish"

        if state.get("current_task"):
            return "prepare_task"

        return "finish"

    @staticmethod
    def route_after_evaluate(state: AgentState) -> str:
        """Route vers l'enregistrement du succès ou la réparation."""

        if state.get("evaluation", {}).get("success"):
            return "record_success"

        return "resolve_failure"

    @staticmethod
    def route_after_resolution(state: AgentState) -> str:
        """Route vers une nouvelle sélection ou la fin du graphe."""

        if state.get("fatal_error"):
            return "finish"

        return "select_task"

    def _build_graph(self):
        """Construit et compile le StateGraph LangGraph complet."""

        builder = StateGraph(AgentState)

        builder.add_node("inspect_environment", self.inspect_environment)
        builder.add_node("plan", self.plan)
        builder.add_node("select_task", self.select_task)
        builder.add_node("prepare_task", self.prepare_task)
        builder.add_node(
            "install_and_register_tools",
            self.install_and_register_tools,
        )
        builder.add_node("execute_commands", self.execute_commands)
        builder.add_node("evaluate", self.evaluate)

        builder.add_node("record_success", self.record_success)
        builder.add_node("resolve_failure", self.resolve_failure)
        builder.add_node("finish", self.finish)

        builder.add_edge(START, "inspect_environment")
        builder.add_edge("inspect_environment", "plan")
        builder.add_edge("plan", "select_task")

        builder.add_conditional_edges(
            "select_task",
            self.route_after_select,
            {
                "prepare_task": "prepare_task",
                "finish": "finish",
            },
        )

        builder.add_edge("prepare_task", "install_and_register_tools")
        builder.add_edge("install_and_register_tools", "execute_commands")
        builder.add_edge("execute_commands", "evaluate")

        builder.add_conditional_edges(
            "evaluate",
            self.route_after_evaluate,
            {
                "record_success": "record_success",
                "resolve_failure": "resolve_failure",
            },
        )

        builder.add_edge("record_success", "select_task")

        builder.add_conditional_edges(
            "resolve_failure",
            self.route_after_resolution,
            {
                "select_task": "select_task",
                "finish": "finish",
            },
        )

        builder.add_edge("finish", END)

        return builder.compile()

    def export_graph_source(self) -> Path:
        """Exporte le diagramme Mermaid du graphe dans le workspace."""

        path = self.config.project_dir / ".agent" / "agent_architecture.mmd"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(MERMAID_ARCHITECTURE + "\n", encoding="utf-8")

        return path

    def run(self, request: str) -> int:
        """Initialise l'état et exécute le graphe pour une demande."""

        self.config.project_dir.mkdir(parents=True, exist_ok=True)
        architecture = self.export_graph_source()

        print("EV AGENT LINUX — LANGCHAIN / LANGGRAPH")
        print(f"IA           : {self.config.ai_id}")
        print(f"Workspace    : {self.config.project_dir}")
        print(f"Modèle       : {self.config.model}")
        print(f"Architecture : {architecture}")
        print("Auto-outillage : EXPÉRIMENTAL")

        initial_state: AgentState = {
            "request": request,
            "queue": [],
            "completed": {},
            "current_task": None,
            "execution": [],
            "generated_tools": [],
            "final_success": False,
            "fatal_error": None,
        }

        final_state = self.graph.invoke(initial_state)

        if final_state.get("final_success"):
            print("\n[SUCCÈS] Toutes les tâches ont été terminées.")
            return 0

        print(
            "\n[ÉCHEC]",
            final_state.get("fatal_error")
            or "La demande n'a pas pu être terminée.",
        )
        return 1


def shlex_quote(value: str) -> str:
    """Protège une valeur avant son insertion dans une commande shell."""

    import shlex

    return shlex.quote(value)
