"""Construction du graphe LangGraph et logique des nœuds."""

import json
import os
import sys
from pathlib import Path
from typing import Any
from langgraph.graph import END, START, StateGraph
from .config import AgentConfig
from .execution import compact, run_command
from .toolsearch import ToolSearchService
from .llm import LocalLLM
from langgraph.checkpoint.sqlite import SqliteSaver
import re
from langchain_core.runnables import RunnableConfig
from .models import (
    AgentState,
    EvaluationOutput,
    GeneratedTool,
    PlanOutput,
    ResolutionOutput,
    TaskPlanOutput,
    ToolEvaluationOutput,
    CodeOutput
)
from .prompts import PromptLoader
from .tools import ToolManager
from .logger import get_logger, preview, timed_node

logger = get_logger("graph")


MERMAID_ARCHITECTURE = """graph TD
    START([START]) --> INSPECT[inspect_environment]
    INSPECT --> PLAN[plan]
    PLAN --> SELECT[select_task]
    SELECT -->|tâche prête| PREPARE[prepare_task]
    SELECT -->|pas de tâche| FINISH[finish]
    PREPARE -->|besoin d'outil| SEARCH[search_tools]
    PREPARE -->|outil existe| EXECUTE[execute_commands]
    SEARCH --> ANALYSIS[analysis_package]
    ANALYSIS --> ELIGIBLE[record_eligible_package]
    ELIGIBLE --> |True| INSTALL[install_package]
    ELIGIBLE --> |False| ANALYSIS[analysis_package]

    INSTALL --> |nécessite code|CONSTRUCT[construct_code_for_package]
    INSTALL --> |échoue|ANALYSIS[analysis_package]
    INSTALL --> |pas besoin code|REGISTER[register_tools]
    CONSTRUCT --> EXECUTE_CODE[execute_code_for_package]
    EXECUTE_CODE --> VERIFY_CODE[verify_code_package]
    VERIFY_CODE --> |réussi| REGISTER[register_tools]
    VERIFY_CODE --> |rate| --> EXECUTE_CODE[execute_code_for_package]
    REGISTER --> PREPARE[search_tools]
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
        self.package_analyzer = ToolSearchService(self)
        self.graph = self._build_graph()

        # Récupération des checkpoints désactivée pour l'instant.
        # graph_config: RunnableConfig = {
        #                     "configurable": {
        #                         "thread_id": str(self.config.ai_id),
        #                         #"checkpoint_id": "1f1c4e7c-1dfd-6703-8006-188baed923a4"
        #                     }
        #                 }
        # for snapshot in self.graph.get_state_history(graph_config):
        #     print(
        #         "Checkpoint :", snapshot.config["configurable"]["checkpoint_id"],
        #         "| Étapes suivantes :", snapshot.next,
        #         "| État :", snapshot.values.keys(),
        #         "\n",
        #     )
        

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
        for task in result.tasks:
            logger.info(
                "Tâche planifiée %s | %s | input=%s | output=%s | depends_on=%s",
                task.id, task.title, task.input, task.output, task.depends_on,
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
        """Demande au LLM un plan d'exécution."""

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

        user_task = task.copy()
        del user_task["resolution_depth"]
        del user_task["depends_on"]
        del user_task["id"]
        user_prompt = json.dumps(user_task, ensure_ascii=False, indent=2)
        logger.info("Préparation de la tâche: %s", task.get("title"))
        logger.debug("USER PROMPT:\n%s", user_prompt)

        result = self.llm.structured(
            TaskPlanOutput,
            taskplanner_prompt,
            user_prompt,
        )
        logger.info("Plan de tâche: expected=%s | commands=%s", result.expected, result.commands)

        return {
            "current_plan": result.model_dump(),
            "execution": [],
        }

    
    def search_tools(self, state: AgentState):
        """Cherche les outils necessaire pour la tâche"""
        # step 1
        tool_spec = self.package_analyzer.identify_required_tool(state)

        packages = self.package_analyzer.search_packages(tool_spec)

        # LIMITE TEMPORAIRE : n'analyser que les N premiers packages PyPI.
        limit = self.config.max_packages_analyzed
        if len(packages) > limit:
            logger.warning(
                "Limite temporaire: %d packages trouvés, seuls les %d premiers seront analysés",
                len(packages), limit,
            )
            packages = packages[:limit]
        logger.info("Packages candidats: %s", [p.url for p in packages])

        if not packages:
            return {
                "current_tool": tool_spec,
                "current_packages": [],
                "fatal_error": "Aucun package trouvé pour l'outil requis.",
            }

        return {
            "current_tool": tool_spec, 
            "current_packages": packages
        }

    def analysis_package(self, state: AgentState):
        current_package = state["current_packages"][0]

        analysed_package = self.package_analyzer.analyze_package(
            current_package,
            state["current_tool"],
        )
        state["current_packages"][0] = analysed_package
        return {
            "current_packages": state["current_packages"]
        }
    
    def record_eligible_package(self, state: AgentState):
            current_package = state["current_packages"][0]
            if self.package_analyzer.is_package_eligible(current_package):
                logger.info("Package éligible: %s", current_package.url)
                return {"eligible_package": True}

            logger.info("Package rejeté (non éligible): %s", current_package.url)
            del state["current_packages"][0]
            return self._next_package_or_fail(state, {"eligible_package": False})
    
    def install_package(self, state: AgentState):
        analysed_package = state["current_packages"][0]
        
        if not self.package_analyzer.install_package(analysed_package):
            logger.warning("Échec d'installation: %s", analysed_package.url)
            del state["current_packages"][0]
            return self._next_package_or_fail(state, {"installed_package": False})
        
        if not analysed_package.package_use.shell_use:
             return {"installed_package": True}
        return {"installed_package": True}
           

    @staticmethod
    def _next_package_or_fail(state: AgentState, update: dict[str, Any]) -> dict[str, Any]:
        """Passe au package suivant, ou termine si la liste est épuisée."""

        update["current_packages"] = state["current_packages"]
        remaining = len(state["current_packages"])
        if remaining == 0:
            logger.error("Plus aucun package candidat à analyser.")
            update["fatal_error"] = "Aucun package candidat utilisable (limite d'analyse atteinte)."
        else:
            logger.info("%d package(s) candidat(s) restant(s)", remaining)
        return update

    def construct_code_for_package(self, state: AgentState):
                    
        analysed_package = state["current_packages"][0]
        tool_spec = state["current_tool"]
        system_prompt = self.prompts.load("tool_generator")
        logger.debug("SYS PROMPT:\n%s", system_prompt)

        user_prompt =f"TASK: \n {json.dumps(tool_spec, ensure_ascii=False, indent=2)} \n\n PACKAGE: \n {json.dumps(analysed_package.usage_context, ensure_ascii=False, indent=2)}" 
        logger.debug("USER PROMPT:\n%s", user_prompt)
        result = self.llm.structured(
            CodeOutput,
            system_prompt,
            user_prompt
            )
        code = result.code
        logger.debug("CODE:\n%s", code)
        with open(self.config.project_dir / (tool_spec["name"] + ".py"), "w") as f:
            f.write(code)
        logger.info("Outil écrit: %s", self.config.project_dir / (tool_spec["name"] + ".py"))
        

                    
    def execute_code_for_package(self, state: AgentState):
        tool_spec = state["current_tool"]
        with open(self.config.project_dir / (tool_spec["name"] + ".py"), "r") as f:
                code = f.read()
        logger.debug("TOOL FILE: %s\nCODE:\n%s", self.config.project_dir / (tool_spec["name"] + ".py"), code)
        result = self.llm.model.invoke(
               [self.prompts.load("tool_tester"),
               f"TOOL: \n - SPEC: {json.dumps(tool_spec, ensure_ascii=False, indent=2)}\n - FILE: \n {self.config.project_dir / (tool_spec["name"] + ".py")} \n - CODE: \n {code}"],
               )
        prompt = result.model_dump()["content"]
        
        logger.info("Commande de test proposée: %s", prompt)
        execution = run_command(
            prompt,
            self.config.project_dir,
            self.config.command_timeout,
        )

        execution = list(state.get("execution", []))
        execution.append(result)
        
        return {"execution": execution}

    def verify_code_package(self, state: AgentState):
        '''
         A verif
        '''
        analysed_package = state["current_packages"][0]
        tool_spec = state["current_tool"]
        execution = state["execution"][-1]
        packages = state["current_packages"]
        with open(self.config.project_dir / (tool_spec["name"] + ".py"), "r") as f:
                        code = f.read()
        user_prompt = f"PACKAGE USAGE: {json.dumps(analysed_package.usage_context, ensure_ascii=False, indent=2)} \n TASK: {json.dumps(tool_spec, ensure_ascii=False, indent=2)} \n TOOL CODE: \n {code} \n EXAMPLE RESULT: \n {execution}"
        result = self.llm.structured(
            ToolEvaluationOutput
            ,self.prompts.load("tool_verifier"),
             user_prompt
        )
        logger.debug("USER PROMPT:\n%s", user_prompt)

        result = result.model_dump()

        logger.info("Vérification outil: %s", preview(result))
        if result.success:
            return {"tool_success":result.success}
        del packages[0]
        # modify code if not working
        code = result.code.model_dump()
        with open(self.config.project_dir / (tool_spec["name"] + ".py"), "a") as f:
            f.write(code)
        return {"tool_success":result.success, "current_packages": packages}
            

                

    def register_tools(self, state: AgentState) -> dict[str, Any]:
        """Enregistre les outils générés et installe leurs dépendances."""

        raise

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
        logger.info("Évaluation: success=%s | %s", result.success, result.reason)


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
            logger.error("Profondeur de résolution max (%d) atteinte pour %s", depth, task.get("id"))
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
        logger.info(
            "Résolution: resolvable=%s | %s | tâches de réparation=%s",
            result.resolvable, result.reason, [t.title for t in result.tasks],
        )

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
    def route_after_prepare(state: AgentState) -> str:
            """Route vers l'éxécution d'une tâche ou l'installation d'un paquet."""
            if state.get("current_plan").get("expected") == "IMPOSSIBLE":
                logger.info("Tâche jugée IMPOSSIBLE avec les outils actuels -> recherche d'outil")
                return "search_tools"
    
            return "execute_commands"
    @staticmethod
    def route_after_record_eligible(state: AgentState) -> str:
                """Route vers l'éxécution d'une tâche ou l'installation d'un paquet."""
                if state.get("fatal_error"):
                    return "finish"
                if state.get("eligible_package"):
                    return "install_package"
        
                return "analysis_package"
    @staticmethod
    def route_after_install(state: AgentState) -> str:
                """Route vers l'éxécution d'une tâche ou l'installation d'un paquet."""
                if state.get("fatal_error"):
                    return "finish"
                if not state.get("installed_package"):
                    return "analysis_package"
                if not (state.get("current_packages")[0].package_use.shell_use):
                    return "construct_code_for_package"
                return "register_tools"

    @staticmethod
    def route_after_verify_code(state: AgentState) -> str:
                """Route vers l'éxécution d'une tâche ou l'installation d'un paquet."""
                if state.get("tool_success"):
                    return "register_tools"
        
                return "execute_code_for_package"
    

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

        builder.add_node("inspect_environment", timed_node(self.inspect_environment))
        builder.add_node("plan", timed_node(self.plan))
        builder.add_node("select_task", timed_node(self.select_task))
        builder.add_node("search_tools", timed_node(self.search_tools))
        builder.add_node("analysis_package", timed_node(self.analysis_package))

        builder.add_node("record_eligible_package", timed_node(self.record_eligible_package))
        builder.add_node("install_package", timed_node(self.install_package))
        builder.add_node("construct_code_for_package", timed_node(self.construct_code_for_package))
        builder.add_node("execute_code_for_package", timed_node(self.execute_code_for_package))
        builder.add_node("verify_code_package", timed_node(self.verify_code_package))
        builder.add_node("register_tools", timed_node(self.register_tools))

        builder.add_node("prepare_task", timed_node(self.prepare_task))
        
        builder.add_node("execute_commands", timed_node(self.execute_commands))
        builder.add_node("evaluate", timed_node(self.evaluate))

        builder.add_node("record_success", timed_node(self.record_success))
        builder.add_node("resolve_failure", timed_node(self.resolve_failure))
        builder.add_node("finish", timed_node(self.finish))
        

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

        builder.add_conditional_edges(
                    "prepare_task",
                    self.route_after_prepare,
                    { 
                        "search_tools": "search_tools",
                        "execute_commands": "execute_commands",
                    },
                )
    
        builder.add_conditional_edges(
            "search_tools",
            lambda state: "finish" if state.get("fatal_error") else "analysis_package",
            {"analysis_package": "analysis_package", "finish": "finish"},
        )

        builder.add_edge("analysis_package", "record_eligible_package")
        builder.add_conditional_edges(
                            "record_eligible_package",
                            self.route_after_record_eligible,
                            { 
                                "install_package": "install_package",
                                "analysis_package": "analysis_package",
                                "finish": "finish",
                            },
                        )
        builder.add_conditional_edges(
                                    "install_package",
                                    self.route_after_install,
                                    { 
                                        "construct_code_for_package": "construct_code_for_package",
                                        "analysis_package": "analysis_package",
                                        "register_tools": "register_tools",
                                        "finish": "finish",
                                    },
                                )

        
        builder.add_edge("construct_code_for_package","execute_code_for_package")
        builder.add_edge("execute_code_for_package","verify_code_package")

        builder.add_conditional_edges(
                                            "verify_code_package",
                                            self.route_after_verify_code,
                                            { 
                                                "execute_code_for_package": "execute_code_for_package",
                                                "register_tools": "register_tools",
                                            },
                                        )


        builder.add_edge("register_tools", "prepare_task")
        # builder.add_edge("prepare_task", "execute_commands")
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

        db_path = self.config.project_dir / f"checkpoints{self.config.ai_id}.sqlite"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # À l'initialisation de AgentGraph
        self._checkpointer_context = SqliteSaver.from_conn_string(str(db_path))
        self.checkpointer = self._checkpointer_context.__enter__()
        

        return builder.compile(checkpointer=self.checkpointer)

    def export_graph_source(self) -> Path:
        """Exporte le diagramme Mermaid du graphe dans le workspace."""

        path = self.config.project_dir / ".agent" / "agent_architecture.mmd"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(MERMAID_ARCHITECTURE + "\n", encoding="utf-8")

        return path

    def run(self, request: str, checkpoint_id: str | None = None) -> int:
        """Initialise l'état et exécute le graphe pour une demande."""

        self.config.project_dir.mkdir(parents=True, exist_ok=True)
        architecture = self.export_graph_source()

        logger.info("EV AGENT LINUX — LANGCHAIN / LANGGRAPH")
        logger.info("IA           : %s", self.config.ai_id)
        logger.info("Workspace    : %s", self.config.project_dir)
        logger.info("Modèle       : %s", self.config.model)
        logger.info("Architecture : %s", architecture)
        logger.info("Demande      : %s", request)

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

        graph_config: RunnableConfig = {
                    "configurable": {
                        "thread_id": str(self.config.ai_id),
                    }
                }
        if checkpoint_id:
            # Reprise : on rejoue le graphe à partir de l'état sauvegardé.
            graph_config["configurable"]["checkpoint_id"] = checkpoint_id
            snapshot = self.graph.get_state(graph_config)
            if not snapshot.values:
                logger.error("Checkpoint introuvable: %s", checkpoint_id)
                return 2
            logger.info("Reprise du checkpoint %s -> %s", checkpoint_id, snapshot.next)
            final_state = self.graph.invoke(None, graph_config)
        else:
            final_state = self.graph.invoke(initial_state, graph_config)

        if final_state.get("final_success"):
            logger.info("[SUCCÈS] Toutes les tâches ont été terminées.")
            return 0

        logger.error(
            "[ÉCHEC] %s",
            final_state.get("fatal_error")
            or "La demande n'a pas pu être terminée.",
        )
        return 1

def shlex_quote(value: str) -> str:
    """Protège une valeur avant son insertion dans une commande shell."""

    import shlex

    return shlex.quote(value)