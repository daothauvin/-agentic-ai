"""Construction du graphe LangGraph et logique des nœuds."""

import ast
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from langgraph.graph import END, START, StateGraph
from .config import AgentConfig
from .execution import changed_files, compact, run_command, start_change_marker
from .toolsearch import ToolSearchService, pypi_name_from_url
from .package_registry import PackageRecord, PackageRegistry, VerifiedUsage
from .package_advisor import PackageAdvisor
from .rag import BgeM3Embedder
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
)
from .prompts import PromptLoader
from .tools import ToolManager
from .logger import get_logger, log_codegen, preview, timed_node

logger = get_logger("graph")


MERMAID_ARCHITECTURE = """graph TD
    START([START]) --> INSPECT[inspect_environment]
    INSPECT --> PLAN[plan]
    PLAN --> SELECT[select_task]
    SELECT -->|tâche prête| FIND[find_installed_packages]
    SELECT -->|pas de tâche| FINISH[finish]
    FIND --> PREPARE[prepare_task]
    PREPARE -->|besoin d'outil| SEARCH[search_tools]
    PREPARE -->|outil existe| EXECUTE[execute_commands]
    PREPARE -->|trop de recherches| FINISH
    SEARCH --> ANALYSIS[analysis_package]
    ANALYSIS --> ELIGIBLE[record_eligible_package]
    ELIGIBLE --> |True| INSTALL[install_package]
    ELIGIBLE --> |False| ANALYSIS[analysis_package]

    INSTALL --> |nécessite code|CONSTRUCT[construct_code_for_package]
    INSTALL --> |échoue|ANALYSIS[analysis_package]
    INSTALL --> |CLI, sans code|EXECUTE_CODE[execute_code_for_package]
    CONSTRUCT --> EXECUTE_CODE
    EXECUTE_CODE --> VERIFY_CODE[verify_code_package]
    VERIFY_CODE --> |réussi| REGISTER[register_tools]
    VERIFY_CODE --> |rate, correction| EXECUTE_CODE
    VERIFY_CODE --> |rate, paquet abandonné| ANALYSIS
    REGISTER --> |sauvegarde du paquet| FIND
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
        # Registre des paquets vérifiés (avant ToolSearchService, qui l'utilise au classement).
        self.packages = PackageRegistry(
            config.project_dir / ".agent",
            embedder=BgeM3Embedder(config.ollama_url).embed,
        )
        self.advisor = PackageAdvisor(self.llm, self.prompts, self.packages, config)
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

        sandbox = self.config.sandbox
        uname = run_command("uname -a", self.config.project_dir, 20, sandbox=sandbox)

        if sandbox is not None:
            # Python et utilisateur du conteneur, pas ceux de l'hôte.
            python = run_command("python3 --version", self.config.project_dir, 20, sandbox=sandbox)
            python_version = python["stdout"].split()[-1] if python["ok"] else "inconnu"
            user = run_command("whoami", self.config.project_dir, 20, sandbox=sandbox)["stdout"].strip()
        else:
            python_version = sys.version.split()[0]
            user = os.getenv("USER", "")

        return {
            "cwd": str(self.config.project_dir),
            "os": compact(uname["stdout"], 2000),
            "python": python_version,
            "user": user,
            "docker": sandbox is not None,
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
        
    
    def find_installed_packages(self, state: AgentState) -> dict[str, Any]:
        """Cherche, parmi les paquets déjà vérifiés, ceux utilisables pour la tâche.

        RAG (bge-m3) sur le registre ``.agent/packages.json`` puis vérification
        LLM paquet par paquet ; le résultat (description courte + exemple de
        commande pour la tâche) est injecté dans le prompt de ``prepare_task``.

        Args:
            state: État courant (``current_task``).

        Returns:
            Mise à jour ``package_hints``.
        """

        hints = self.advisor.hints_for_task(state.get("current_task"))
        logger.info("Paquets installés retenus pour la tâche: %s", [h["name"] for h in hints])
        return {"package_hints": hints}

    def prepare_task(self, state: AgentState) -> dict[str, Any]:
        """Demande au LLM un plan d'exécution.

        Les paquets installés jugés utilisables (``package_hints``) sont décrits
        dans le prompt avec un exemple d'utilisation pour la tâche. Si le plan
        est ``IMPOSSIBLE`` alors que la tâche a déjà déclenché
        ``max_tool_searches_per_task`` recherches de paquet, on s'arrête.

        Args:
            state: État courant.

        Returns:
            Mise à jour ``current_plan`` (et ``fatal_error`` si bloqué).
        """

        task = state["current_task"]
        if task is None:
            raise RuntimeError("Aucune tâche courante à préparer.")

        dependencies = {
            dependency: state.get("completed", {}).get(dependency)
            for dependency in task.get("depends_on", [])
        }

        packages = self.advisor.format_hints(state.get("package_hints") or [])
        logger.debug("Paquets installés décrits au planificateur:\n%s", packages)
        taskplanner_prompt = self.prompts.render(
            "task_planner",
            environment=json.dumps(
                state["environment"],
                ensure_ascii=False,
                indent=2,
            ),
            dependencies=dependencies,
            packages=packages,
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

        update: dict[str, Any] = {
            "current_plan": result.model_dump(),
            "execution": [],
        }
        searches = (state.get("tool_searches") or {}).get(task.get("id", ""), 0)
        if result.expected == "IMPOSSIBLE" and searches >= self.config.max_tool_searches_per_task:
            logger.error("Tâche %s toujours IMPOSSIBLE après %d recherche(s) de paquet",
                         task.get("id"), searches)
            update["fatal_error"] = (
                f"Tâche impossible même après {searches} recherche(s) de paquet."
            )
        return update

    
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

        task_id = (state.get("current_task") or {}).get("id", "")
        searches = dict(state.get("tool_searches") or {})
        searches[task_id] = searches.get(task_id, 0) + 1

        if not packages:
            return {
                "current_tool": tool_spec,
                "current_packages": [],
                "tool_searches": searches,
                "fatal_error": "Aucun package trouvé pour l'outil requis.",
            }

        return {
            "current_tool": tool_spec,
            "current_packages": packages,
            "tool_searches": searches,
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
        """Installe le package candidat et choisit son mode de vérification.

        ``cli`` : le paquet s'utilise directement en ligne de commande, sans
        code généré ; ``python`` : un outil Python est généré puis testé.

        Args:
            state: État courant (``current_packages[0]`` analysé).

        Returns:
            Mise à jour ``installed_package``, ``tool_mode``, compteurs remis à zéro.
        """

        analysed_package = state["current_packages"][0]

        if not self.package_analyzer.install_package(analysed_package):
            logger.warning("Échec d'installation: %s", analysed_package.url)
            del state["current_packages"][0]
            return self._next_package_or_fail(state, {"installed_package": False})

        mode = "cli" if analysed_package.package_use.shell_use else "python"
        logger.info("Paquet installé: %s | vérification en mode %s", analysed_package.url, mode)
        return {
            "installed_package": True,
            "tool_mode": mode,
            "tool_command": None,
            "tool_attempts": 0,
            "tool_success": None,
            "tool_retry": False,
            "execution": [],
        }

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

    def _tool_path(self, tool_spec: dict[str, Any]) -> Path:
        """Chemin du fichier de l'outil Python en cours de vérification.

        Args:
            tool_spec: Spécification de l'outil.

        Returns:
            Chemin ``<workspace>/<nom>.py``.
        """

        return self.config.project_dir / (tool_spec["name"] + ".py")

    def construct_code_for_package(self, state: AgentState):
        """Génère l'outil Python qui utilise le paquet (mode ``python``).

        Args:
            state: État courant.

        Returns:
            Mise à jour des compteurs de vérification.
        """

        analysed_package = state["current_packages"][0]
        tool_spec = state["current_tool"]
        system_prompt = self.prompts.load("tool_generator")
        # Doc en texte brut (pas json.dumps : les \n échappés gênent un 3B) et
        # réponse en texte libre (le JSON contraint casse le code généré).
        user_prompt = (
            f"TASK:\n{json.dumps(tool_spec, ensure_ascii=False, indent=2)}\n\n"
            f"PACKAGE DOCUMENTATION:\n{analysed_package.usage_context}"
        )
        raw = self.llm.text(system_prompt, user_prompt, "tool_generator", code=True)
        code = strip_code_fences(raw)
        path = self._tool_path(tool_spec)
        log_codegen(
            logger, "construct_code_for_package",
            {"package": analysed_package.url, "system_prompt": system_prompt,
             "user_prompt": user_prompt},
            {"raw": raw, "code": code, "file": path},
        )
        path.write_text(code, encoding="utf-8")
        logger.info("Outil écrit: %s", path)
        return {"tool_mode": "python", "tool_attempts": 0, "tool_command": None, "execution": []}

    def execute_code_for_package(self, state: AgentState):
        """Teste le paquet : exécute l'outil Python généré, ou la commande du paquet.

        * mode ``python`` : le LLM (prompt ``tool_tester``) propose
          ``python3 FICHIER ARGUMENTS`` ;
        * mode ``cli`` : le paquet est utilisé directement, sans code ; le LLM
          (prompt ``package_cli_tester``) propose la commande à partir de la doc.
          Après un échec, la commande corrigée par le vérificateur est rejouée.

        Args:
            state: État courant.

        Returns:
            Mise à jour ``execution`` (résultat de la commande) et ``tool_command``.
        """

        tool_spec = state["current_tool"]
        mode = state.get("tool_mode") or "python"

        if mode == "cli":
            package = state["current_packages"][0]
            command = state.get("tool_command")
            if not command:
                name = pypi_name_from_url(package.url) or package.title
                system_prompt = self.prompts.load("package_cli_tester")
                user_prompt = (
                    f"PACKAGE: {name}\n"
                    f"SPEC: {json.dumps(tool_spec, ensure_ascii=False, indent=2)}\n"
                    f"{self._test_context(state)}\n\n"
                    f"DOCUMENTATION:\n{package.usage_context}"
                )
                result = self.llm.model.invoke([system_prompt, user_prompt])
                command = strip_code_fences(result.content)
                log_codegen(
                    logger, "execute_code_for_package (cli)",
                    {"system_prompt": system_prompt, "user_prompt": user_prompt},
                    {"raw": result.content, "command": command},
                )
            else:
                logger.debug("Commande CLI corrigée rejouée: %s", command)
        else:
            path = self._tool_path(tool_spec)
            code = path.read_text(encoding="utf-8")
            system_prompt = self.prompts.load("tool_tester")
            names = positional_arguments(code)
            if names == []:
                # Aucun argument obligatoire : rien à demander au LLM.
                arguments = ""
                logger.debug("Outil sans argument positionnel : test sans argument")
            else:
                if names is None:
                    # Code illisible par ast : le LLM lit le code lui-même.
                    wanted = f"CODE:\n{code}"
                else:
                    wanted = "ARGUMENTS:\n" + "\n".join(
                        f"{i}. {name}" for i, name in enumerate(names, 1)
                    )
                user_prompt = (
                    f"SPEC:\n{json.dumps(tool_spec, ensure_ascii=False, indent=2)}\n\n"
                    f"{self._test_context(state)}\n\n{wanted}"
                )
                # Testeur = agent de code (modèle ``code_model``).
                result = self.llm.code_model.invoke([system_prompt, user_prompt])
                # Le LLM ne donne que les arguments : le chemin du fichier est
                # connu, on construit la commande nous-mêmes (un 3B le recopie mal).
                arguments = strip_test_arguments(strip_code_fences(result.content))
                log_codegen(
                    logger, "execute_code_for_package (python)",
                    {"system_prompt": system_prompt, "user_prompt": user_prompt},
                    {"raw": result.content, "arguments": arguments},
                )
            command = f"python3 {shlex_quote(str(path))} {arguments}".strip()

        logger.info("Commande de test proposée (%s): %s", mode, command)
        # Le test tourne dans un dossier à part (ses fichiers ne se mélangent pas
        # aux sorties de la vraie tâche), sans le vider : les fichiers produits
        # sont ceux dont la ctime est postérieure au marqueur posé juste avant.
        test_dir = self.config.project_dir / ".agent" / "tool_test"
        test_dir.mkdir(parents=True, exist_ok=True)
        marker = start_change_marker(self.config.project_dir / ".agent" / "tool_test.marker")
        test_result = run_command(
            command,
            test_dir,
            self.config.command_timeout,
            sandbox=self.config.sandbox,
        )
        logger.debug(
            "RÉSULTAT DU TEST: ok=%s code=%s\n--- stdout ---\n%s\n--- stderr ---\n%s",
            test_result.get("ok"), test_result.get("returncode"),
            test_result.get("stdout"), test_result.get("stderr"),
        )
        if test_result["ok"] and not test_result["stdout"].strip():
            logger.warning("Test de l'outil sans sortie (stdout vide) : rien à vérifier")

        # Fichiers créés ou modifiés par le test, où qu'ils soient : un fait que
        # le vérificateur ne peut pas deviner depuis stdout (« test.jpg » affiché
        # ≠ fichier écrit). Un fichier laissé par un test précédent et non
        # réécrit n'y figure pas.
        test_result["produced_files"] = [
            f"{f['path']} ({f['size']} octets)"
            for f in changed_files(marker, self.config.project_dir, test_dir,
                                   sandbox=self.config.sandbox)
        ]
        logger.debug("Fichiers produits par le test: %s", test_result["produced_files"])

        # On conserve le résultat de la commande (stdout/stderr), pas la réponse du LLM.
        execution = list(state.get("execution", []))
        execution.append(test_result)

        return {"execution": execution, "tool_command": command}

    def verify_code_package(self, state: AgentState):
        """Vérifie le test du paquet ; corrige puis réessaie, ou abandonne le paquet.

        Seul un paquet validé ici atteint ``register_tools`` (et donc le registre).
        Échec : jusqu'à ``max_tool_fix_attempts`` corrections (code Python
        réécrit, ou commande corrigée en mode ``cli``), puis le paquet est
        abandonné au profit du candidat suivant.

        Args:
            state: État courant.

        Returns:
            Mise à jour ``tool_success`` / ``tool_retry`` / ``current_packages``.
        """

        analysed_package = state["current_packages"][0]
        tool_spec = state["current_tool"]
        execution = state["execution"][-1]
        packages = state["current_packages"]
        mode = state.get("tool_mode") or "python"

        if mode == "cli":
            system_prompt = self.prompts.load("cli_verifier")
            tried = f"COMMAND: \n {state.get('tool_command')}"
        else:
            system_prompt = self.prompts.load("tool_verifier")
            tried = f"TOOL CODE: \n {self._tool_path(tool_spec).read_text(encoding='utf-8')}"
        produced = execution.get("produced_files")
        user_prompt = (
            f"PACKAGE USAGE:\n{analysed_package.usage_context}\n\n"
            f"TASK:\n{json.dumps(tool_spec, ensure_ascii=False, indent=2)}\n\n"
            f"{tried}\n\n"
            f"TEST RESULT:\n"
            f"- command: {execution.get('command')}\n"
            f"- return code: {execution.get('returncode')}\n"
            f"- stdout:\n{execution.get('stdout')}\n"
            f"- stderr:\n{execution.get('stderr')}\n"
            f"- produced files: {', '.join(produced) if produced else 'aucun'}"
        )
        result = self.llm.structured(
            ToolEvaluationOutput,
            system_prompt,
            user_prompt,
        )
        log_codegen(
            logger, f"verify_code_package ({mode})",
            {"system_prompt": system_prompt, "user_prompt": user_prompt},
            {"success": result.success, "raw_fix": result.code},
        )

        logger.info("Vérification outil (%s): %s", mode, preview(result.model_dump()))
        if result.success and execution.get("ok"):
            return {"tool_success": True, "tool_retry": False}
        if result.success:
            logger.warning("Vérificateur positif mais commande en échec (code %s) : refusé",
                           execution.get("returncode"))

        attempts = int(state.get("tool_attempts") or 0) + 1
        if attempts > self.config.max_tool_fix_attempts:
            fix = ""
        elif mode == "cli":
            fix = strip_code_fences(result.code or "")
        else:
            fix = self._fix_tool_code(analysed_package, tool_spec, execution)
        if fix:
            logger.info("Correction %d/%d du paquet %s", attempts,
                        self.config.max_tool_fix_attempts, analysed_package.url)
            logger.debug("Correction proposée par le vérificateur:\n%s", fix)
            update = {"tool_success": False, "tool_retry": True, "tool_attempts": attempts}
            if mode == "cli":
                update["tool_command"] = fix
            else:
                self._tool_path(tool_spec).write_text(fix, encoding="utf-8")
                update["tool_command"] = None
            return update

        logger.warning("Paquet abandonné après %d tentative(s): %s", attempts, analysed_package.url)
        del packages[0]
        if not packages:
            logger.error(
                "Vérification de l'outil %s échouée et plus aucun package candidat",
                tool_spec.get("name"),
            )
            return {
                "tool_success": False,
                "tool_retry": False,
                "current_packages": packages,
                "fatal_error": "Outil généré invalide : plus aucun package candidat.",
            }
        logger.info("%d package(s) candidat(s) restant(s)", len(packages))
        return {"tool_success": False, "tool_retry": False, "current_packages": packages}

    @staticmethod
    def _test_context(state: AgentState, limit: int = 1500) -> str:
        """Contexte donné au testeur pour choisir des valeurs de test réelles.

        Le testeur ne connaît pas forcément une URL ou une donnée valide : on
        lui donne la demande de l'utilisateur, la tâche en cours et le résultat
        (stdout des commandes) des étapes dont elle dépend, où ces valeurs se
        trouvent souvent.

        Args:
            state: État courant.
            limit: Taille maximale du stdout gardé par étape précédente.

        Returns:
            Texte avec les sections ``USER REQUEST``, ``CURRENT TASK`` et
            ``PREVIOUS STEPS RESULTS``.
        """

        task = state.get("current_task") or {}
        lines = [f"USER REQUEST:\n{state.get('request') or '(aucune)'}", "",
                 "CURRENT TASK:"]
        for key in ("title", "input", "output"):
            if task.get(key):
                lines.append(f"- {key}: {task[key]}")
        lines += ["", "PREVIOUS STEPS RESULTS:"]
        completed = state.get("completed") or {}
        found = False
        for dependency in task.get("depends_on") or []:
            done = completed.get(dependency)
            if not done:
                continue
            found = True
            dep_task = done.get("task") or {}
            stdout = "\n".join(
                (run.get("stdout") or "").strip() for run in done.get("execution") or []
            ).strip()
            if len(stdout) > limit:
                stdout = stdout[:limit] + "\n...[tronqué]..."
            lines.append(f"- {dep_task.get('title', dependency)} "
                         f"(output: {dep_task.get('output', '?')}):\n{stdout or '(stdout vide)'}")
        if not found:
            lines.append("(aucune)")
        return "\n".join(lines)

    def _fix_tool_code(
        self,
        analysed_package: Any,
        tool_spec: dict[str, Any],
        execution: dict[str, Any],
    ) -> str:
        """Demande au LLM (prompt ``tool_fixer``) le script Python corrigé.

        La correction est générée en texte libre, comme l'outil initial : le
        vérificateur (sortie JSON) ne fait que juger.

        Args:
            analysed_package: Paquet en cours de vérification.
            tool_spec: Spécification de l'outil.
            execution: Résultat du dernier test (commande, code retour, sorties).

        Returns:
            Script corrigé, sans délimiteurs de bloc de code.
        """

        path = self._tool_path(tool_spec)
        system_prompt = self.prompts.load("tool_fixer")
        user_prompt = (
            f"TASK:\n{json.dumps(tool_spec, ensure_ascii=False, indent=2)}\n\n"
            f"PACKAGE DOCUMENTATION:\n{analysed_package.usage_context}\n\n"
            f"CURRENT CODE ({path.name}):\n{path.read_text(encoding='utf-8')}\n\n"
            f"TEST COMMAND:\n{execution.get('command')}\n"
            f"RETURN CODE: {execution.get('returncode')}\n"
            f"STDOUT:\n{execution.get('stdout')}\n"
            f"STDERR:\n{execution.get('stderr')}"
        )
        raw = self.llm.text(system_prompt, user_prompt, "tool_fixer", code=True)
        fix = strip_code_fences(raw)
        log_codegen(
            logger, "fix_tool_code",
            {"system_prompt": system_prompt, "user_prompt": user_prompt},
            {"raw": raw, "code": fix},
        )
        return fix

    def register_tools(self, state: AgentState) -> dict[str, Any]:
        """Sauvegarde le paquet vérifié (et son outil) dans le registre.

        La fiche garde toutes les informations disponibles : métadonnées PyPI
        (version, résumé, mots-clés, page d'accueil), analyse d'installation et
        d'usage, extraits de doc, outil Python généré et commande validée, et
        une description + des capacités rédigées par le LLM (index RAG).

        Args:
            state: État courant (paquet validé en ``current_packages[0]``).

        Returns:
            Remise à zéro de l'état de recherche de paquet.
        """

        package = state["current_packages"][0]
        tool_spec = state["current_tool"] or {}
        mode = state.get("tool_mode") or "python"
        name = pypi_name_from_url(package.url) or package.title.lower()
        last = (state.get("execution") or [{}])[-1]
        command = state.get("tool_command") or last.get("command", "")

        tool_file = None
        if mode == "python":
            source = self._tool_path(tool_spec)
            target = self.tools.tools_dir / source.name
            shutil.copyfile(source, target)
            tool_file = str(target)
            command = command.replace(str(source), tool_file)
            self.tools.record_tool(tool_spec.get("name", source.stem),
                                   tool_spec.get("description", ""), target, [name])
            logger.info("Outil Python sauvegardé: %s", target)

        info = (self.package_analyzer.fetch_pypi_json_by_name(name) or {}).get("info") or {}
        install = package.package_install
        use = package.package_use
        urls = info.get("project_urls") or {}
        keywords = info.get("keywords") or ""
        record = PackageRecord(
            name=name,
            title=info.get("name") or package.title,
            url=package.url,
            version=info.get("version") or "",
            summary=info.get("summary") or package.content,
            keywords=[k.strip() for k in keywords.replace(";", ",").split(",") if k.strip()]
            if isinstance(keywords, str) else list(keywords),
            homepage=info.get("home_page") or urls.get("Homepage") or urls.get("Source") or "",
            manager="pip" if (install is None or install.pip_install) else "apt",
            install_commands=self.package_analyzer.last_install_commands
            or (install.install_commands if install else []),
            python_use=bool(use and use.python_use),
            shell_use=bool(use and use.shell_use),
            requires_api_key=bool((use and use.requires_API_key) or (install and install.requires_API_key)),
            paid=bool((use and use.paid) or (install and install.paid)),
            usage_context=package.usage_context or "",
            usages=[VerifiedUsage(
                tool_name=tool_spec.get("name", ""),
                tool_description=tool_spec.get("description", ""),
                tool_input=tool_spec.get("input", ""),
                tool_output=tool_spec.get("output", ""),
                mode=mode,
                tool_file=tool_file,
                command=command,
                output_preview=compact(last.get("stdout", ""), 1000),
                verified_at=datetime.now().isoformat(timespec="seconds"),
            )],
        )
        described = self.advisor.describe(record)
        record.description = described.description
        record.capabilities = described.capabilities
        self.packages.add(record)
        self.tools.record_packages([name], record.manager)

        return {
            "current_packages": [],
            "package_hints": [],
            "tool_success": None,
            "tool_retry": False,
            "tool_command": None,
            "tool_attempts": 0,
            "execution": [],
        }

    def execute_commands(self, state: AgentState) -> dict[str, Any]:
        """Exécute une commande par étape et conserve chaque résultat."""

        execution = list(state.get("execution", []))
        
        for command in state.get("current_plan", {}).get("commands", []):
            result = run_command(
                command,
                self.config.project_dir,
                self.config.command_timeout,
                sandbox=self.config.sandbox,
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
            return "find_installed_packages"

        return "finish"
    
    @staticmethod
    def route_after_prepare(state: AgentState) -> str:
            """Route vers l'éxécution d'une tâche ou l'installation d'un paquet."""
            if state.get("fatal_error"):
                return "finish"
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
                if state.get("tool_mode") == "cli":
                    # Paquet utilisable directement en ligne de commande : test sans code.
                    return "execute_code_for_package"
                return "construct_code_for_package"

    @staticmethod
    def route_after_verify_code(state: AgentState) -> str:
                """Route vers l'éxécution d'une tâche ou l'installation d'un paquet."""
                if state.get("fatal_error"):
                    return "finish"
                if state.get("tool_success"):
                    return "register_tools"
                if state.get("tool_retry"):
                    return "execute_code_for_package"
                # Paquet abandonné : analyse du candidat suivant.
                return "analysis_package"
    

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

        builder.add_node("find_installed_packages", timed_node(self.find_installed_packages))
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
                "find_installed_packages": "find_installed_packages",
                "finish": "finish",
            },
        )
        builder.add_edge("find_installed_packages", "prepare_task")

        builder.add_conditional_edges(
                    "prepare_task",
                    self.route_after_prepare,
                    { 
                        "search_tools": "search_tools",
                        "execute_commands": "execute_commands",
                        "finish": "finish",
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
                                        "execute_code_for_package": "execute_code_for_package",
                                        "analysis_package": "analysis_package",
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
                                                "analysis_package": "analysis_package",
                                                "register_tools": "register_tools",
                                                "finish": "finish",
                                            },
                                        )


        # Le paquet sauvegardé est retrouvé par find_installed_packages puis décrit à prepare_task.
        builder.add_edge("register_tools", "find_installed_packages")
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
        logger.info("Modèle code  : %s", self.config.code_model or self.config.model)
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

def strip_code_fences(text: str) -> str:
    """Retire les blocs ```...``` qu'ajoute parfois le LLM autour d'une commande/code.

    Si la réponse contient un bloc de code entouré de texte (explication
    avant ou après), seul le premier bloc est gardé.

    Args:
        text: Réponse brute du LLM.

    Returns:
        Texte sans délimiteurs de bloc de code, espaces de bord retirés.
    """

    text = (text or "").strip()
    match = re.search(r"```[\w+-]*\n(.*?)\n?```", text, re.S)
    return match.group(1).strip() if match else text


def positional_arguments(code: str) -> list[str] | None:
    """Noms des arguments positionnels argparse d'un script Python.

    Cherche les appels ``*.add_argument("nom", ...)`` dont le premier
    argument est une chaîne qui ne commence pas par ``-``.

    Args:
        code: Code source de l'outil.

    Returns:
        Noms dans l'ordre de déclaration, ou ``None`` si le code ne se
        parse pas.
    """

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    names = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and not node.args[0].value.startswith("-")):
            names.append((node.lineno, node.col_offset, node.args[0].value))
    return [name for _, _, name in sorted(names)]


def strip_test_arguments(text: str) -> str:
    """Garde seulement les arguments d'une réponse du testeur d'outil.

    Le prompt ``tool_tester`` demande uniquement les arguments, mais le LLM
    renvoie parfois la commande entière (``python3 chemin/outil.py ARGS``) :
    le préfixe ``python3 <fichier>.py`` est alors retiré.

    Args:
        text: Réponse du LLM, sans bloc de code.

    Returns:
        Arguments shell, sur une seule ligne.
    """

    line = (text or "").strip().splitlines()[0].strip() if (text or "").strip() else ""
    match = re.match(r"^(?:python3?\s+)?(?:'[^']*\.py'|\"[^\"]*\.py\"|\S+\.py)\s*", line)
    return line[match.end():].strip() if match else line


def shlex_quote(value: str) -> str:
    """Protège une valeur avant son insertion dans une commande shell."""

    import shlex

    return shlex.quote(value)