#!/usr/bin/env python3
"""EV Agent Linux — LangChain + LangGraph.

Objectif cible : un agent capable de planifier, générer ses propres outils,
installer leurs dépendances, les utiliser, vérifier le résultat et réparer ses échecs.
L'auto-outillage est expérimental dans cette version et peut encore échouer.

Le diagramme ci-dessous est aussi exporté vers .agent/agent_architecture.mmd.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

MERMAID_ARCHITECTURE = r'''graph TD
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
'''

@dataclass
class Config:
    ollama_url: str
    model: str
    ai_id: int
    workspace_root: Path
    command_timeout: int = 300
    max_tasks: int = 30
    max_resolution_depth: int = 5
    @property
    def project_dir(self) -> Path:
        return self.workspace_root / f"projet_{self.ai_id}"

class Task(BaseModel):
    id: str
    title: str
    goal: str
    depends_on: list[str] = Field(default_factory=list)
    resolution_depth: int = 0

class GeneratedTool(BaseModel):
    name: str
    description: str
    filename: str
    install_manager: str | None = None
    packages: list[str] = Field(default_factory=list)
    code: str

class PlanOutput(BaseModel):
    tasks: list[Task]
class TaskPlanOutput(BaseModel):
    tools: list[GeneratedTool] = Field(default_factory=list)
    commands: list[str] = Field(default_factory=list)
    expected: str
class EvaluationOutput(BaseModel):
    success: bool
    reason: str
class ResolutionOutput(BaseModel):
    resolvable: bool
    reason: str
    tasks: list[Task] = Field(default_factory=list)

class AgentState(TypedDict, total=False):
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

def compact(text: str, limit: int = 16000) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + "\n...[tronqué]..."

def parse_install_command(command: str) -> tuple[str, list[str]] | None:
    p = shlex.split(command)
    if not p: return None
    if p[0] in ("pip", "pip3") and len(p) >= 3 and p[1] == "install":
        return "pip", [x for x in p[2:] if not x.startswith("-")]
    if p[0] in ("apt", "apt-get") and len(p) >= 3 and p[1] == "install":
        return "apt", [x for x in p[2:] if not x.startswith("-")]
    if len(p) >= 4 and p[0] == "sudo" and p[1] in ("apt", "apt-get") and p[2] == "install":
        return "apt", [x for x in p[3:] if not x.startswith("-")]
    if p[0] == "npm" and len(p) >= 3 and p[1] == "install":
        return "npm", [x for x in p[2:] if not x.startswith("-")]
    if p[0] == "cargo" and len(p) >= 3 and p[1] == "install":
        return "cargo", [x for x in p[2:] if not x.startswith("-")]
    return None

def run_command(command: str, cwd: Path, timeout: int) -> dict[str, Any]:
    print(f"\n$ {command}")
    installation = parse_install_command(command)
    try:
        p = subprocess.run(command, shell=True, cwd=str(cwd), executable="/bin/bash",
                           text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        result = {"command": command, "ok": p.returncode == 0, "returncode": p.returncode,
                  "stdout": compact(p.stdout), "stderr": compact(p.stderr)}
    except subprocess.TimeoutExpired as e:
        result = {"command": command, "ok": False, "returncode": -2,
                  "stdout": compact(e.stdout or ""), "stderr": f"Timeout après {timeout}s."}
    except Exception as e:
        result = {"command": command, "ok": False, "returncode": -3, "stdout": "", "stderr": repr(e)}
    if installation:
        result["package_manager"], result["package"] = installation
    if result["stdout"]: print(result["stdout"])
    if result["stderr"]: print(result["stderr"], file=sys.stderr)
    return result

class ToolManager:
    def __init__(self, project_dir: Path) -> None:
        self.dir = project_dir / ".agent"
        self.tools_dir = self.dir / "tools"
        self.dir.mkdir(parents=True, exist_ok=True); self.tools_dir.mkdir(parents=True, exist_ok=True)
        self.registry = self.dir / "tools.json"
    def installed_registry(self) -> dict[str, Any]:
        if not self.registry.exists(): return {}
        try: return json.loads(self.registry.read_text(encoding="utf-8"))
        except Exception: return {}
    def save(self, data: dict[str, Any]) -> None:
        self.registry.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    def register_generated_tool(self, tool: GeneratedTool) -> Path:
        filename = Path(tool.filename).name
        if not filename.endswith(".py"): filename += ".py"
        target = self.tools_dir / filename
        target.write_text(tool.code.rstrip() + "\n", encoding="utf-8")
        r = self.installed_registry()
        r[tool.name] = {"type": "generated", "description": tool.description,
                        "path": str(target), "packages": tool.packages}
        self.save(r); return target
    def record_packages(self, packages: list[str], manager: str) -> None:
        r = self.installed_registry()
        for package in packages: r[package] = {"type": "package", "manager": manager, "status": "installed"}
        self.save(r)

class LocalLLM:
    def __init__(self, config: Config) -> None:
        self.model = ChatOllama(base_url=config.ollama_url, model=config.model, temperature=0.1)
    def structured(self, schema: type[BaseModel], system: str, user: str) -> BaseModel:
        result = self.model.with_structured_output(schema).invoke([
            SystemMessage(content=system), HumanMessage(content=user)])
        return result if isinstance(result, BaseModel) else schema.model_validate(result)

class Agent:
    def __init__(self, config: Config) -> None:
        self.c = config; self.llm = LocalLLM(config); self.tools = ToolManager(config.project_dir)
        self.graph = self._build_graph()
    def environment(self) -> dict[str, Any]:
        uname = run_command("uname -a", self.c.project_dir, 20)
        return {"cwd": str(self.c.project_dir), "os": compact(uname["stdout"], 2000),
                "python": sys.version.split()[0], "user": os.getenv("USER", ""),
                "ai_id": self.c.ai_id, "packages_and_tools": self.tools.installed_registry()}
    def inspect_environment(self, state: AgentState) -> dict[str, Any]:
        return {"environment": self.environment()}
    def plan(self, state: AgentState) -> dict[str, Any]:
        prompt = f'''DEMANDE UTILISATEUR:\n{state["request"]}\n\nENVIRONNEMENT:\n{json.dumps(state["environment"], ensure_ascii=False, indent=2)}\n\nTransforme la demande en tâches atomiques, concrètes et vérifiables. Les tâches ne contiennent aucune commande shell. Les dépendances doivent être explicites.'''
        result = self.llm.structured(PlanOutput, "Tu es le PLANIFICATEUR d'un agent Linux autonome. Sois précis.", prompt)
        tasks = [x.model_dump() for x in result.tasks[:self.c.max_tasks]]
        if not tasks: raise RuntimeError("Le planificateur n'a créé aucune tâche.")
        return {"queue": tasks, "execution": []}
    def select_task(self, state: AgentState) -> dict[str, Any]:
        queue, completed = list(state.get("queue", [])), state.get("completed", {})
        current = None
        for i, task in enumerate(queue):
            if all(dep in completed for dep in task.get("depends_on", [])):
                current = queue.pop(i); break
        if current is None and queue:
            return {"fatal_error": "Dépendances circulaires ou invalides.", "queue": [], "current_task": None}
        return {"queue": queue, "current_task": current, "execution": []}
    def prepare_task(self, state: AgentState) -> dict[str, Any]:
        task = state["current_task"]; assert task is not None
        deps = {d: state.get("completed", {}).get(d) for d in task.get("depends_on", [])}
        prompt = f'''DEMANDE:\n{state["request"]}\n\nENVIRONNEMENT:\n{json.dumps(state["environment"], ensure_ascii=False, indent=2)}\n\nTÂCHE:\n{json.dumps(task, ensure_ascii=False, indent=2)}\n\nDÉPENDANCES:\n{json.dumps(deps, ensure_ascii=False, indent=2)}\n\nDécide comment réaliser exactement la tâche. AUTO-OUTILLAGE : si un outil Python manque, génère-le dans tools avec du code exécutable, son nom de fichier, ses packages et le gestionnaire apt/pip/npm/cargo. Les commandes doivent être exécutables dans le workspace.'''
        result = self.llm.structured(TaskPlanOutput, "Tu es l'ingénieur outillage/exécution d'un agent Linux autonome.", prompt)
        return {"current_plan": result.model_dump(), "generated_tools": [x.model_dump() for x in result.tools], "execution": []}
    def install_and_register_tools(self, state: AgentState) -> dict[str, Any]:
        plan, execution = state.get("current_plan", {}), list(state.get("execution", []))
        for raw in plan.get("tools", []):
            tool = GeneratedTool.model_validate(raw); self.tools.register_generated_tool(tool)
            if not tool.packages or not tool.install_manager: continue
            pkgs = " ".join(shlex.quote(x) for x in tool.packages)
            commands = {"pip": f"python3 -m pip install {pkgs}",
                        "apt": f"sudo apt-get update && sudo apt-get install -y {pkgs}",
                        "npm": f"npm install {pkgs}", "cargo": f"cargo install {pkgs}"}
            command = commands.get(tool.install_manager)
            if not command: continue
            result = run_command(command, self.c.project_dir, self.c.command_timeout); execution.append(result)
            if result.get("ok"): self.tools.record_packages(tool.packages, tool.install_manager)
        return {"execution": execution}
    def execute_commands(self, state: AgentState) -> dict[str, Any]:
        execution = list(state.get("execution", []))
        for command in state.get("current_plan", {}).get("commands", []):
            result = run_command(command, self.c.project_dir, self.c.command_timeout); execution.append(result)
            if result.get("package"): self.tools.record_packages(result["package"], result.get("package_manager", "unknown"))
        return {"execution": execution}
    def evaluate(self, state: AgentState) -> dict[str, Any]:
        prompt = f'''TÂCHE:\n{json.dumps(state.get("current_task", {}), ensure_ascii=False, indent=2)}\n\nPLAN:\n{json.dumps(state.get("current_plan", {}), ensure_ascii=False, indent=2)}\n\nRÉSULTATS:\n{json.dumps(state.get("execution", []), ensure_ascii=False, indent=2)}\n\nLe résultat attendu est-il réellement obtenu ? Si non, explique précisément la cause.'''
        result = self.llm.structured(EvaluationOutput, "Tu es le VÉRIFICATEUR. Tu ne proposes pas de nouvelle commande.", prompt)
        return {"evaluation": result.model_dump()}
    def record_success(self, state: AgentState) -> dict[str, Any]:
        completed = dict(state.get("completed", {})); task = state.get("current_task") or {}
        completed[task.get("id", "unknown")] = {"task": task, "execution": state.get("execution", []), "evaluation": state.get("evaluation", {})}
        return {"completed": completed, "current_task": None, "execution": []}
    def resolve_failure(self, state: AgentState) -> dict[str, Any]:
        task = state.get("current_task") or {}; depth = int(task.get("resolution_depth", 0))
        if depth >= self.c.max_resolution_depth:
            return {"final_success": False, "fatal_error": "Nombre maximal de niveaux de résolution atteint."}
        prompt = f'''TÂCHE ÉCHOUÉE:\n{json.dumps(task, ensure_ascii=False, indent=2)}\n\nEXÉCUTION:\n{json.dumps(state.get("execution", []), ensure_ascii=False, indent=2)}\n\nDIAGNOSTIC:\n{json.dumps(state.get("evaluation", {}), ensure_ascii=False, indent=2)}\n\nConstruis un plan de résolution atomique, sans commandes shell. Il peut installer un package, générer un outil ou préparer une vérification. Résous la cause de l'échec.'''
        result = self.llm.structured(ResolutionOutput, "Tu es le PLANIFICATEUR DE RÉSOLUTION d'un agent Linux autonome.", prompt)
        resolution = result.model_dump()
        if not resolution.get("resolvable"):
            return {"resolution_plan": resolution, "final_success": False, "fatal_error": resolution.get("reason")}
        subs = resolution.get("tasks", []); retry = dict(task); retry["resolution_depth"] = depth + 1; retry["depends_on"] = [x["id"] for x in subs]
        for x in subs: x["resolution_depth"] = depth + 1
        return {"resolution_plan": resolution, "queue": subs + [retry] + list(state.get("queue", [])), "current_task": None, "execution": []}
    def finish(self, state: AgentState) -> dict[str, Any]:
        ok = not state.get("fatal_error") and not state.get("queue") and not state.get("current_task")
        return {"final_success": bool(ok)}
    @staticmethod
    def route_after_select(state: AgentState) -> str:
        return "finish" if state.get("fatal_error") else ("prepare_task" if state.get("current_task") else "finish")
    @staticmethod
    def route_after_evaluate(state: AgentState) -> str:
        return "record_success" if state.get("evaluation", {}).get("success") else "resolve_failure"
    @staticmethod
    def route_after_resolution(state: AgentState) -> str:
        return "finish" if state.get("fatal_error") else "select_task"
    def _build_graph(self):
        b = StateGraph(AgentState)
        b.add_node("inspect_environment", self.inspect_environment); b.add_node("plan", self.plan); b.add_node("select_task", self.select_task)
        b.add_node("prepare_task", self.prepare_task); b.add_node("install_and_register_tools", self.install_and_register_tools)
        b.add_node("execute_commands", self.execute_commands); b.add_node("evaluate", self.evaluate); b.add_node("record_success", self.record_success)
        b.add_node("resolve_failure", self.resolve_failure); b.add_node("finish", self.finish)
        b.add_edge(START, "inspect_environment"); b.add_edge("inspect_environment", "plan"); b.add_edge("plan", "select_task")
        b.add_conditional_edges("select_task", self.route_after_select, {"prepare_task": "prepare_task", "finish": "finish"})
        b.add_edge("prepare_task", "install_and_register_tools"); b.add_edge("install_and_register_tools", "execute_commands"); b.add_edge("execute_commands", "evaluate")
        b.add_conditional_edges("evaluate", self.route_after_evaluate, {"record_success": "record_success", "resolve_failure": "resolve_failure"})
        b.add_edge("record_success", "select_task")
        b.add_conditional_edges("resolve_failure", self.route_after_resolution, {"select_task": "select_task", "finish": "finish"})
        b.add_edge("finish", END); return b.compile()
    def export_graph_source(self) -> Path:
        p = self.c.project_dir / ".agent" / "agent_architecture.mmd"; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(MERMAID_ARCHITECTURE + "\n", encoding="utf-8"); return p
    def run(self, request: str) -> int:
        self.c.project_dir.mkdir(parents=True, exist_ok=True); architecture = self.export_graph_source()
        print("EV AGENT LINUX — LANGCHAIN/LANGGRAPH"); print(f"IA={self.c.ai_id}  workspace={self.c.project_dir}  model={self.c.model}"); print(f"architecture={architecture}"); print("auto-outillage=EXPÉRIMENTAL")
        initial: AgentState = {"request": request, "queue": [], "completed": {}, "current_task": None, "execution": [], "generated_tools": [], "final_success": False, "fatal_error": None}
        final = self.graph.invoke(initial)
        if final.get("final_success"): print("\n[SUCCÈS] Toutes les tâches ont été terminées."); return 0
        print("\n[ÉCHEC]", final.get("fatal_error") or "La demande n'a pas pu être terminée."); return 1

def main() -> int:
    p = argparse.ArgumentParser(description="Agent Linux autonome basé sur LangChain + LangGraph")
    p.add_argument("request", nargs="?"); p.add_argument("--id", type=int, default=None); p.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "qwen2.5:3b"))
    p.add_argument("--workspace", default=os.getenv("AI_WORKSPACE", "AI_Workspace")); p.add_argument("--ollama-url", default=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")); p.add_argument("--timeout", type=int, default=300)
    args = p.parse_args(); ai_id = args.id
    if ai_id is None: ai_id = int(input("Numéro de l'IA [1] > ").strip() or "1")
    if ai_id < 1: print("Le numéro d'IA doit être >= 1."); return 2
    request = args.request or input("Demande utilisateur > ").strip()
    if not request: print("Demande vide."); return 2
    config = Config(args.ollama_url, args.model, ai_id, Path(args.workspace).expanduser().resolve(), args.timeout)
    try: return Agent(config).run(request)
    except Exception as exc: print(f"\nERREUR: {exc}\nVérifie Ollama, le modèle et les dépendances Python."); return 4

if __name__ == "__main__": raise SystemExit(main())
