"""Bench de la sauvegarde des paquets (registre + RAG + usage), suivi dans Phoenix.

Le registre ``.agent/packages.json`` d'un workspace temporaire est rempli avec
20 paquets très connus (bench/fixtures/registry_packages.json, fiches au format
de ``register_tools``). Rien n'est installé : on teste uniquement ce que le
modèle fait des fiches. Cas : bench/registry_cases.json (2 tâches par paquet
+ 5 tâches sans paquet adapté, ``expected = null``).

Deux tests (une expérience Phoenix chacun) :

``retrieval`` — le RAG retrouve-t-il le bon paquet à donner au modèle ?
    Nœud réel ``find_installed_packages`` : RAG bge-m3 sur le registre puis
    vérification LLM (prompt ``package_matcher``). Scores :
    ``rag_top1`` / ``rag_topk`` / ``rag_rr`` (rang du bon paquet avant le LLM,
    tâches positives), ``selected_ok`` (bon paquet retenu, ou rien retenu pour
    une tâche négative), ``selected_precision``, ``false_positive``.

``usage`` — le modèle utilise-t-il correctement le paquet ?
    Les indices (description + exemple) du bon paquet sont construits par
    ``PackageAdvisor.match`` (oracle par défaut, ``--hints rag`` pour le
    pipeline complet), puis le nœud réel ``prepare_task`` produit les
    commandes. Scores : ``matcher_accepts``, ``not_impossible``,
    ``uses_package`` (commande ou fichier d'outil du paquet présent),
    ``no_reinstall``, ``bash_syntax``, ``io_files`` (fichiers d'entrée/sortie
    de la tâche présents), ``judge`` (optionnel, ``--judge-model``).

Prérequis : Ollama (qwen2.5:3b, bge-m3) et Phoenix
(``PHOENIX_WORKING_DIR=.phoenix phoenix serve``).

    python bench/registry_bench.py --test retrieval --label defaut
    python bench/registry_bench.py --test usage --label defaut
    python bench/registry_bench.py --test all --label seuil_050 --set registry_min_score=0.5
    python bench/registry_bench.py --test usage --label juge --judge-model qwen3.5:9b
"""

import argparse
import json
import re
import subprocess
import tempfile
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from common import PHOENIX_URL, ROOT, parse_overrides, register_tracing
from phoenix.client import Client
from pydantic import BaseModel, Field

from agent.config import AgentConfig
from agent.graph import AgentGraph
from agent.logger import get_logger, setup_logging
from agent.package_registry import PackageRecord

FIXTURE = ROOT / "bench/fixtures/registry_packages.json"
CASES = ROOT / "bench/registry_cases.json"
RESULTS = ROOT / "bench/results"
logger = get_logger("bench.registry")


class Judgement(BaseModel):
    """Verdict du juge LLM sur les commandes produites."""

    correct: bool = Field(description="True if the commands correctly perform the task with the package.")
    reason: str = ""


# ------------------------------------------------------------------
# Préparation
# ------------------------------------------------------------------

def make_agent(model: str, overrides: dict) -> AgentGraph:
    """Agent réel dans un workspace temporaire, registre rempli avec la fixture.

    Args:
        model: Modèle Ollama.
        overrides: Surcharges d'AgentConfig.

    Returns:
        L'agent (graphe compilé, registre et outils remplis).
    """

    config = AgentConfig(
        ollama_url="http://localhost:11434", model=model, ai_id=0,
        workspace_root=Path(tempfile.mkdtemp(prefix="registry_bench_")),
        prompts_dir=ROOT / "agent/prompts",
    )
    config = replace(config, **overrides)
    config.project_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(config.project_dir / ".agent/logs", console_level="WARNING")
    print(f"Logs DEBUG : {log}")
    agent = AgentGraph(config)

    tools_dir = str(agent.tools.tools_dir)
    for raw in json.loads(FIXTURE.read_text(encoding="utf-8")):
        raw = json.loads(json.dumps(raw).replace("{TOOLS}", tools_dir))
        record = PackageRecord(
            title=raw["name"], url=f"https://pypi.org/project/{raw['name']}/",
            install_commands=[f"pip install {raw['name']}"], **raw,
        )
        agent.packages.add(record)
        agent.tools.record_packages([record.name], "pip")
        for usage in record.usages:
            if usage.tool_file:
                agent.tools.record_tool(usage.tool_name, usage.tool_description,
                                        Path(usage.tool_file), [record.name])
    # Calcule (et met en cache) les embeddings des 20 fiches une fois pour toutes.
    agent.packages.search("warmup", k=1)
    return agent


def get_or_create_dataset(client, name: str, cases: list[dict], description: str):
    """Dataset Phoenix des cas (entrée : tâche, sortie : paquet attendu).

    Args:
        client: Client Phoenix.
        name: Nom du dataset.
        cases: Cas du bench.
        description: Description du dataset.

    Returns:
        Le dataset Phoenix.
    """

    try:
        return client.datasets.get_dataset(dataset=name)
    except Exception:
        return client.datasets.create_dataset(
            name=name, dataset_description=description,
            inputs=[{"case_id": c["id"], "task": c["task"]} for c in cases],
            outputs=[{"expected": c["expected"], "marker": c["marker"]} for c in cases],
        )


def files_in(text: str) -> list[str]:
    """Noms de fichiers (``nom.ext``) cités dans un texte, URLs exclues.

    Args:
        text: Texte d'entrée / de sortie d'une tâche.

    Returns:
        Liste des noms de fichiers.
    """

    text = re.sub(r"https?://\S+", " ", text or "")
    return re.findall(r"[\w-]+\.(?:png|jpg|jpeg|mp4|mp3|csv|xlsx|pdf|txt|md|html|docx|py|enc|json|yaml|yml|db|zip)\b", text)


def summarize(name: str, outputs: list[dict], metrics: dict) -> dict:
    """Moyennes locales des métriques (affichées et sauvegardées en JSON).

    Args:
        name: Nom du test.
        outputs: Liste (cas, sortie de la tâche).
        metrics: Nom -> fonction(sortie, attendu) renvoyant bool/float/None.

    Returns:
        Dictionnaire métrique -> moyenne.
    """

    summary = {}
    for metric, func in metrics.items():
        values = [func(o["output"], o["expected"]) for o in outputs]
        values = [float(v[0] if isinstance(v, tuple) else v) for v in values
                  if v is not None and not (isinstance(v, tuple) and v[0] is None)]
        summary[metric] = round(sum(values) / len(values), 3) if values else None
    print(f"\n=== {name} ===")
    for metric, value in summary.items():
        print(f"  {metric:<20} {value}")
    return summary


# ------------------------------------------------------------------
# Test 1 : retrieval
# ------------------------------------------------------------------

def run_retrieval(agent: AgentGraph, client, cases, args, metadata) -> dict:
    """Le RAG (+ vérification LLM) retrouve-t-il le bon paquet ?"""

    dataset = get_or_create_dataset(client, "agent-registry-retrieval", cases,
                                    "Registre de 20 paquets : bon paquet retrouvé pour la tâche ?")
    collected = []

    def task(input, expected):
        start = time.perf_counter()
        update = agent.find_installed_packages({"current_task": input["task"]})
        search = agent.advisor.last_search
        output = {
            "retrieved": [name for name, _ in search.get("retrieved", [])],
            "scores": search.get("retrieved", []),
            "selected": [h["name"] for h in update["package_hints"]],
            "hints": update["package_hints"],
            "time_s": round(time.perf_counter() - start, 2),
        }
        collected.append({"output": output, "expected": expected})
        return output

    def rank(output, expected):
        names = (output or {}).get("retrieved", [])
        return names.index(expected["expected"]) + 1 if expected["expected"] in names else None

    def rag_top1(output, expected):
        if expected["expected"] is None:
            return (None, "négatif")
        return rank(output, expected) == 1

    def rag_topk(output, expected):
        if expected["expected"] is None:
            return (None, "négatif")
        return rank(output, expected) is not None

    def rag_rr(output, expected):
        if expected["expected"] is None:
            return (None, "négatif")
        r = rank(output, expected)
        return 1 / r if r else 0.0

    def selected_ok(output, expected):
        selected = (output or {}).get("selected", [])
        if expected["expected"] is None:
            return not selected
        return expected["expected"] in selected

    def selected_precision(output, expected):
        selected = (output or {}).get("selected", [])
        if not selected:
            return 1.0 if expected["expected"] is None else 0.0
        return sum(s == expected["expected"] for s in selected) / len(selected)

    def false_positive(output, expected):
        selected = (output or {}).get("selected", [])
        return any(s != expected["expected"] for s in selected)

    def time_s(output):
        return float((output or {}).get("time_s", 0))

    evaluators = {"rag_top1": rag_top1, "rag_topk": rag_topk, "rag_rr": rag_rr,
                  "selected_ok": selected_ok, "selected_precision": selected_precision,
                  "false_positive": false_positive}
    experiment = client.experiments.run_experiment(
        dataset=dataset, task=task, evaluators={**evaluators, "time_s": time_s},
        experiment_name=f"{args.label} {datetime.now():%H:%M}",
        experiment_metadata=metadata, timeout=600,
    )
    summary = summarize("retrieval", collected, evaluators)
    print(f"Expérience : {PHOENIX_URL}/datasets/{dataset.id}/compare?experimentId={experiment['experiment_id']}")
    misses = [(c["expected"]["expected"], c["output"]["scores"], c["output"]["selected"])
              for c in collected if not selected_ok(c["output"], c["expected"])]
    for miss in misses:
        print("  ✖ attendu=%s | RAG=%s | retenus=%s" % miss)
    return {"summary": summary, "cases": collected}


# ------------------------------------------------------------------
# Test 2 : usage
# ------------------------------------------------------------------

def run_usage(agent: AgentGraph, client, cases, args, metadata) -> dict:
    """Le modèle utilise-t-il correctement le paquet dans prepare_task ?"""

    cases = [c for c in cases if c["expected"]]
    dataset = get_or_create_dataset(client, "agent-registry-usage", cases,
                                    "Registre de 20 paquets : commandes de prepare_task avec le paquet")
    environment = agent.environment()
    judge = None
    if args.judge_model:
        from agent.llm import LocalLLM
        judge = LocalLLM(replace(agent.config, model=args.judge_model))
    collected = []

    def task(input, expected):
        start = time.perf_counter()
        current = {"id": input["case_id"], **input["task"], "depends_on": [], "resolution_depth": 0}
        accepted = None
        if args.hints == "rag":
            hints = agent.find_installed_packages({"current_task": current})["package_hints"]
        else:
            record = agent.packages.get(expected["expected"])
            decision = agent.advisor.match(current, [record])[0]
            accepted = decision.usable
            usage = record.usages[-1]
            hints = [{
                "name": record.name,
                "description": decision.description or record.description,
                "example": decision.example if decision.usable and decision.example else usage.command,
                "mode": "cli" if record.shell_use else "python",
                "tool_file": usage.tool_file,
            }]
        state = {"current_task": current, "environment": environment, "completed": {},
                 "package_hints": hints}
        plan = agent.prepare_task(state)["current_plan"]
        output = {"hints": hints, "matcher_accepts": accepted, "commands": plan["commands"],
                  "expected_output": plan["expected"], "time_s": round(time.perf_counter() - start, 2)}
        if judge is not None:
            verdict = judge.structured(
                Judgement,
                "You check whether shell commands correctly perform a task using an installed package. "
                "The package and its tool files are already installed; do not penalize that.",
                json.dumps({"task": input["task"], "package": hints, "commands": plan["commands"]},
                           ensure_ascii=False, indent=2),
            )
            output["judge"] = verdict.model_dump()
        collected.append({"output": output, "expected": expected, "task": input["task"]})
        return output

    def joined(output):
        return "\n".join((output or {}).get("commands", []))

    def matcher_accepts(output):
        value = (output or {}).get("matcher_accepts")
        return (None, "rag") if value is None else bool(value)

    def not_impossible(output):
        return (output or {}).get("expected_output") != "IMPOSSIBLE" and bool((output or {}).get("commands"))

    def uses_package(output, expected):
        return expected["marker"].lower() in joined(output).lower()

    def no_reinstall(output):
        return not re.search(r"\b(pip3?|apt(-get)?)\s+install\b", joined(output))

    def bash_syntax(output):
        commands = (output or {}).get("commands", [])
        if not commands:
            return False
        return all(subprocess.run(["bash", "-n", "-c", c], capture_output=True).returncode == 0
                   for c in commands)

    def io_files(output, input):
        names = files_in(input["task"].get("input", "")) + files_in(input["task"].get("output", ""))
        text = joined(output)
        return all(n in text for n in names) if names else (None, "aucun fichier")

    def judge_ok(output):
        verdict = (output or {}).get("judge")
        return (None, "pas de juge") if verdict is None else bool(verdict["correct"])

    def time_s(output):
        return float((output or {}).get("time_s", 0))

    evaluators = {"matcher_accepts": matcher_accepts, "not_impossible": not_impossible,
                  "uses_package": uses_package, "no_reinstall": no_reinstall,
                  "bash_syntax": bash_syntax, "io_files": io_files, "judge": judge_ok}
    experiment = client.experiments.run_experiment(
        dataset=dataset, task=task, evaluators={**evaluators, "time_s": time_s},
        experiment_name=f"{args.label} {datetime.now():%H:%M}",
        experiment_metadata={**metadata, "hints": args.hints, "judge_model": args.judge_model},
        timeout=600,
    )

    # Les évaluateurs ont des signatures différentes : moyennes locales via des adaptateurs.
    local = {
        "matcher_accepts": lambda o, e: matcher_accepts(o),
        "not_impossible": lambda o, e: not_impossible(o),
        "uses_package": uses_package,
        "no_reinstall": lambda o, e: no_reinstall(o),
        "bash_syntax": lambda o, e: bash_syntax(o),
        "judge": lambda o, e: judge_ok(o),
    }
    for item in collected:
        item["output"]["_io"] = io_files(item["output"], {"task": item["task"]})
    local["io_files"] = lambda o, e: o["_io"]
    summary = summarize("usage", collected, local)
    print(f"Expérience : {PHOENIX_URL}/datasets/{dataset.id}/compare?experimentId={experiment['experiment_id']}")
    for item in collected:
        if not uses_package(item["output"], item["expected"]) or not not_impossible(item["output"]):
            print(f"  ✖ {item['expected']['expected']}: {item['output']['expected_output']} | "
                  f"{item['output']['commands']}")
    return {"summary": summary, "cases": collected}


def main() -> int:
    """Point d'entrée du bench."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["retrieval", "usage", "all"], default="all")
    parser.add_argument("--label", required=True, help="nom de l'expérience Phoenix")
    parser.add_argument("--set", action="append", default=[], help="option=valeur de AgentConfig")
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--hints", choices=["oracle", "rag"], default="oracle",
                        help="usage : indices du bon paquet (oracle) ou du pipeline complet (rag)")
    parser.add_argument("--judge-model", default=None, help="usage : modèle Ollama juge (ex. qwen3.5:9b)")
    args = parser.parse_args()

    overrides = parse_overrides(parser, args.set)
    for key, value in list(overrides.items()):
        if isinstance(value, str):
            try:
                overrides[key] = float(value)
            except ValueError:
                pass
    register_tracing("agent-registry-bench")
    agent = make_agent(args.model, overrides)
    client = Client(base_url=PHOENIX_URL)
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    metadata = {"model": args.model, **overrides}

    results = {}
    if args.test in ("retrieval", "all"):
        results["retrieval"] = run_retrieval(agent, client, cases, args, metadata)
    if args.test in ("usage", "all"):
        results["usage"] = run_usage(agent, client, cases, args, metadata)

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"registry_{args.label}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nRésultats : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
