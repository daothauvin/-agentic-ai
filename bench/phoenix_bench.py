"""Bench de la recherche de packages, suivi dans Arize Phoenix VVV

Même pipeline que bench/search_bench.py, mais chaque run devient une
« expérience » Phoenix sur le dataset construit depuis bench/search_cases.json :
traces complètes (appels LLM LangChain), scores par cas et comparaison
entre expériences dans l'interface web http://localhost:6006.

Prérequis : serveur Phoenix lancé (PHOENIX_WORKING_DIR=.phoenix phoenix serve).

    python bench/phoenix_bench.py --label defaut --pause 5
    python bench/phoenix_bench.py --label sans_prerank --set search_prerank=false
"""

import argparse
import json
import time
from datetime import datetime

from common import (CASES, PHOENIX_URL, environment, get_or_create_dataset, make_service,
                    parse_overrides, register_tracing)
from phoenix.client import Client

from agent.toolsearch import pypi_name_from_url

DATASET = "agent-search-packages"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="nom de l'expérience Phoenix")
    parser.add_argument("--set", action="append", default=[], help="option=valeur de AgentConfig")
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--pause", type=float, default=0, help="pause entre deux cas (s)")
    args = parser.parse_args()

    overrides = parse_overrides(parser, args.set)
    register_tracing("agent-search-bench")
    service = make_service(args.model, overrides, prefix="phoenix_bench_")
    top_n = service.config.max_packages_analyzed
    client = Client(base_url=PHOENIX_URL)
    dataset = get_or_create_dataset(client, DATASET, "Recherche de packages PyPI : bench/search_cases.json",
                                    json.loads(CASES.read_text(encoding="utf-8")))
    calls = {"n": 0}

    def task(input):
        if calls["n"] and args.pause:
            time.sleep(args.pause)
        calls["n"] += 1
        state = {"request": input["request"], "current_task": input["task"], "environment": environment()}
        start = time.perf_counter()
        spec = service.identify_required_tool(state)
        packages = service.search_packages(spec)
        found = [pypi_name_from_url(p.url) or p.url for p in packages]
        return {"tool": spec.get("description"), "top": found[:top_n], "candidates": found,
                "time_s": round(time.perf_counter() - start, 1)}

    def first_rank(output, expected):
        wanted = set(expected["expected"])
        return next((i + 1 for i, n in enumerate((output or {}).get("candidates", [])) if n in wanted), None)

    def hit_top(output, expected) -> bool:
        rank = first_rank(output, expected)
        return bool(rank and rank <= top_n)

    def hit_any(output, expected) -> bool:
        return first_rank(output, expected) is not None

    def reciprocal_rank(output, expected) -> float:
        rank = first_rank(output, expected)
        return 1 / rank if rank else 0.0

    def time_s(output) -> float:
        return float((output or {}).get("time_s", 0))

    experiment = client.experiments.run_experiment(
        dataset=dataset,
        task=task,
        evaluators={f"hit_top{top_n}": hit_top, "hit_any": hit_any,
                    "reciprocal_rank": reciprocal_rank, "time_s": time_s},
        experiment_name=f"{args.label} {datetime.now():%H:%M}",
        experiment_metadata={"model": args.model, **overrides},
        timeout=600,
    )
    print(f"\nDataset    : {PHOENIX_URL}/datasets/{dataset.id}/experiments")
    print(f"Expérience : {PHOENIX_URL}/datasets/{dataset.id}/compare?experimentId={experiment['experiment_id']}")
    print("Traces     : bouton « trace » de chaque ligne de l'expérience (appels LLM inclus)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
