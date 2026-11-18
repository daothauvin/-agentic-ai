"""Bench étape par étape de la recherche de packages, suivi dans Phoenix.

Chaque étape de ToolSearchService.search_packages est testée seule, sur des
entrées figées : un run de référence (`--snapshot`, configuration par défaut)
enregistre l'entrée et la sortie de chaque étape dans bench/fixtures/stages.json.
Une étape rejouée ne dépend donc ni du bruit des étapes précédentes, ni de
SearXNG (sauf l'étape `search`, qui l'interroge).

Étapes (une expérience Phoenix par étape, dataset `etape-<nom>`) :
    identify   demande + tâche        -> spécification de l'outil (LLM)
    queries    spécification figée    -> requêtes de recherche (LLM)        [option 1]
    search     requêtes figées        -> résultats SearXNG
    guess      spécification figée    -> noms proposés, vérifiés sur PyPI  [option 2]
    normalize  résultats bruts figés  -> URL normalisées, dédoublonnées    [option 5]
    filter     candidats normalisés   -> candidats après filtre PyPI       [option 4]
    rank       candidats filtrés      -> classement LLM                    [option 3]

    python bench/stage_bench.py --snapshot              # (re)crée les entrées figées
    python bench/stage_bench.py --label defaut          # toutes les étapes
    python bench/stage_bench.py --stage rank --label essai --set model=...
"""

import argparse
import json
import time
from datetime import datetime

from common import (CASES, PHOENIX_URL, ROOT, environment, get_or_create_dataset, make_service,
                    parse_overrides, register_tracing, scores)
from phoenix.client import Client

from agent.models import Package
from agent.toolsearch import pypi_name_from_url

FIXTURE = ROOT / "bench/fixtures/stages.json"
STAGES = ["identify", "queries", "search", "guess", "normalize", "filter", "rank"]
TOP_N = 3


def names_of(packages: list[Package]) -> list[str]:
    return [pypi_name_from_url(p.url) or p.url for p in packages]


def dump(packages: list[Package]) -> list[dict]:
    return [p.model_dump(mode="json") for p in packages]


def load(packages: list[dict]) -> list[Package]:
    return [Package.model_validate(p) for p in packages]


# ------------------------------------------------------------------
# Entrées figées
# ------------------------------------------------------------------

def snapshot(model: str, cases: list[dict]) -> dict:
    """Run de référence : enregistre l'entrée/sortie de chaque étape pour chaque cas."""

    service = make_service(model, {}, prefix="stage_snapshot_")
    fixture = {"created": datetime.now().isoformat(timespec="seconds"), "model": model, "cases": {}}
    for case in cases:
        start = time.perf_counter()
        state = {"request": case["request"], "current_task": case["task"], "environment": environment()}
        spec = service.identify_required_tool(state)
        queries = service.build_package_search_queries(spec)
        raw, hits = service.run_search_queries(queries)
        guessed = service.guess_package_names(spec)
        normalized = service.normalize_and_sort(guessed + raw, hits)
        filtered = service.filter_by_metadata(normalized)
        ranked = service.rank_candidates(filtered, spec) if filtered else []
        fixture["cases"][case["id"]] = {
            "spec": spec, "queries": queries, "hits": hits, "raw": dump(raw),
            "guessed": dump(guessed), "normalized": dump(normalized),
            "filtered": dump(filtered), "ranked": names_of(ranked),
        }
        print(f"snapshot {case['id']:<14} {time.perf_counter() - start:5.1f}s  "
              f"brut={len(raw)} devinés={len(guessed)} normalisés={len(normalized)} "
              f"filtrés={len(filtered)} top={names_of(ranked)[:TOP_N]}", flush=True)
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(fixture, ensure_ascii=False, indent=1), encoding="utf-8")
    return fixture


# ------------------------------------------------------------------
# Tâches : une par étape, à partir des entrées figées
# ------------------------------------------------------------------

def make_task(stage: str, service, frozen: dict):
    def run(input) -> dict:
        f = frozen[input["case_id"]]
        if stage == "identify":
            state = {"request": input["request"], "current_task": input["task"], "environment": environment()}
            return {"spec": service.identify_required_tool(state)}
        if stage == "queries":
            return {"spec": f["spec"], "queries": service.build_package_search_queries(f["spec"])}
        if stage == "search":
            raw, _ = service.run_search_queries(f["queries"])
            return {"queries": f["queries"], "names": names_of(raw)}
        if stage == "guess":
            return {"spec": f["spec"], "names": names_of(service.guess_package_names(f["spec"]))}
        if stage == "normalize":
            before = load(f["guessed"]) + load(f["raw"])
            return {"input_names": names_of(before), "names": names_of(service.normalize_and_sort(before, f["hits"]))}
        if stage == "filter":
            before = load(f["normalized"])
            return {"input_names": names_of(before), "names": names_of(service.filter_by_metadata(before))}
        if stage == "rank":
            before = load(f["filtered"])
            ranked = service.rank_candidates(before, f["spec"]) if before else []
            return {"input_names": names_of(before), "names": names_of(ranked)}
        raise ValueError(stage)

    def task(input) -> dict:
        start = time.perf_counter()
        output = run(input)
        output["time_s"] = round(time.perf_counter() - start, 2)
        return output

    return task


# ------------------------------------------------------------------
# Évaluateurs
# ------------------------------------------------------------------

def first_rank(names: list[str], expected: dict) -> int | None:
    wanted = set(expected["expected"])
    return next((i + 1 for i, n in enumerate(names) if n in wanted), None)


def hit_top(output, expected) -> bool:
    rank = first_rank(output.get("names", []), expected)
    return bool(rank and rank <= TOP_N)


def hit_any(output, expected) -> bool:
    return first_rank(output.get("names", []), expected) is not None


def reciprocal_rank(output, expected) -> float:
    rank = first_rank(output.get("names", []), expected)
    return 1 / rank if rank else 0.0


def expected_kept(output, expected) -> float:
    """1 si l'étape n'a perdu aucun package attendu présent en entrée."""

    wanted = set(expected["expected"])
    before = wanted & set(output.get("input_names", []))
    return float(before <= set(output.get("names", [])))


def ceiling(output, expected) -> bool:
    """Un package attendu est-il présent en entrée ? (meilleur score possible de l'étape)"""

    return bool(set(expected["expected"]) & set(output.get("input_names", [])))


def n_out(output) -> float:
    return float(len(output.get("names", [])))


def n_removed(output) -> float:
    return float(len(output.get("input_names", [])) - len(output.get("names", [])))


def time_s(output) -> float:
    return float(output.get("time_s", 0))


def spec_complete(output) -> bool:
    spec = output.get("spec") or {}
    return all(str(spec.get(k, "")).strip() for k in ("name", "description", "input", "output"))


def queries_count(output) -> float:
    return float(len(output.get("queries", [])))


def queries_short(output) -> bool:
    queries = output.get("queries", [])
    return bool(queries) and all(len(q.split()) <= 8 for q in queries)


def queries_distinct(output) -> bool:
    queries = [q.lower() for q in output.get("queries", [])]
    return len(set(queries)) == len(queries)


EVALUATORS = {
    "identify": {"spec_complete": spec_complete},
    "queries": {"n_requetes": queries_count, "requetes_courtes": queries_short, "requetes_distinctes": queries_distinct},
    "search": {"hit_any": hit_any, "reciprocal_rank": reciprocal_rank, "n_resultats": n_out},
    "guess": {f"hit_top{TOP_N}": hit_top, "hit_any": hit_any, "n_valides": n_out},
    "normalize": {"attendu_conserve": expected_kept, "n_sortie": n_out, "n_retires": n_removed},
    "filter": {"attendu_conserve": expected_kept, f"hit_top{TOP_N}": hit_top, "n_retires": n_removed},
    "rank": {f"hit_top{TOP_N}": hit_top, "reciprocal_rank": reciprocal_rank, "plafond": ceiling},
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=["all", *STAGES])
    parser.add_argument("--label", default="defaut", help="suffixe du nom des expériences")
    parser.add_argument("--set", action="append", default=[], help="option=valeur de AgentConfig")
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--snapshot", action="store_true", help="recrée les entrées figées avant le bench")
    args = parser.parse_args()
    overrides = parse_overrides(parser, args.set)

    register_tracing("agent-stage-bench")
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    if args.snapshot or not FIXTURE.exists():
        fixture = snapshot(args.model, cases)
    else:
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    frozen = fixture["cases"]
    cases = [c for c in cases if c["id"] in frozen]

    service = make_service(args.model, overrides, prefix="stage_bench_")
    client = Client(base_url=PHOENIX_URL)
    summary = []
    for stage in STAGES if args.stage == "all" else [args.stage]:
        dataset = get_or_create_dataset(client, f"etape-{stage}", f"Étape « {stage} » de la recherche de packages", cases)
        evaluators = {**EVALUATORS[stage], "time_s": time_s}
        experiment = client.experiments.run_experiment(
            dataset=dataset,
            task=make_task(stage, service, frozen),
            evaluators=evaluators,
            experiment_name=f"{stage} {args.label} {datetime.now():%H:%M}",
            experiment_metadata={"stage": stage, "model": args.model, "fixture": fixture["created"], **overrides},
            timeout=600,
            print_summary=False,
        )
        url = f"{PHOENIX_URL}/datasets/{dataset.id}/compare?experimentId={experiment['experiment_id']}"
        summary.append((stage, scores(experiment), url))

    print(f"\n== Bench par étape « {args.label} » (entrées figées du {fixture['created']})")
    for stage, values, url in summary:
        metrics = "  ".join(f"{k}={v:.2f}" for k, v in values.items())
        print(f"{stage:<10} {metrics}\n           {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
