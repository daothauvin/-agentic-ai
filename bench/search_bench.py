"""Bench de la recherche de packages (sans installation ni indexation).

Pour chaque cas de bench/search_cases.json : identification de l'outil,
recherche, filtres et classement, puis comparaison avec les packages attendus.

    python bench/search_bench.py --label baseline --set search_prerank=false
"""

import argparse
import json
import sys
import tempfile
import time
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.config import AgentConfig  # noqa: E402
from agent.llm import LocalLLM  # noqa: E402
from agent.logger import setup_logging  # noqa: E402
from agent.prompts.loader import PromptLoader  # noqa: E402
from agent.toolsearch import ToolSearchService, pypi_name_from_url  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def parse_value(raw: str):
    return {"true": True, "false": False}.get(raw.lower(), int(raw) if raw.isdigit() else raw)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--set", action="append", default=[], help="option=valeur de AgentConfig")
    parser.add_argument("--cases", default=str(ROOT / "bench/search_cases.json"))
    parser.add_argument("--only", default="", help="ids de cas séparés par des virgules")
    parser.add_argument("--out", default=str(ROOT / "bench/results"))
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--replay", default="", help="label d'un résultat précédent : rejoue ses candidats au lieu d'appeler SearXNG")
    parser.add_argument("--pause", type=float, default=0, help="pause entre deux cas (s), contre la limitation des moteurs")
    args = parser.parse_args()

    workspace = Path(tempfile.mkdtemp(prefix="search_bench_"))
    config = AgentConfig(
        ollama_url="http://localhost:11434", model=args.model, ai_id=0,
        workspace_root=workspace, prompts_dir=ROOT / "agent/prompts",
    )
    names = {f.name for f in fields(AgentConfig)}
    overrides = {}
    for item in args.set:
        key, value = item.split("=", 1)
        if key not in names:
            parser.error(f"option inconnue: {key}")
        overrides[key] = parse_value(value)
    config = replace(config, **overrides)
    config.project_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(config.project_dir / ".agent/logs", console_level="WARNING")

    agent = SimpleNamespace(llm=LocalLLM(config), prompts=PromptLoader(config.prompts_dir), config=config, tools=None)
    service = ToolSearchService(agent)
    environment = {"os": "Linux x86_64", "python": "%d.%d" % sys.version_info[:2], "packages_and_tools": {}}

    replay = {}
    if args.replay:
        from agent.models import Package
        stored = json.loads((Path(args.out) / f"{args.replay}.json").read_text(encoding="utf-8"))
        replay = {r["id"]: r["all"] for r in stored["rows"]}
        current = {"id": None}
        # Une seule "requête" par cas, dont les résultats sont ceux enregistrés.
        service.build_package_search_queries = lambda spec: ["replay"]
        service.search_web = lambda query: current["id"]
        service.parse_search_results = lambda case_id: [
            Package(title=n, url=f"https://pypi.org/project/{n}/", content="",
                    package_use=None, package_install=None, usage_context=None, tree=None)
            for n in replay.get(case_id, [])
        ]

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    if args.only:
        cases = [c for c in cases if c["id"] in args.only.split(",")]

    rows = []
    top_n = config.max_packages_analyzed
    for index, case in enumerate(cases):
        if index and args.pause:
            time.sleep(args.pause)
        if replay:
            current["id"] = case["id"]
        state = {"request": case["request"], "current_task": case["task"], "environment": environment}
        start = time.perf_counter()
        error = None
        try:
            spec = service.identify_required_tool(state)
            packages = service.search_packages(spec)
        except Exception as exc:  # le bench continue sur les autres cas
            spec, packages, error = {}, [], f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - start
        found = [pypi_name_from_url(p.url) or p.url for p in packages]
        expected = set(case["expected"])
        first = next((i + 1 for i, n in enumerate(found) if n in expected), None)
        row = {
            "id": case["id"], "time_s": round(elapsed, 1), "n_candidates": len(found),
            "top": found[:top_n], "all": found, "hit_top": bool(first and first <= top_n),
            "hit_any": first is not None, "first_rank": first, "error": error,
            "tool": spec.get("description"),
        }
        rows.append(row)
        print(f"{case['id']:<14} {elapsed:6.1f}s  top{top_n}={'OK ' if row['hit_top'] else 'NON'} "
              f"any={'OK ' if row['hit_any'] else 'NON'} rang={first}  {found[:top_n]}"
              + (f"  ERREUR {error}" if error else ""), flush=True)

    n = len(rows)
    summary = {
        "label": args.label, "overrides": overrides, "cases": n,
        "hit_top": sum(r["hit_top"] for r in rows), "hit_any": sum(r["hit_any"] for r in rows),
        "avg_time_s": round(sum(r["time_s"] for r in rows) / max(n, 1), 1),
        "errors": sum(1 for r in rows if r["error"]),
    }
    print(f"== {args.label}: top{top_n} {summary['hit_top']}/{n} | any {summary['hit_any']}/{n} | "
          f"{summary['avg_time_s']}s/cas | erreurs {summary['errors']}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.label}.json").write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
