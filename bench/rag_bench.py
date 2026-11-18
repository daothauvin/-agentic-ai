"""Bench du RAG de documentation (sections vs multi-niveau), suivi dans Phoenix.

Pour chaque package de ``bench/rag_cases.json`` (doc PyPI figée dans
``bench/fixtures/rag/``), on lance les retrievals *de production*
(``ToolSearchService.extract_installation_context`` / ``extract_usage_context``)
et on vérifie que le contexte renvoyé contient le passage attendu (regex
``install`` / ``usage``). La métrique ne dépend donc pas du découpage.

Chaque variante devient une expérience Phoenix sur le dataset ``agent-rag-docs`` :
la sortie de chaque cas contient les contextes envoyés au LLM et le détail des
retrievals (requête, rang, score, section, phrase qui a fait remonter le chunk).

Évaluateurs :
    install_hit / usage_hit  passage attendu présent dans le contexte de production
    install_mrr / usage_mrr  1/rang du premier chunk contenant le passage (top-10)
    install_cmd              (--llm-install) le LLM renvoie la bonne commande pip
    ctx_install / ctx_usage  taille du contexte envoyé au LLM (car.)

Variantes : ``base`` (sections Markdown) et ``ml`` (sections + phrases, défaut).
``--k`` fait varier le nombre de chunks donnés au LLM (install : k, usage : k par requête).

Prérequis : PHOENIX_WORKING_DIR=.phoenix phoenix serve

    python bench/rag_bench.py --variants base ml --label essai
    python bench/rag_bench.py --variants ml --llm-install
    python bench/rag_bench.py --no-phoenix            # rapport JSON seul
    python bench/rag_bench.py --variants ml --k 1 2 3 4 --llm-install --label topk
"""

import argparse
import inspect
import json
import re
import time
from dataclasses import replace
from datetime import datetime
from types import SimpleNamespace

from common import PHOENIX_URL, ROOT, register_tracing

from agent.config import AgentConfig
from agent.llm import LocalLLM
from agent.logger import get_logger, setup_logging
from agent.prompts.loader import PromptLoader
from agent.rag import PackageRAG
from agent.toolsearch import ToolSearchService

CASES = ROOT / "bench/rag_cases.json"
FIXTURES = ROOT / "bench/fixtures/rag"
CACHE = ROOT / "bench/.rag_cache"
RESULTS = ROOT / "bench/results"
DATASET = "agent-rag-docs"
VARIANTS = {"base": False, "ml": True}
INSTALL_QUERY = "How to install the package? pip install command"
logger = get_logger("bench.rag")


def usage_queries(task: dict) -> list[str]:
    """Requêtes d'usage identiques à ``ToolSearchService.extract_usage_context``."""

    return [
        f"How to use the package to: {task.get('description', '')}",
        f"Usage example: input {task.get('input', '')}, output {task.get('output', '')}",
        "Quickstart usage example code / command line",
    ]


def reciprocal_rank(rag: PackageRAG, index, queries: list[str], pattern: str | None) -> float | None:
    """Meilleur 1/rang (top-10) d'un chunk contenant ``pattern``, sur toutes les requêtes.

    Args:
        rag: RAG utilisé (pour l'embedding des requêtes).
        index: Index du package.
        queries: Requêtes à évaluer.
        pattern: Regex du passage attendu ; ``None`` = non évaluable.

    Returns:
        Réciproque du meilleur rang, 0 si absent du top-10, ``None`` si non évaluable.
    """

    if not pattern:
        return None
    regex = re.compile(pattern, re.I)
    best = 0.0
    for row in index.search(rag.embedder.embed(queries), 10):
        for rank, (chunk, _) in enumerate(row, 1):
            if regex.search(chunk.render()):
                best = max(best, 1.0 / rank)
                break
    return best


def make_task(rag: PackageRAG, llm: LocalLLM, prompts: PromptLoader, config: AgentConfig,
              llm_install: bool):
    """Fonction de tâche : retrievals de production sur un cas, sortie détaillée.

    Returns:
        Callable ``task(input, expected) -> dict`` compatible Phoenix.
    """

    service = SimpleNamespace(rag=rag, llm=llm, prompts=prompts, config=config)

    def task(input, expected):
        name = input["package"]
        pypi_json = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
        t0 = time.perf_counter()
        index = rag.index_package(pypi_json)
        t_index = time.perf_counter() - t0

        t0 = time.perf_counter()
        install_ctx = ToolSearchService.extract_installation_context(service, index)
        install_retrieval = rag.last_retrieval
        usage_ctx = ToolSearchService.extract_usage_context(service, index, input["task"])
        usage_retrieval = rag.last_retrieval
        t_retrieval = time.perf_counter() - t0

        output = {
            "package": name,
            "chunks": len(index.chunks), "entries": len(index.entries),
            "t_index_s": round(t_index, 2), "t_retrieval_s": round(t_retrieval, 2),
            "install_mrr": reciprocal_rank(rag, index, [INSTALL_QUERY], expected.get("install")),
            "usage_mrr": reciprocal_rank(rag, index, usage_queries(input["task"]),
                                         expected["usage"]),
            "install_retrieval": install_retrieval, "usage_retrieval": usage_retrieval,
            "install_context": install_ctx, "usage_context": usage_ctx,
        }
        if llm_install:
            try:
                info = ToolSearchService.analyze_installation(service, install_ctx)
                output["install_commands"] = info.install_commands
            except Exception as exc:  # sortie structurée invalide = échec
                logger.warning("analyze_installation %s : %s", name, exc)
                output["install_commands"] = []
        logger.info("[%s] retrieval install : %s", name, install_retrieval)
        logger.info("[%s] retrieval usage : %s", name, usage_retrieval)
        return output

    return task


# ------------------------------------------------------------------ évaluateurs

def _found(pattern: str | None, text: str) -> bool | None:
    return None if not pattern else bool(re.search(pattern, text or "", re.I))


def install_hit(output, expected):
    """Commande d'installation attendue présente dans le contexte « installation »."""

    return _found(expected.get("install"), output["install_context"])


def usage_hit(output, expected):
    """Exemple d'usage attendu présent dans le contexte « usage »."""

    return _found(expected["usage"], output["usage_context"])


def install_mrr(output):
    """1/rang du premier chunk contenant la commande d'installation."""

    return output["install_mrr"]


def usage_mrr(output):
    """1/rang du premier chunk contenant l'exemple d'usage."""

    return output["usage_mrr"]


def install_cmd(output, expected):
    """Le LLM a extrait la bonne commande pip du contexte."""

    return _found(expected.get("install"), " ; ".join(output.get("install_commands", [])))


def ctx_install(output):
    """Taille du contexte « installation » (car.)."""

    return len(output["install_context"])


def ctx_usage(output):
    """Taille du contexte « usage » (car.)."""

    return len(output["usage_context"])


def for_phoenix(fn):
    """Adapte un évaluateur : Phoenix refuse ``None`` (cas non évaluable)."""

    params = list(inspect.signature(fn).parameters)

    def wrapper(output, expected):
        value = fn(*[{"output": output, "expected": expected}[p] for p in params])
        return value if value is not None else (None, "n/a", "non évaluable pour ce cas")

    wrapper.__name__ = fn.__name__
    return wrapper


# ------------------------------------------------------------------ main

def summarize(outputs: list[tuple[dict, dict]], llm_install: bool) -> dict:
    """Moyennes locales des évaluateurs (même calcul que Phoenix)."""

    evaluators = {"install_hit": install_hit, "install_mrr": lambda o, e: install_mrr(o),
                  "usage_hit": usage_hit, "usage_mrr": lambda o, e: usage_mrr(o),
                  "ctx_install": lambda o, e: ctx_install(o),
                  "ctx_usage": lambda o, e: ctx_usage(o)}
    if llm_install:
        evaluators["install_cmd"] = install_cmd
    summary = {}
    for key, fn in evaluators.items():
        values = [float(v) for o, e in outputs if (v := fn(o, e)) is not None]
        summary[key] = round(sum(values) / len(values), 3) if values else None
    summary["t_retrieval_s"] = round(
        sum(o["t_retrieval_s"] for o, _ in outputs) / len(outputs), 3)
    return summary


def main() -> int:
    """Lance les variantes (expériences Phoenix) et écrit le rapport JSON."""

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--k", nargs="+", type=int, default=None,
                        help="top-k à tester (défaut : valeurs d'AgentConfig)")
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--label", default="rag")
    parser.add_argument("--llm-install", action="store_true",
                        help="mesure aussi la commande d'installation renvoyée par le LLM")
    parser.add_argument("--no-phoenix", action="store_true", help="rapport JSON local seulement")
    args = parser.parse_args()

    config = AgentConfig(ollama_url="http://localhost:11434", model=args.model, ai_id=0,
                         workspace_root=CACHE, prompts_dir=ROOT / "agent/prompts")
    log_file = setup_logging(CACHE / "logs", console_level="WARNING")
    print(f"Log : {log_file}")
    llm, prompts = LocalLLM(config), PromptLoader(config.prompts_dir)
    cases = json.loads(CASES.read_text(encoding="utf-8"))

    client = dataset = None
    if not args.no_phoenix:
        from phoenix.client import Client

        register_tracing("agent-rag-bench")
        client = Client(base_url=PHOENIX_URL)
        try:
            dataset = client.datasets.get_dataset(dataset=DATASET)
        except Exception:
            dataset = client.datasets.create_dataset(
                name=DATASET,
                dataset_description="RAG doc PyPI : passages install/usage attendus (bench/rag_cases.json)",
                inputs=[{"package": c["package"], "task": c["task"]} for c in cases],
                outputs=[{"install": c["install"], "usage": c["usage"]} for c in cases],
            )

    report = {"model": args.model, "date": datetime.now().isoformat(timespec="seconds"),
              "variants": []}
    runs = [(v, k) for v in args.variants for k in (args.k or [None])]
    for variant, k in runs:
        run_config = config if k is None else replace(config, rag_install_k=k, rag_usage_k=k)
        name = f"{variant} k={run_config.rag_install_k}/{run_config.rag_usage_k}"
        print(f"== {name}", flush=True)
        rag = PackageRAG(config.ollama_url, CACHE / ".rag", multi_level=VARIANTS[variant])
        task = make_task(rag, llm, prompts, run_config, args.llm_install)
        outputs = []

        def recorded(input, expected, _task=task, _outputs=outputs):
            out = _task(input, expected)
            _outputs.append((out, expected))
            print(f"  {variant:>5} {input['package']:<22} install={install_hit(out, expected)!s:<5} "
                  f"usage={usage_hit(out, expected)!s:<5}", flush=True)
            return out

        entry = {"variant": name, "multi_level": VARIANTS[variant],
                 "install_k": run_config.rag_install_k, "usage_k": run_config.rag_usage_k}
        if client is not None:
            evaluators = {"install_hit": install_hit, "install_mrr": install_mrr,
                          "usage_hit": usage_hit, "usage_mrr": usage_mrr,
                          "ctx_install": ctx_install, "ctx_usage": ctx_usage}
            if args.llm_install:
                evaluators["install_cmd"] = install_cmd
            experiment = client.experiments.run_experiment(
                dataset=dataset, task=recorded,
                evaluators={k: for_phoenix(f) for k, f in evaluators.items()},
                experiment_name=f"{args.label} {name} {datetime.now():%H:%M}",
                experiment_metadata={"variant": variant, "multi_level": VARIANTS[variant],
                                     "install_k": run_config.rag_install_k,
                                     "usage_k": run_config.rag_usage_k,
                                     "model": args.model, "llm_install": args.llm_install},
                timeout=600, print_summary=False,
            )
            entry["phoenix"] = (f"{PHOENIX_URL}/datasets/{dataset.id}/compare"
                                f"?experimentId={experiment['experiment_id']}")
        else:
            for c in cases:
                recorded({"package": c["package"], "task": c["task"]},
                         {"install": c["install"], "usage": c["usage"]})
        entry["summary"] = summarize(outputs, args.llm_install)
        entry["cases"] = [{k: v for k, v in o.items() if not k.endswith("_context")}
                          for o, _ in outputs]
        report["variants"].append(entry)

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"rag_{args.label}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    keys = list(report["variants"][0]["summary"])
    print("\n| variante | " + " | ".join(keys) + " |")
    print("|---" * (len(keys) + 1) + "|")
    for v in report["variants"]:
        print(f"| {v['variant']} | " + " | ".join(str(v["summary"][k]) for k in keys) + " |")
    for v in report["variants"]:
        if "phoenix" in v:
            print(f"Phoenix {v['variant']} : {v['phoenix']}")
    if dataset is not None:
        print(f"Comparaison : {PHOENIX_URL}/datasets/{dataset.id}/experiments")
    print(f"Résultats : {out}\nLog : {log_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
