"""Outils communs aux benchs : configuration, service de recherche, Phoenix."""

import sys
import tempfile
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.config import AgentConfig  # noqa: E402
from agent.llm import LocalLLM  # noqa: E402
from agent.logger import setup_logging  # noqa: E402
from agent.prompts.loader import PromptLoader  # noqa: E402
from agent.toolsearch import ToolSearchService  # noqa: E402

PHOENIX_URL = "http://localhost:6006"
CASES = ROOT / "bench/search_cases.json"


def parse_value(raw: str):
    return {"true": True, "false": False}.get(raw.lower(), int(raw) if raw.isdigit() else raw)


def parse_overrides(parser, items: list[str]) -> dict:
    """Transforme les `--set option=valeur` en surcharges d'AgentConfig."""

    names = {f.name for f in fields(AgentConfig)}
    overrides = {}
    for item in items:
        key, value = item.split("=", 1)
        if key not in names:
            parser.error(f"option inconnue: {key}")
        overrides[key] = parse_value(value)
    return overrides


def make_service(model: str, overrides: dict, prefix: str = "bench_") -> ToolSearchService:
    """Service de recherche autonome, dans un workspace temporaire."""

    config = AgentConfig(
        ollama_url="http://localhost:11434", model=model, ai_id=0,
        workspace_root=Path(tempfile.mkdtemp(prefix=prefix)), prompts_dir=ROOT / "agent/prompts",
    )
    config = replace(config, **overrides)
    config.project_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(config.project_dir / ".agent/logs", console_level="WARNING")
    agent = SimpleNamespace(llm=LocalLLM(config), prompts=PromptLoader(config.prompts_dir), config=config, tools=None)
    return ToolSearchService(agent)


def environment() -> dict:
    return {"os": "Linux x86_64", "python": "%d.%d" % sys.version_info[:2], "packages_and_tools": {}}


def register_tracing(project_name: str) -> None:
    """Envoie les traces LangChain (appels ChatOllama) à Phoenix."""

    from phoenix.otel import register

    register(endpoint=f"{PHOENIX_URL}/v1/traces", project_name=project_name,
             auto_instrument=True, verbose=False)


def get_or_create_dataset(client, name: str, description: str, cases: list[dict]):
    """Dataset Phoenix des cas du bench (entrée : demande, sortie : packages attendus)."""

    try:
        return client.datasets.get_dataset(dataset=name)
    except Exception:
        return client.datasets.create_dataset(
            name=name,
            dataset_description=description,
            inputs=[{"case_id": c["id"], "request": c["request"], "task": c["task"]} for c in cases],
            outputs=[{"expected": c["expected"]} for c in cases],
        )


def scores(experiment) -> dict[str, float]:
    """Moyenne de chaque évaluateur d'une expérience Phoenix."""

    values: dict[str, list[float]] = {}
    for run in experiment["evaluation_runs"]:
        run = run if isinstance(run, dict) else run.__dict__
        result = run.get("result") or {}
        result = result if isinstance(result, dict) else result.__dict__
        if result.get("score") is not None:
            values.setdefault(run["name"], []).append(float(result["score"]))
    return {k: sum(v) / len(v) for k, v in values.items()}
