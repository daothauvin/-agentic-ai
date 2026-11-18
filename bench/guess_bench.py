"""Bench de la modif 2 seule : noms proposés par le LLM, vérifiés sur PyPI (sans SearXNG)."""

import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.config import AgentConfig  # noqa: E402
from agent.llm import LocalLLM  # noqa: E402
from agent.prompts.loader import PromptLoader  # noqa: E402
from agent.toolsearch import ToolSearchService, pypi_name_from_url  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
use_request = "--no-request" not in sys.argv
config = AgentConfig(ollama_url="http://localhost:11434", model="qwen2.5:3b", ai_id=0,
                     workspace_root=Path(tempfile.mkdtemp()), prompts_dir=ROOT / "agent/prompts",
                     search_use_request=use_request)
service = ToolSearchService(SimpleNamespace(llm=LocalLLM(config), prompts=PromptLoader(config.prompts_dir), config=config, tools=None))
env = {"os": "Linux x86_64", "python": "3.14", "packages_and_tools": {}}
ok = 0
cases = json.loads((ROOT / "bench/search_cases.json").read_text())
for case in cases:
    t = time.perf_counter()
    spec = service.identify_required_tool({"request": case["request"], "current_task": case["task"], "environment": env})
    names = [pypi_name_from_url(p.url) for p in service.guess_package_names(spec)]
    hit = bool(set(names[:3]) & set(case["expected"]))
    ok += hit
    print(f"{case['id']:<14} {time.perf_counter()-t:5.1f}s {'OK ' if hit else 'NON'} {names}")
print(f"== guess top3 {ok}/{len(cases)}")
