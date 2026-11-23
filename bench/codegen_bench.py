"""Bench de l'agent de code : boucle génération → test → vérification → correction.

Pour chaque (modèle de code, version de prompts, paquet), on rejoue les nœuds
réels ``construct_code_for_package`` → ``execute_code_for_package`` →
``verify_code_package`` (jusqu'à ``max_tool_fix_attempts`` corrections). Le
contexte de doc vient du RAG de production sur des docs PyPI figées
(``bench/fixtures/rag/``). Les métriques ne dépendent pas du vérificateur LLM :

* ``first_ok`` : le premier script, lancé avec des arguments FIXÉS (sans le
  testeur LLM), réussit : code 0 et fichier non vide produit (ou stdout non
  vide si la tâche n'écrit pas de fichier). Mesure le générateur seul.
* ``loop_ok`` : en fin de boucle, le vérificateur a validé ET le dernier test
  réussit vraiment.
* ``false_pos`` : le vérificateur a validé un test raté.
* ``attempts`` : nombre de tests joués ; ``gen_s`` : durée moyenne d'un appel
  au modèle de code (génération ou correction).
* ``detect_ok`` : à chaque test, les fichiers réellement écrits dans le
  dossier de test figurent dans les fichiers détectés par l'agent
  (``changed_files``, d'après la ctime).
* ``ctx_used`` : la première commande de test reprend la valeur fournie par
  la demande utilisateur ou par l'étape précédente (``CONTEXTS``), ``None``
  pour le cas témoin sans valeur.

Chaque cas a une demande utilisateur et, parfois, une étape précédente simulée
(``CONTEXTS``), donnés au testeur comme dans l'agent (``AgentGraph._test_context``).
``--no-context`` les retire, pour comparer avec l'ancien comportement.

Seul le modèle de code change (``--models`` : prompts ``tool_generator``,
``tool_tester`` et ``tool_fixer``) ; le vérificateur reste sur ``--model``.
Une version de prompts (``bench/codegen_prompts/<version>/``) peut ne
contenir que certains fichiers : les autres viennent de ``agent/prompts``.

Exemple::

    python bench/codegen_bench.py --workspace ~/.cache/ev_codegen_bench \\
        --models qwen2.5:3b --prompts P0_original P7_noexample
"""

import argparse
import json
import shlex
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.config import AgentConfig  # noqa: E402
from agent.execution import run_command  # noqa: E402
from agent.graph import AgentGraph  # noqa: E402
from agent.logger import get_logger, setup_logging  # noqa: E402
from agent.models import Package  # noqa: E402
from agent.toolsearch import ToolSearchService  # noqa: E402

FIXTURES = ROOT / "bench" / "fixtures" / "rag"
PROMPT_VERSIONS = ROOT / "bench" / "codegen_prompts"
CODEGEN_PROMPTS = ("tool_generator", "tool_tester", "tool_verifier", "tool_fixer")
IMG = "https://httpbin.org/image/jpeg"

# (paquet, spec de l'outil, arguments du test à arguments fixés, fichier attendu)
# La spec py3-wget est celle du checkpoint du run « telecharge une image de chat ».
CASES = [
    ("py3-wget", {"name": "wget", "description": "Télécharge un fichier depuis une URL",
                  "input": "URL d'une image de chat", "output": "Fichier image de chat téléchargé"},
     [IMG], True),
    ("requests", {"name": "download_file", "description": "Télécharge un fichier depuis une URL",
                  "input": "URL d'une image de chat", "output": "Fichier image de chat téléchargé"},
     [IMG], True),
    ("httpx", {"name": "fetch_image", "description": "Télécharge une image depuis une URL",
               "input": "URL de l'image", "output": "Fichier image téléchargé"}, [IMG], True),
    ("qrcode", {"name": "qr_code", "description": "Génère une image QR code",
                "input": "Texte à encoder", "output": "Image PNG du QR code"},
     ["Bonjour le monde"], True),
    ("segno", {"name": "qr_segno", "description": "Crée un QR code dans un fichier image",
               "input": "Texte du QR code", "output": "Fichier image du QR code"}, ["hello"], True),
    ("markdown", {"name": "md_to_html", "description": "Convertit du texte Markdown en HTML",
                  "input": "Texte Markdown", "output": "Code HTML"}, ["# Titre"], False),
    ("mistune", {"name": "md_render", "description": "Convertit du Markdown en HTML",
                 "input": "Texte Markdown", "output": "HTML affiché"}, ["**gras**"], False),
    ("markdown2", {"name": "md2_html", "description": "Transforme du Markdown en HTML",
                   "input": "Texte Markdown", "output": "HTML"}, ["*italique*"], False),
]
# Contexte du testeur par paquet : demande utilisateur, étapes précédentes
# simulées (titre, sortie attendue, stdout) et valeur que le test doit reprendre.
CONTEXTS = {
    "py3-wget": ("telecharge une image de chat",
                 [("Recherche d'une URL d'image de chat", "URL d'une image de chat",
                   "https://cataas.com/cat")], "https://cataas.com/cat"),
    "requests": ("Télécharge l'image https://httpbin.org/image/png dans le dossier courant",
                 [], "https://httpbin.org/image/png"),
    "httpx": ("récupère l'image trouvée à l'étape précédente",
              [("Recherche d'une image", "URL de l'image",
                '{"results": [{"title": "logo", "url": "https://httpbin.org/image/webp"}]}')],
              "https://httpbin.org/image/webp"),
    "qrcode": ("fais un QR code qui contient https://example.com", [], "https://example.com"),
    "segno": ("crée un QR code pour le texte Wifi maison 1234", [], "Wifi maison 1234"),
    "markdown": ("convertis le rapport en HTML",
                 [("Rédaction du rapport en Markdown", "Texte Markdown du rapport",
                   "# Rapport mensuel")], "# Rapport mensuel"),
    "mistune": ("convertis du markdown en html", [], None),
    "markdown2": ("transforme '**Bonjour** le monde' en HTML", [], "**Bonjour** le monde"),
}
PIP_PACKAGES = ["py3-wget", "requests", "httpx", "qrcode[pil]", "segno", "markdown",
                "mistune", "markdown2"]

logger = get_logger("codegen_bench")


def build_prompts_dir(version: str, target: Path) -> Path:
    """Assemble un répertoire de prompts complet pour une version.

    Args:
        version: Nom du dossier dans ``bench/codegen_prompts``.
        target: Répertoire de travail où écrire le résultat.

    Returns:
        Répertoire contenant ``agent/prompts`` surchargé par la version.
    """

    out = target / "prompts" / version
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    for src in (ROOT / "agent" / "prompts").glob("*.txt"):
        shutil.copy(src, out / src.name)
    for name in CODEGEN_PROMPTS:
        src = PROMPT_VERSIONS / version / f"{name}.txt"
        if src.exists():
            shutil.copy(src, out / src.name)
        else:
            logger.info("%s : %s.txt absent, pris dans agent/prompts", version, name)
    return out


def real_ok(res: dict, test_dir: Path, wants_file: bool) -> bool:
    """Vérité terrain d'un test, sans LLM.

    Args:
        res: Résultat de ``run_command``.
        test_dir: Dossier où le test a tourné.
        wants_file: La tâche doit produire un fichier.

    Returns:
        True si code 0 et (fichier non vide produit si attendu, sinon stdout non vide).
    """

    if res.get("returncode") != 0:
        return False
    if wants_file:
        return any(p.is_file() and p.stat().st_size > 0 for p in test_dir.rglob("*"))
    return bool(res.get("stdout", "").strip())


def last_line(text: str, limit: int = 100) -> str:
    """Dernière ligne non vide d'un texte (message d'erreur), tronquée.

    Args:
        text: Texte (stderr).
        limit: Longueur maximale.

    Returns:
        Dernière ligne, ou chaîne vide.
    """

    lines = [line for line in (text or "").strip().splitlines() if line.strip()]
    return lines[-1][:limit] if lines else ""


def context_state(name: str, spec: dict, with_context: bool) -> tuple[dict, str | None]:
    """État « demande + étapes précédentes » d'un cas, comme dans l'agent.

    Args:
        name: Paquet du cas.
        spec: Spécification de l'outil.
        with_context: False pour ne donner ni demande ni étape précédente.

    Returns:
        ``(clés d'état, valeur attendue dans la commande de test)``.
    """

    if not with_context:
        return {"request": "", "current_task": {"title": spec["description"],
                                                "input": spec["input"],
                                                "output": spec["output"]}}, None
    request, deps, expect = CONTEXTS[name]
    completed = {
        f"step{i}": {"task": {"title": title, "output": output},
                     "execution": [{"stdout": stdout}]}
        for i, (title, output, stdout) in enumerate(deps)
    }
    task = {"title": spec["description"], "input": spec["input"], "output": spec["output"],
            "depends_on": list(completed)}
    return {"request": request, "current_task": task, "completed": completed}, expect


def run_case(agent: AgentGraph, case: tuple, doc: str, with_context: bool = True) -> dict:
    """Joue un cas : premier script à arguments fixés, puis boucle complète.

    Args:
        agent: Agent configuré (modèle de code, prompts).
        case: ``(paquet, spec, arguments fixés, fichier attendu)``.
        doc: Contexte d'usage (RAG) du paquet.
        with_context: Donne au testeur la demande et les étapes précédentes.

    Returns:
        Ligne de résultat (métriques du module).
    """

    name, spec, args, wants_file = case
    pkg = Package(title=name, url=f"https://pypi.org/project/{name}/", content="",
                  package_use=None, package_install=None, usage_context=doc, tree=None)
    ctx, expect = context_state(name, spec, with_context)
    state = {"current_packages": [pkg], "current_tool": dict(spec), "execution": [],
             "tool_mode": "python", **ctx}
    test_dir = agent.config.project_dir / ".agent" / "tool_test"

    gen_times = []
    start = time.perf_counter()
    state.update(agent.construct_code_for_package(state))
    gen_times.append(time.perf_counter() - start)
    path = agent._tool_path(spec)
    first_code = path.read_text(encoding="utf-8")

    shutil.rmtree(test_dir, ignore_errors=True)
    test_dir.mkdir(parents=True)
    fixed = run_command(
        f"python3 {shlex.quote(str(path))} " + " ".join(shlex.quote(a) for a in args),
        test_dir, agent.config.command_timeout, sandbox=agent.config.sandbox,
    )
    first_ok = real_ok(fixed, test_dir, wants_file)

    attempts, verdict, loop_real, first_command = 0, False, False, None
    detect_ok = True
    while True:
        # Vérité terrain : le BENCH vide le dossier avant chaque test (l'agent,
        # lui, ne le vide plus et détecte les fichiers par leur ctime).
        shutil.rmtree(test_dir, ignore_errors=True)
        test_dir.mkdir(parents=True)
        state.update(agent.execute_code_for_package(state))
        attempts += 1
        first_command = first_command or state.get("tool_command")
        loop_real = real_ok(state["execution"][-1], test_dir, wants_file)
        truth = {str(p.relative_to(test_dir)) for p in test_dir.rglob("*") if p.is_file()}
        seen = {f.rsplit(" (", 1)[0] for f in state["execution"][-1].get("produced_files", [])}
        detect_ok &= truth <= seen
        start = time.perf_counter()
        update = agent.verify_code_package(state)
        if update.get("tool_retry"):
            gen_times.append(time.perf_counter() - start)
        state.update(update)
        if update.get("tool_success"):
            verdict = True
            break
        if not update.get("tool_retry"):
            break

    return {
        "package": name,
        "first_ok": first_ok,
        "first_err": "" if first_ok else last_line(fixed["stderr"]) or f"code {fixed['returncode']}",
        "loop_ok": verdict and loop_real,
        "false_pos": verdict and not loop_real,
        "attempts": attempts,
        "detect_ok": detect_ok,
        "ctx_used": None if expect is None else expect in (first_command or ""),
        "first_command": first_command,
        "gen_s": round(sum(gen_times) / len(gen_times), 1),
        "last_command": state.get("tool_command"),
        "first_code": first_code,
    }


def main() -> int:
    """Point d'entrée : lance la matrice modèles × versions × paquets.

    Returns:
        Code de sortie (0).
    """

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--models", nargs="+", default=["qwen2.5:3b"],
                        help="Modèles de code à comparer.")
    parser.add_argument("--model", default="qwen2.5:3b",
                        help="Modèle du vérificateur et du reste de l'agent (fixe).")
    parser.add_argument("--prompts", nargs="+",
                        default=sorted(p.name for p in PROMPT_VERSIONS.iterdir() if p.is_dir()))
    parser.add_argument("--cases", nargs="*", default=None, help="Sous-ensemble de paquets.")
    parser.add_argument("--no-context", action="store_true",
                        help="Testeur sans demande utilisateur ni étape précédente.")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "bench" / "results" / f"codegen_{datetime.now():%Y%m%d_%H%M%S}.jsonl")
    args = parser.parse_args()

    workspace = args.workspace.expanduser().resolve()
    base = dict(ollama_url=args.ollama_url, model=args.model, ai_id=1,
                workspace_root=workspace, command_timeout=120, docker=True)
    log_file = setup_logging(workspace / "projet_1" / ".agent" / "logs", "WARNING")
    print(f"Log DEBUG : {log_file}", flush=True)

    agent0 = AgentGraph(AgentConfig(**base))
    sandbox = agent0.config.sandbox
    sandbox.ensure()
    run_command("pip install -q " + " ".join(shlex.quote(p) for p in PIP_PACKAGES),
                agent0.config.project_dir, 600, sandbox=sandbox)

    cases = [c for c in CASES if not args.cases or c[0] in args.cases]
    search = ToolSearchService(agent0)
    docs = {}
    for name, spec, _, _ in cases:
        index = search.rag.index_package(json.loads((FIXTURES / f"{name}.json").read_text()))
        docs[name] = search.extract_usage_context(index, spec)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for code_model in args.models:
        for version in args.prompts:
            prompts_dir = build_prompts_dir(version, workspace)
            agent = AgentGraph(AgentConfig(**base, prompts_dir=prompts_dir, code_model=code_model))
            for case in cases:
                logger.warning("[%s | %s] %s", code_model, version, case[0])
                row = {"code_model": code_model, "prompts": version,
                       "context": not args.no_context,
                       **run_case(agent, case, docs[case[0]], not args.no_context)}
                with args.out.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(json.dumps({k: v for k, v in row.items() if k != "first_code"},
                                 ensure_ascii=False), flush=True)
    print(f"Résultats : {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
