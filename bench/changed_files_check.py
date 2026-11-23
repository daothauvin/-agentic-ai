"""Vérifie la détection des fichiers produits par un test (``changed_files``).

Un faux « outil » fait des écritures difficiles à détecter : arborescence
profonde, fichier existant écrasé, écriture hors du dossier courant, mtime
antidatée (``os.utime``, ``touch -r``), écriture dans un cache ignoré. On
compare la liste détectée à la liste attendue, avec et sans Docker, puis on
relance le test sans vider le dossier pour vérifier qu'un fichier d'un test
précédent n'est pas compté.

Usage::

    python bench/changed_files_check.py --workspace ~/.cache/ev_changed_files
"""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.config import AgentConfig  # noqa: E402
from agent.execution import changed_files, run_command, start_change_marker  # noqa: E402
from agent.logger import setup_logging  # noqa: E402

TOOL = r'''
import os, pathlib, subprocess, sys
p = pathlib.Path
p("sortie/2026/11/23").mkdir(parents=True, exist_ok=True)
p("sortie/2026/11/23/chat.jpg").write_text("img")            # arborescence profonde
p("ecrase.txt").write_text("nouveau")                         # fichier existant écrasé
p("vieux.bin").write_text("x"); os.utime("vieux.bin", (0, 0))  # mtime remise à 1970
subprocess.run(["touch", "-r", "existant/a/b/deja.txt", "copie.txt"], check=True)  # mtime copiée
p("../hors_cwd").mkdir(exist_ok=True); p("../hors_cwd/rapport.txt").write_text("r")  # hors cwd
p("__pycache__").mkdir(exist_ok=True); p("__pycache__/x.pyc").write_text("c")       # cache ignoré
if sys.argv[1:] == ["container"]:
    p("/tmp/ailleurs").mkdir(exist_ok=True); p("/tmp/ailleurs/hors.txt").write_text("h")
'''
TOOL2 = 'import pathlib; pathlib.Path("ecrase.txt").write_text("encore")\n'


def scenario(docker: bool, workspace: Path) -> bool:
    """Joue le scénario dans un mode et affiche détecté / attendu.

    Args:
        docker: Exécuter dans le conteneur du projet.
        workspace: Racine des workspaces de test.

    Returns:
        True si les deux tests détectent exactement les fichiers attendus.
    """

    config = AgentConfig(ollama_url="http://localhost:11434", model="qwen2.5:3b",
                         ai_id=1 if docker else 2, workspace_root=workspace, docker=docker)
    project = config.project_dir
    sandbox = config.sandbox
    if sandbox is not None:
        sandbox.ensure()
    cwd = project / ".agent" / "tool_test"
    shutil.rmtree(cwd, ignore_errors=True)
    shutil.rmtree(project / ".agent" / "hors_cwd", ignore_errors=True)
    (cwd / "existant" / "a" / "b").mkdir(parents=True)
    (cwd / "existant" / "a" / "b" / "deja.txt").write_text("old")
    (cwd / "ecrase.txt").write_text("ancien")
    tool = project / "tool.py"
    tool.write_text(TOOL)
    run_command("touch -d 2020-01-01 existant/a/b/deja.txt ecrase.txt; sleep 1", cwd, 30,
                sandbox=sandbox)

    expected = {"sortie/2026/11/23/chat.jpg", "ecrase.txt", "vieux.bin", "copie.txt",
                str(project / ".agent" / "hors_cwd" / "rapport.txt")}
    if docker:
        expected.add("/tmp/ailleurs/hors.txt")
        run_command("rm -rf /tmp/ailleurs", cwd, 30, sandbox=sandbox)

    ok = True
    for label, code, want in (("test 1", TOOL, expected), ("test 2 (sans vider)", TOOL2,
                                                            {"ecrase.txt"})):
        tool.write_text(code)
        marker = start_change_marker(project / ".agent" / "tool_test.marker")
        run_command(f"python3 {tool} {'container' if docker else 'host'}", cwd, 30,
                    sandbox=sandbox)
        found = {f["path"] for f in changed_files(marker, project, cwd, sandbox=sandbox)}
        good = found == want
        ok &= good
        print(f"[{'docker' if docker else 'hôte  '}] {label}: {'OK' if good else 'ÉCHEC'}")
        for path in sorted(found | want):
            mark = "✔" if path in found and path in want else ("manquant" if path in want
                                                                else "en trop")
            print(f"    {mark:9} {path}")
    return ok


def main() -> int:
    """Point d'entrée.

    Returns:
        0 si tous les scénarios passent, 1 sinon.
    """

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workspace", required=True, type=Path)
    args = parser.parse_args()
    workspace = args.workspace.expanduser().resolve()
    log = setup_logging(workspace / "logs", "WARNING")
    print(f"Log DEBUG : {log}")
    results = [scenario(True, workspace), scenario(False, workspace)]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
