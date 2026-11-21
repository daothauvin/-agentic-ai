import json
import re
import requests
import unicodedata
from bs4 import BeautifulSoup
from pathlib import Path
from .models import (
    AgentState,
    GeneratedSpeTool,
    PackageResearch,
    PackageResearch,
    GeneratedSpeTool,
    PackageInstall,
    PackageUse,
    IDs,
    Package,
    PackageQueries,
    PackageNames,
    PackageRanking,
)
from .execution import run_command
from urllib.parse import urljoin
from .rag import PackageRAG
from .package_advisor import tool_query
from .logger import get_logger, preview, timed

logger = get_logger("toolsearch")

class ToolSearchService:
    """
    Responsable de la découverte et de l'analyse des outils/packages
    nécessaires à une tâche.
    """

    def __init__(self, agent):
        self.agent = agent

        self.llm = agent.llm
        self.prompts = agent.prompts
        self.config = agent.config
        self.tools = agent.tools
        # Registre des paquets vérifiés (absent dans les benchs de recherche).
        self.registry = getattr(agent, "packages", None)
        #: Commande(s) d'installation qui ont réussi lors du dernier install_package.
        self.last_install_commands: list[str] = []
        #: Paquets installés ajoutés au dernier classement (benchs / logs).
        self.last_installed_candidates: list[str] = []
        self.rag = PackageRAG(
            ollama_url=self.config.ollama_url,
            cache_dir=self.config.project_dir / ".rag",
            multi_level=self.config.rag_multi_level,
        )

    # ==============================================================
    # PUBLIC API
    # ==============================================================

    def search_tools(self, state: AgentState):
        """
        Point d'entrée principal.
        """

        tool_spec = self.identify_required_tool(state)

        package_candidates = self.search_packages(tool_spec)
        for package in package_candidates:

            analysis = self.analyze_package(
                package=package,
                tool_spec=tool_spec,
            )
            

            if not self.is_package_eligible(analysis):
                continue
            if not self.install_package(analysis):
                continue

            self.record_package(
                tool_spec=tool_spec,
                analysis=analysis,
            )

            self.generate_tool(
                tool_spec=tool_spec,
                analysis=analysis,
            )

            # TODO:
            # Tester l'outil généré
            #
            # self.test_tool(...)

            # TODO:
            # Vérifier l'outil
            #
            # self.verify_tool(...)

            # Pour l'instant on s'arrête au premier package valide.
            break

        # TODO:
        # Retourner / modifier le state lorsque la logique sera définie.

    # ==============================================================
    # 1. TOOL IDENTIFICATION
    # ==============================================================

    def identify_required_tool(self, state: "AgentState"):
        """
        Demande au LLM quels outils/packages sont nécessaires
        pour accomplir la tâche.
        """

        task = state.get("current_task", {}) or {}
        if self.config.search_use_request and state.get("request"):
            # 7. La demande initiale garde le contexte concret ("chien" et non "animal").
            payload = {"demande_utilisateur": state["request"], "tache": task}
        else:
            payload = task
        user_prompt = json.dumps(payload, ensure_ascii=False, indent=2)

        prompt = self.prompts.render(
            "tools_identifier",
            environment=json.dumps(
                state["environment"],
                ensure_ascii=False,
                indent=2,
            ),
        )

        result = self.llm.structured(
            GeneratedSpeTool,
            prompt,
            user_prompt,
        )

        tool = result.model_dump()
        if self.config.search_use_request:
            tool["request"] = state.get("request", "")
            tool["task"] = task.get("title", "")

        logger.info("Outil requis: %s", tool)
        return tool

    # ==============================================================
    # 2. PACKAGE SEARCH
    # ==============================================================

    def search_packages(self, tool_spec):
        """
        Retourne les packages PyPI candidats, classés du plus au moins pertinent.

        Étapes (chacune activable dans AgentConfig) :
            1. requêtes courtes multiples (sinon une requête longue) ;
            2. noms proposés par le LLM et vérifiés sur PyPI ;
            5. normalisation des URLs + dédoublonnage ;
            4. filtre sur les métadonnées PyPI ;
            3. classement LLM sur nom + résumé.
        """

        queries = self.build_package_search_queries(tool_spec)
        results, hits = self.run_search_queries(queries)

        if self.config.search_name_guess:
            with timed("Noms de packages proposés par le LLM", logger):
                results = self.guess_package_names(tool_spec) + results

        if self.config.search_normalize:
            results = self.normalize_and_sort(results, hits)

        if self.config.search_metadata_filter:
            with timed("Filtre métadonnées PyPI", logger):
                results = self.filter_by_metadata(results)

        if self.config.search_prerank and (results or self.registry is not None):
            with timed("Classement LLM des candidats", logger):
                results = self.rank_candidates(results, tool_spec)

        logger.info("%d packages candidats: %s", len(results), [p.url for p in results])
        return results

    def run_search_queries(self, queries: list[str]) -> tuple[list[Package], dict[str, int]]:
        """Lance chaque requête sur SearXNG et fusionne les résultats en alternance.

        Retourne aussi, pour chaque package, le nombre de requêtes qui l'ont trouvé.
        """

        per_query: list[list[Package]] = []
        hits: dict[str, int] = {}
        for query in queries:
            with timed(f"Recherche web SearXNG '{query}'", logger):
                try:
                    found = self.parse_search_results(self.search_web(query))
                except requests.RequestException as error:
                    logger.warning("Recherche SearXNG échouée (%s): %s", query, error)
                    found = []
            logger.info("%d résultats pour '%s'", len(found), query)
            # Un hit par requête : un doublon dans la même page ne compte pas.
            for key in {pypi_name_from_url(p.url) or p.url for p in found}:
                hits[key] = hits.get(key, 0) + 1
            per_query.append(found)

        # Fusion en alternance (1er de chaque requête, puis 2e...) : aucune requête n'écrase les autres.
        results = [
            ranked[i] for i in range(max((len(r) for r in per_query), default=0))
            for ranked in per_query if i < len(ranked)
        ]
        return results, hits

    def normalize_and_sort(self, packages: list[Package], hits: dict[str, int]) -> list[Package]:
        """5. Normalise et dédoublonne ; les packages trouvés par plusieurs requêtes passent devant."""

        results = self.normalize_packages(packages)
        results.sort(key=lambda p: -hits.get(pypi_name_from_url(p.url), 0))  # tri stable
        return results

    def build_package_search_queries(self, tool_spec) -> list[str]:
        """Requêtes de recherche : 3 requêtes courtes (1.) ou une requête longue."""

        user_prompt = json.dumps(tool_spec, ensure_ascii=False, indent=2)

        if not self.config.search_multi_query:
            result = self.llm.structured(
                PackageResearch,
                self.prompts.render("package_searcher"),
                user_prompt,
            )
            logger.info("Requête de recherche: %s", result.internet_research)
            return [result.internet_research]

        result = self.llm.structured(
            PackageQueries,
            self.prompts.render("package_queries"),
            user_prompt,
        )
        queries = []
        for query in result.queries:
            # Garde-fou : le petit modèle écrit parfois une phrase entière.
            query = " ".join(query.replace('"', " ").split()[:8])
            if query and query.lower() not in (q.lower() for q in queries):
                queries.append(query)
        queries = queries[:3]
        logger.info("Requêtes de recherche: %s", queries)
        return queries

    def guess_package_names(self, tool_spec) -> list[Package]:
        """2. Le LLM propose des noms ; seuls ceux qui existent sur PyPI sont gardés."""

        result = self.llm.structured(
            PackageNames,
            self.prompts.render("package_names"),
            json.dumps(tool_spec, ensure_ascii=False, indent=2),
        )
        names = [n.strip().split("[")[0].split("=")[0] for n in result.names if n.strip()]
        logger.info("Noms proposés par le LLM: %s", names)

        packages = []
        for name in names[:5]:
            data = self.fetch_pypi_json_by_name(name)
            if data is None:
                logger.info("Nom proposé inexistant sur PyPI: %s", name)
                continue
            info = data.get("info", {})
            packages.append(Package(
                title=info.get("name") or name,
                url=f"https://pypi.org/project/{info.get('name') or name}/",
                content=info.get("summary") or "",
                package_use=None, package_install=None, usage_context=None, tree=None,
            ))
        return packages

    def normalize_packages(self, packages: list[Package]) -> list[Package]:
        """5. Ramène chaque URL à https://pypi.org/project/<nom>/ et retire les doublons."""

        seen, unique = set(), []
        for package in packages:
            name = pypi_name_from_url(package.url)
            if name is None:
                logger.debug("URL ignorée (pas un projet PyPI): %s", package.url)
                continue
            if name in seen:
                continue
            seen.add(name)
            unique.append(package.model_copy(update={"url": f"https://pypi.org/project/{name}/"}))
        return unique

    def filter_by_metadata(self, packages: list[Package]) -> list[Package]:
        """4. Rejette, sans LLM, les packages inactifs, incompatibles ou sans fichier."""

        import sys
        from concurrent.futures import ThreadPoolExecutor
        from datetime import datetime, timezone

        def check(package: Package):
            data = self.fetch_pypi_json_by_name(pypi_name_from_url(package.url) or "")
            if data is None:
                return package, "introuvable sur PyPI", None
            info = data.get("info", {}) or {}
            urls = data.get("urls") or []
            if not urls:
                return package, "aucun fichier pour la dernière version", None
            classifiers = info.get("classifiers") or []
            if any("Development Status :: 7 - Inactive" in c for c in classifiers):
                return package, "marqué Inactive", None
            upload = max((u.get("upload_time_iso_8601") or "" for u in urls), default="")
            if upload:
                age = datetime.now(timezone.utc) - datetime.fromisoformat(upload.replace("Z", "+00:00"))
                if age.days > 3 * 365:
                    return package, f"dernière release il y a {age.days // 365} ans", None
            if not python_version_ok(info.get("requires_python"), sys.version_info[:2]):
                return package, f"requires_python {info.get('requires_python')}", None
            downloads = self.monthly_downloads(info.get("name") or "")
            if downloads is not None and downloads < self.config.search_min_monthly_downloads:
                # Protège aussi contre les noms inventés/squattés (slopsquatting).
                return package, f"{downloads} téléchargements/mois", None
            summary = info.get("summary") or package.content
            return package.model_copy(update={"content": summary}), None, info.get("name")

        with ThreadPoolExecutor(max_workers=8) as pool:
            checked = list(pool.map(check, packages))

        kept = []
        for package, reason, _ in checked:
            if reason:
                logger.info("Package filtré (%s): %s", reason, package.url)
            else:
                kept.append(package)
        return kept

    def installed_candidates(self, tool_spec) -> list[Package]:
        """Paquets déjà installés et vérifiés proches de l'outil recherché (RAG).

        Args:
            tool_spec: Spécification de l'outil.

        Returns:
            Paquets du registre au format ``Package`` (``content`` = description).
        """

        if self.registry is None or not self.config.registry_in_rank:
            return []
        try:
            found = self.registry.search(
                tool_query(tool_spec),
                k=self.config.registry_rank_k,
                min_score=self.config.registry_min_score,
            )
        except Exception as error:
            logger.warning("Registre de paquets indisponible pour le classement: %s", error)
            return []
        logger.info("Paquets installés ajoutés au classement: %s",
                    [(r.name, round(score, 3)) for r, score in found])
        return [
            Package(
                title=record.name,
                url=f"https://pypi.org/project/{record.name}/",
                content=record.description or record.summary,
                package_use=None, package_install=None,
                usage_context=record.usage_context or None, tree=None,
            )
            for record, _ in found
        ]

    def rank_candidates(self, packages: list[Package], tool_spec) -> list[Package]:
        """3. Un seul appel LLM classe les candidats sur leur nom et leur résumé.

        Les paquets déjà installés et vérifiés proches de l'outil (registre)
        sont ajoutés en tête de liste, marqués ``[INSTALLED]``.

        Args:
            packages: Candidats issus de la recherche web.
            tool_spec: Spécification de l'outil.

        Returns:
            Candidats classés, du plus au moins pertinent.
        """

        installed = self.installed_candidates(tool_spec)
        installed_names = {pypi_name_from_url(p.url) for p in installed}
        self.last_installed_candidates = sorted(installed_names)
        web = [p for p in packages if pypi_name_from_url(p.url) not in installed_names]
        packages = installed + web
        if not packages:
            return []

        listing = "\n".join(
            f"{i}. {pypi_name_from_url(p.url) or p.title}"
            f"{' [INSTALLED]' if pypi_name_from_url(p.url) in installed_names else ''}"
            f" — {preview(p.content, 200)}"
            for i, p in enumerate(packages, start=1)
        )
        logger.debug("Candidats soumis au classement LLM:\n%s", listing)
        task = {k: tool_spec.get(k) for k in ("request", "task", "description", "input", "output") if tool_spec.get(k)}
        result = self.llm.structured(
            PackageRanking,
            self.prompts.render("package_ranking"),
            f"TASK:\n{json.dumps(task, ensure_ascii=False, indent=2)}\n\nCANDIDATES:\n{listing}",
        )
        order = []
        for number in result.ranking:
            if 1 <= number <= len(packages) and number - 1 not in order:
                order.append(number - 1)
        logger.info("Classement LLM: %s", [pypi_name_from_url(packages[i].url) for i in order])
        if not order:
            # Le LLM n'a rien retenu : on garde l'ordre d'origine plutôt que de tout perdre.
            logger.warning(
                "Classement LLM vide (réponse brute: %s) : ordre d'origine conservé, "
                "les candidats analysés risquent d'être hors sujet",
                result.ranking,
            )
            return web or packages
        return [packages[i] for i in order]

    def monthly_downloads(self, name: str) -> int | None:
        """Téléchargements du dernier mois (pypistats.org), None si indisponible."""

        try:
            response = requests.get(
                f"https://pypistats.org/api/packages/{name.lower()}/recent",
                headers={"User-Agent": "agent-langchain/1.0"},
                timeout=15,
            )
            if response.status_code != 200:
                return 0 if response.status_code == 404 else None
            return int(response.json()["data"]["last_month"])
        except (requests.RequestException, KeyError, ValueError):
            return None

    def fetch_pypi_json_by_name(self, name: str) -> dict | None:
        """JSON PyPI d'un package (mis en cache), None si le package n'existe pas."""

        if not name:
            return None
        cache = self.__dict__.setdefault("_pypi_cache", {})
        key = name.lower()
        if key not in cache:
            try:
                response = requests.get(
                    f"https://pypi.org/pypi/{name}/json",
                    headers={"User-Agent": "agent-langchain/1.0"},
                    timeout=30,
                )
                cache[key] = response.json() if response.status_code == 200 else None
            except requests.RequestException as error:
                logger.warning("PyPI injoignable pour %s: %s", name, error)
                return None
        return cache[key]

    def build_package_search_query(self, tool_spec):
        """Compatibilité : première requête de recherche."""

        return self.build_package_search_queries(tool_spec)[0]

    def search_web(self, query: str):
        """Effectue la recherche web via SearXNG (limitée à pypi.org)."""

        response = requests.get(
            self.config.searxng_url,
            params={
                "q": query + " site:pypi.org",
                "language": "auto",
                **({"engines": self.config.searxng_engines} if self.config.searxng_engines else {}),
            },
            timeout=300,
        )

        response.raise_for_status()
        logger.debug("SearXNG GET %s -> HTTP %s (%d car.)",
                     response.url, response.status_code, len(response.text))
        self.log_engine_errors(response.text, query)

        return response.text

    @staticmethod
    def log_engine_errors(html: str, query: str) -> dict[str, str]:
        """Logue les moteurs SearXNG en erreur (captcha, 403, parsing...).

        Sans ce log, un moteur cassé se traduit seulement par « 0 résultats ».

        Args:
            html: Page de résultats SearXNG.
            query: Requête envoyée (pour le contexte du log).

        Returns:
            Dictionnaire moteur -> message d'erreur.
        """

        soup = BeautifulSoup(html, "html.parser")
        errors = {}
        for row in soup.select("#engines_msg-table tr"):
            name = row.select_one(".engine-name")
            error = row.select_one(".response-error")
            if name and error:
                errors[name.get_text(strip=True)] = error.get_text(strip=True)
        if errors:
            logger.warning("SearXNG: moteur(s) en erreur pour '%s': %s", query, errors)
        return errors

    def parse_search_results(self, html: str):
        """
        Transforme la réponse HTML du moteur de recherche
        en liste de résultats structurés.
        """

        soup = BeautifulSoup(
            html,
            "html.parser",
        )

        results = []

        for result in soup.select(".result"):

            title = result.select_one("h3")
            title = title.get_text(strip=True) if title else ""
            link = result.select_one("a")
            url  = link.get("href") if link else ""
            content = result.select_one(".content")
            content = content.get_text(" ",strip=True,) if content else ""
            package = Package(title = title,
                    url = url,
                    content = content,
                    package_use = None,
                    package_install = None,
                    usage_context = None,
                    tree = None
            )
            results.append(
                package
            )

        return results

    # ==============================================================
    # 3. PACKAGE ANALYSIS
    # ==============================================================

    def analyze_package(self, package, tool_spec):
        """
        Analyse un repository et détermine s'il peut répondre
        au besoin.
        """

        logger.info("Analyse du package: %s", package.url)

        with timed(f"[{package.title}] Métadonnées PyPI", logger):
            pypi_json = self.fetch_pypi_json(package)

        with timed(f"[{package.title}] Indexation FAISS (bge-m3)", logger):
            index = self.rag.index_package(pypi_json)

        with timed(f"[{package.title}] Retrieval installation", logger):
            installation_context = self.extract_installation_context(index)
        logger.debug("Contexte installation:\n%s", installation_context)

        with timed(f"[{package.title}] Retrieval usage", logger):
            usage_context = self.extract_usage_context(index, tool_spec)
        logger.debug("Contexte usage:\n%s", usage_context)

        with timed(f"[{package.title}] Analyse installation (LLM)", logger):
            installation_info = self.analyze_installation(installation_context)

        with timed(f"[{package.title}] Analyse usage (LLM)", logger):
            usage_info = self.analyze_usage(usage_context, tool_spec)

        logger.info("INSTALLATION INFO: %s", installation_info)
        logger.info("USAGE INFO: %s", usage_info)
        
        return Package(
            title = package.title,
            url = package.url,
            content = package.content,
            tree = None,
            package_use = usage_info,
            package_install = installation_info,
            usage_context = usage_context

        )

    

    # ==============================================================
    # 3.2 PyPI metadata
    # ==============================================================

    def fetch_pypi_json(self, package):
        """Récupère les métadonnées JSON PyPI (README inclus) du package."""

        name = pypi_name_from_url(package.url)
        if name and self.fetch_pypi_json_by_name(name) is not None:
            return self.fetch_pypi_json_by_name(name)
        response = requests.get(
            urljoin(package.url.replace("project", "pypi"), "json"),
            headers={"User-Agent": "agent-langchain/1.0"},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    # ==============================================================
    # 3.3 Retrieval sémantique (bge-m3 + FAISS)
    # ==============================================================

    def extract_installation_context(self, index):
        """Passages de la doc concernant l'installation."""

        chunks = self.rag.retrieve(
            index,
            # Une seule requête, top-k court (AgentConfig.rag_install_k) : un
            # contexte trop long fait perdre la commande au petit LLM.
            ["How to install the package? pip install command"],
            k=self.config.rag_install_k,
            purpose="installation",
        )
        return self.rag.to_context(chunks)

    def extract_usage_context(self, index, tool_spec):
        """Passages expliquant comment utiliser le package pour la tâche."""

        chunks = self.rag.retrieve(
            index,
            [
                f"How to use the package to: {tool_spec.get('description', '')}",
                f"Usage example: input {tool_spec.get('input', '')}, "
                f"output {tool_spec.get('output', '')}",
                "Quickstart usage example code / command line",
            ],
            k=self.config.rag_usage_k,
            purpose="usage",
        )
        return self.rag.to_context(chunks)

    # ==============================================================
    # 4. PACKAGE INSTALLATION ANALYSIS
    # ==============================================================

    def analyze_installation(
        self,
        context,
    ):
        """
        Analyse les informations d'installation du package.
        """

        return self.llm.structured(
            PackageInstall,
            self.prompts.render(
                            "package_install"
                        ),
            f"Package INFORMATION:{context}"
        )

    # ==============================================================
    # 5. PACKAGE USAGE ANALYSIS
    # ==============================================================

    def analyze_usage(
        self,
        context,
        tool_spec,
    ):
        """
        Analyse comment le package peut être utilisé.
        """

        tool_description = json.dumps(
            tool_spec,
            ensure_ascii=False,
            indent=2,
        )

        return self.llm.structured(
            PackageUse,
            self.prompts.render(
                "package_utility"
            ),
            (
                f"Package Description:{context}\n"
                f"Task Description:{tool_description}"
            ),
        )

    # ==============================================================
    # 6. PACKAGE SELECTION
    # ==============================================================

    def is_package_eligible(self, analysis):
        """
        Détermine si un package peut être sélectionné.

        Cette méthode contient la règle métier actuelle.
        """

        if analysis is None:
            return False

        installation = analysis.package_install
        usage = analysis.package_use

        can_install = (
            installation.pip_install
            or installation.apt_get_install
        )

        can_use = (
            usage.shell_use
            or usage.python_use
        )

        no_api_key = (
            not installation.requires_API_key
            and not usage.requires_API_key
        )

        not_paid = (
            not installation.paid
            and not usage.paid
        )

        return (
            can_install
            and can_use
            and no_api_key
            and not_paid
        )

    # ==============================================================
    # 7. INSTALL PACKAGE
    # ==============================================================

    def install_package(self, analysis):
        """
        Installe le package.

        Retourne True si l'installation réussit.
        """

        installation = analysis.package_install
        logger.info("Commandes d'installation: %s", installation.install_commands)
        if not installation.install_commands:
            logger.warning("Aucune commande d'installation fournie pour %s", analysis.url)

        self.last_install_commands = []
        for command in installation.install_commands:

            result = run_command(
                command,
                self.config.project_dir,
                self.config.command_timeout,
                sandbox=self.config.sandbox,
            )

            if result.get("ok"):
                self.last_install_commands = [command]
                return True
            logger.warning("Installation échouée: %s -> %s", command, preview(result["stderr"]))

        return False

    # ==============================================================
    # 8. RECORD PACKAGE
    # ==============================================================

    def record_package(
        self,
        tool_spec,
        analysis,
    ):
        """
        Enregistre le package sélectionné.

        TODO:
        - définir précisément le format attendu par self.tools
        """

        # self.tools.record_packages(
        #     tool_spec.packages,
        #     tool_spec.install_manager,
        # )

        pass

    # ==============================================================
    # 9. GENERATE TOOL
    # ==============================================================

    def generate_tool(
        self,
        tool_spec,
        analysis,
    ):
        """
        Génère l'outil utilisable par l'agent.

        TODO:
        - génération de code Python
        - génération d'une commande shell
        - création du fichier
        - ajout de l'outil au registre
        """

        pass

    # ==============================================================
    # 10. TEST TOOL
    # ==============================================================

    def test_tool(
        self,
        tool,
    ):
        """
        Teste l'outil généré.

        TODO:
        - tests unitaires ?
        - test réel ?
        - sandbox ?
        - validation de la sortie ?
        """

        pass

    # ==============================================================
    # 11. VERIFY TOOL
    # ==============================================================

    def verify_tool(
        self,
        tool,
        tool_spec,
    ):
        """
        Vérifie que l'outil satisfait réellement la tâche.
        """

        pass




_PYPI_PROJECT = re.compile(r"^https?://(?:www\.)?pypi\.org/project/([^/?#]+)", re.I)


def pypi_name_from_url(url: str) -> str | None:
    """Nom normalisé (PEP 503) d'un projet PyPI depuis son URL, None sinon.

    https://pypi.org/project/InstaPy/0.3.4/ -> "instapy"
    """

    match = _PYPI_PROJECT.match(url or "")
    if not match:
        return None
    return re.sub(r"[-_.]+", "-", match.group(1)).lower()


def python_version_ok(requires_python: str | None, version: tuple[int, int]) -> bool:
    """Vérifie grossièrement un specifier requires_python (>=, <, <=, ==, !=, ~=)."""

    if not requires_python:
        return True
    current = version
    for spec in requires_python.split(","):
        spec = spec.strip()
        match = re.match(r"^(>=|<=|==|!=|~=|>|<)\s*(\d+)(?:\.(\d+|\*))?", spec)
        if not match:
            continue
        op, major, minor = match.group(1), int(match.group(2)), match.group(3)
        if minor == "*":
            if op == "==" and current[0] != major:
                return False
            continue
        target = (major, int(minor or 0))
        ok = {
            ">=": current >= target, ">": current > target,
            "<=": current <= target, "<": current < target,
            "==": current == target, "!=": current != target,
            "~=": current >= target and current[0] == target[0],
        }[op]
        if not ok:
            return False
    return True
