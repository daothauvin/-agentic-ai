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
    Package
)
from .execution import run_command
import os
from  .create_pdf import download_pdf
from urllib.parse import urljoin
from pageindex import PageIndexClient

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

        user_prompt = json.dumps(
            state.get("current_task", {}),
            ensure_ascii=False,
            indent=2,
        )

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

        print("TOOL:", tool)
        return tool

    # ==============================================================
    # 2. PACKAGE SEARCH
    # ==============================================================

    def search_packages(self, tool_spec):
        """
        Recherche des repositories correspondant au besoin.

        Un résultat peut être :
        - directement un repository GitHub
        - un topic GitHub
        - éventuellement une page de recherche GitHub

        Dans tous les cas, cette méthode retourne uniquement
        des repositories exploitables.
        """

        search_query = self.build_package_search_query(tool_spec)
        html = self.search_web(search_query)
        return self.parse_search_results(html)
    

    def is_repository_path(self, path: str) -> bool:
        """
        Vérifie qu'un path ressemble à :

            /owner/repository

        et non :

            /topics/...
            /login
            /settings/...
            etc.
        """

        match = re.match(
            r"^/([^/]+)/([^/]+)$",
            path,
        )

        if not match:
            return False

        owner, repository = match.groups()

        excluded = {
            "topics",
            "search",
            "orgs",
            "organizations",
            "marketplace",
            "features",
            "login",
            "signup",
            "about",
            "pricing",
        }

        return (
            owner not in excluded
            and not repository.startswith("?")
        )

    def deduplicate_repositories(
        self,
        repositories: list[str],
    ) -> list[str]:
        """
        Supprime les repositories en double
        tout en conservant l'ordre.
        """

        seen = set()
        unique = []

        for repository in repositories:
            repository = self.normalize_repository_url(
                repository
            )

            key = repository.lower()

            if key in seen:
                continue
            seen.add(key)
            unique.append(repository)
        return unique
    
    def build_package_search_query(self, tool_spec):
        """
        Utilise le LLM pour transformer la description de l'outil
        en requête de recherche de packages.
        """

        prompt = self.prompts.render(
            "package_searcher"
        )

        user_prompt = json.dumps(
            tool_spec,
            ensure_ascii=False,
            indent=2,
        )

        result = self.llm.structured(
            PackageResearch,
            prompt,
            user_prompt,
        )

        print(
            "INTERNET PROMPT:",
            result.internet_research,
        )

        return result.internet_research

    def search_web(self, query: str):
        """
        Effectue la recherche web.

        TODO:
        - éventuellement remplacer localhost:8080 par une abstraction
        """
        response = requests.get(
            "http://localhost:8080/search",
            params={
                "q": query + " site:pypi.org",
                "language": "auto",
            },
            timeout=300,
        )

        response.raise_for_status()

        return response.text

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

        print("--> TEST REPO:", package.url)

        pdf_path = self.create_pdf(
            package=package,
        )

        tree = self.index_pdf(
            pdf_path
        )

        installation_context = self.extract_installation_context(
            tree
        )

        usage_context = self.extract_usage_context(
            tree,
            tool_spec,
        )
        print(usage_context)
        installation_info = self.analyze_installation(
            installation_context
        )

        usage_info = self.analyze_usage(
            usage_context,
            tool_spec,
        )
        print("INSTALLATION INFO:", installation_info)
        print("USAGE INFO:", usage_info)
        
        return Package(
            title = package.title,
            url = package.url,
            content = package.content,
            tree = tree,
            package_use = usage_info,
            package_install = installation_info,
            usage_context = usage_context

        )

    

    # ==============================================================
    # 3.2 README -> PDF
    # ==============================================================

    def create_pdf(self,package):
        """
        Transforme le README Markdown en PDF.

        TODO:
        - gérer les images
        - gérer les liens
        - choisir un nom de fichier unique
        - éviter les collisions entre repositories
        """
        url = package.url
        response = requests.get(
            urljoin(url.replace("project", "pypi"),"json"),
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; PyPI-to-PDF/1.0)"
            },
            timeout=30,
        )
        response.raise_for_status()

        pdf_path = (
            self.config.project_dir
            / "README.pdf"
        )

        download_pdf(response.json(), pdf_path)
        return pdf_path


    # ==============================================================
    # 3.3 PageIndex
    # ==============================================================

    def create_pageindex_client(self):
        """
        Crée le client PageIndex.

        Cette méthode permet d'isoler la configuration
        de PageIndex du reste de la logique.
        """

        backend = {
            "api_key": "ollama",
            "base_url": "http://localhost:11434/v1",
        }

        return PageIndexClient(
            index={
                "model": "openai/qwen2.5:3b",
                "storage_path": "./.pageindex",
                "backend": backend,
            }
        )

    def index_pdf(self, pdf_path):
        """
        Soumet le PDF à PageIndex et récupère son arbre.
        """

        client = self.create_pageindex_client()

        print("WAITING FOR PDF SUBMISSION")

        result = client.submit_document(
            pdf_path,
            wait=True,
        )

        tree_result = client.get_tree(
            result["doc_id"]
        ).get("result", [])

        print(
            f"FINISHED WITH {count(tree_result)} NODES"
        )

        return tree_result

    # ==============================================================
    # 3.4 Relevant sections
    # ==============================================================

    def extract_installation_context(self, tree):
        """
        Cherche dans l'arbre les sections concernant l'installation.
        """

        result = self.find_relevant_nodes(
            tree=tree,
            questions=[
                "How to install the package?",
            ],
        )
        

        return self.nodes_to_context(
            tree,
            [x for sl in result for x in sl.node_list],
        )

    def extract_usage_context(
        self,
        tree,
        tool_spec,
    ):
        """
        Cherche les sections expliquant comment utiliser
        le package pour la tâche demandée.
        """

        tool_description = json.dumps(
            tool_spec,
            ensure_ascii=False,
            indent=2,
        )

        result = self.find_relevant_nodes(
            tree=tree,
            questions=[
                (
                    "How to use the package for "
                    f"the following task? "
                    f"{tool_description}"
                )
            
            ],
        )
        

        return self.nodes_to_context(
                    tree,
                    [x for sl in result for x in sl.node_list],
                )

    def find_relevant_nodes(
        self,
        tree,
        questions,
    ):
        """
        Utilise le LLM pour identifier les nodes pertinentes
        dans l'arbre PageIndex.
        """

        prompt = f"""
You are given a query and a document's tree structure
(like a Table of Contents).

Your task: identify which node IDs (3 maximum) most likely contain
the answer to the query. 

Document Tree:

{json.dumps(
    compress(tree),
    indent=2,
)}
"""

        return self.llm.structured_batch(
            IDs,
            prompt,
            questions,
        )

    def nodes_to_context(
        self,
        tree,
        node_ids,
    ):
        """
        Transforme des IDs de nodes en contexte textuel.
        """

        nodes = find_nodes_by_ids(
            tree,
            node_ids,
        )

        context_parts = []

        for node in nodes:

            context_parts.append(
                f"[Section: '{node['title']}']\n"
                f"{node.get('text', 'Content not available')}"
            )

        return "\n\n---\n\n".join(
            context_parts
        )

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
        print(installation)
        input()
        for command in installation.install_commands:

            result = run_command(
                command,
                self.config.project_dir,
                self.config.command_timeout,
            )

            if result.get("ok"):
                return True
            print(result)

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


def compress(nodes):
    out = []
    for n in nodes:
                entry = {"node_id": 
                         n["node_id"], 
                         "title": n["title"], 
                         "page": n.get("page_index", "?"),
                         "summary": n.get("text","")[:200]
                         }
                if n.get("nodes"):
                    entry["children"] = compress(n["nodes"])
                out.append(entry)
    return out

def count(nodes):
    res = 0
    for n in nodes:
                res += 1
                if n.get("nodes"):
                    res = res + count(n["nodes"])
                
    return res


def find_nodes_by_ids(tree: list, target_ids:list) -> list:
    """
        Recursively walk the tree and collect nodes matching target_ids.
    """
    found = []
    for node in tree:
        if node["node_id"] in target_ids:
            found.append(node)
        if node.get("nodes"):
            found.extend(find_nodes_by_ids(node["nodes"], target_ids))
    return found