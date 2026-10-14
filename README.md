# EV Agent Linux

Agent Linux autonome expérimental basé sur **LangChain**, **LangGraph** et **Ollama**.

> ⚠️ **Projet expérimental.** L'agent peut générer du code et exécuter des commandes Linux. Il ne faut pas l'utiliser sur une machine de production ou contenant des données importantes sans sandbox/VM et contrôles supplémentaires.

## 🎯 Objectif du projet

L'objectif architectural est de construire une IA capable de **déterminer elle-même les outils dont elle a besoin, générer ces outils, installer les packages nécessaires, les utiliser, vérifier le résultat et apprendre de ses échecs via une boucle de résolution**.

Cette capacité d'auto-outillage est **la direction cible du projet**. 

## Choix Techniques 

- **Les commandes Linux** sont utilisées pour utiliser et installer des packages. Linux à l'avantage d'avoir un grand nombre de packages libre d'accès. 
- **LangChain** fournit l'intégration avec le LLM Ollama, les messages et les sorties structurées. Il s'agit de la reférence actuelle pour les LLMs.
- **LangGraph** décrit explicitement le workflow sous forme de machine à états : planification, exécution, vérification et résolution d'erreurs. Cette séparation rend l'agent plus facile à faire évoluer, en proposant de nombreux outils pour du streaming, du checkpointing, de la reprise après interruption et du human-in-the-loop. 

## Architecture

![Architecture de l'agent](graph.png)

Le graphe est aussi enregistré directement dans `main.py` dans la constante `MERMAID_ARCHITECTURE`. À chaque lancement, une copie est écrite dans :

```text
AI_Workspace/projet_<id>/.agent/agent_architecture.mmd
```

```mermaid
graph TD
    START([START]) --> INSPECT[inspect_environment]
    INSPECT --> PLAN[plan]
    PLAN --> SELECT[select_task]
    SELECT -->|tâche prête| PREPARE[prepare_task]
    SELECT -->|pas de tâche| FINISH[finish]
    PREPARE -->|besoin d'outil| SEARCH[search_tools]
    PREPARE -->|outil existe| EXECUTE[execute_commands]
    SEARCH --> ANALYSIS[analysis_package]
    ANALYSIS --> ELIGIBLE[record_eligible_package]
    ELIGIBLE --> |True| INSTALL[install_package]
    ELIGIBLE --> |False| ANALYSIS[analysis_package]

    INSTALL --> |nécessite code|CONSTRUCT[construct_code_for_package]
    INSTALL --> |échoue|ANALYSIS[analysis_package]
    INSTALL --> |pas besoin code|REGISTER[register_tools]
    CONSTRUCT --> EXECUTE_CODE[execute_code_for_package]
    EXECUTE_CODE --> VERIFY_CODE[verify_code_package]
    VERIFY_CODE --> |réussi| REGISTER[register_tools]
    VERIFY_CODE --> |rate| --> EXECUTE_CODE[execute_code_for_package]
    REGISTER --> PREPARE[search_tools]
    EXECUTE --> EVALUATE[evaluate]
    EVALUATE -->|success| RECORD[record_success]
    RECORD --> SELECT
    EVALUATE -->|failure| RESOLVE[resolve_failure]
    RESOLVE -->|resolvable| SELECT
    RESOLVE -->|not resolvable| FINISH
    FINISH --> END([END])
```

### Flux détaillé

1. **`inspect_environment`**
   - prépare le contexte envoyé au modèle.
   - détecte le répertoire de travail, l'OS, Python et le catalogue local des outils ;

2. **`plan`**
   - transforme la demande en tâches atomiques avec un LLM ;
   - représente les dépendances entre tâches ;

3. **`select_task`**
   - choisit une tâche dont toutes les dépendances sont terminées ;
   - détecte les dépendances impossibles ou circulaires.

4. **`prepare_task`**
   - demande au LLM comment réaliser la tâche ;
   - produit les commandes nécessaires ;
   - peut demander la génération d'un nouvel outil Python ;
   - peut déclarer les packages nécessaires.

5. **`install_and_register_tools`**
   - écrit les outils générés dans `.agent/tools/` ;
   - met à jour `.agent/tools.json` ;
   - peut installer les dépendances avec `pip`, `apt`, `npm` ou `cargo`.

6. **`execute_commands`**
   - exécute les commandes dans le workspace du projet ;
   - conserve stdout, stderr, code retour et informations de packages.

7. **`evaluate`**
   - vérifie que la tâche a réellement produit le résultat attendu ;
   - distingue succès et échec.

8. **`resolve_failure`**
   - analyse l'échec ;
   - génère un plan de résolution sous forme de nouvelles tâches ;
   - exécute ensuite ces tâches avant de retenter la tâche originale ;
   - limite la profondeur de résolution pour éviter les boucles infinies.

## Installation

### 1. Créer un environnement virtuel

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Installer et lancer Ollama

Installer Ollama selon votre OS, puis vérifier qu'il répond :

```bash
ollama serve
```

Dans un autre terminal :

```bash
ollama pull qwen2.5:3b
```

Vous pouvez choisir un autre modèle compatible avec votre installation Ollama :

```bash
export OLLAMA_MODEL=qwen2.5:3b
```

## Utilisation

```bash
python3 main.py --id 1 "Crée un petit projet Python qui lit un fichier CSV et produit un rapport."
```

Ou en mode interactif :

```bash
python3 main.py --id 1
```

Paramètres utiles :

```text
--id          Identifiant de l'IA / workspace
--model       Modèle Ollama
--workspace   Racine des workspaces
--ollama-url  URL de l'API Ollama
--timeout     Timeout d'une commande shell
```

Exemple :

```bash
python3 main.py \
  --id 2 \
  --model qwen2.5:3b \
  --workspace ./AI_Workspace \
  --timeout 300 \
  "Analyse ce dossier et crée un script qui automatise le traitement demandé."
```
### Ce qui fonctionne déjà dans l'architecture

- installation expérimentale de dépendances `pip`, `apt`, `npm` et `cargo` ;
- boucle d'évaluation/résolution ;
- tâches dépendantes ;
- limitation de profondeur de résolution.

## Roadmap

### Phase 1 — architecture IA

- [x] LangChain
- [x] LangGraph
- [x] Ollama
- [x] planification
- [x] exécution Linux
- [x] résolution récursive transformée en boucle LangGraph
- [x] installation expérimentale de packages
- [x] registre d'outils
- [x] architecture Mermaid enregistrée dans le code

### Phase 2 — outils et packages

- [ ] découverte de packages et documentation
- [ ] stocker la documentation des packages
- [ ] générer des outils codés en python
- [ ] tests automatiques des outils générés
- [ ] remplacement automatique d'un outil défectueux

### Phase 3 - sécurité

- [ ] ajouter des vérifications utilisateurs pour l'IA
- [ ] vérifier la conformité des demandes utilisateurs au bon fonctionnement de l'ordinateur et de l'IA
- [ ] empécher des attaques sur l'IA
- [ ] isoler dans un environnement les packages installés par une IA

### Phase 4 - évaluation
- [ ] créer un bench de questions que le modèle doit passer
- [ ] évaluer la réussite de ces fonctions par une IA

### Phase 5 — gateways

- [ ] proposer des modèles adaptés au différentes tâches
- [ ] prévoir des modèles de secours

### Phase 6 — Interface
- [ ] Séparer serveur et utilisateur
- [ ] Permettre des conversations longues durées

### Phase 7 — README

- [ ] Ajouter une architecture du code dans le README
- [ ] Ajouter des tests automatiques de vérification de bon fonctionnement

## Références

- Documentation LangGraph : https://langchain-ai.github.io/langgraph/
- Documentation LangChain : https://python.langchain.com/
- Intégration Ollama LangChain : https://python.langchain.com/docs/integrations/chat/ollama/
