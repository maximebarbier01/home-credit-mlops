# Home Credit MLOps

Projet de scoring crédit réalisé dans le cadre du parcours MLOps
OpenClassrooms.

L'objectif consiste à estimer la probabilité de défaut d'un demandeur de crédit,
à convertir cette probabilité en décision métier et à assurer la traçabilité du
cycle de vie du modèle avec MLflow.

Le projet adopte une approche **script-first** : la préparation des données,
l'entraînement et l'exploitation MLflow reposent sur des scripts Python
versionnés plutôt que sur des notebooks.

## Objectifs métier et ML

- Consolider les tables Home Credit à la granularité d'un client.
- Nettoyer et enrichir les données sans introduire de fuite de cible.
- Comparer plusieurs familles de modèles et stratégies de rééquilibrage.
- Optimiser les hyperparamètres par validation croisée stratifiée.
- Déterminer un seuil de décision minimisant le coût des erreurs métier.
- Expliquer les prédictions globalement et localement avec SHAP.
- Tracer les expériences, versionner le modèle final et tester son serving avec MLflow.

La classe `TARGET = 1` représente un client en défaut. La configuration actuelle
attribue un coût de `10` à un faux négatif et un coût de `1` à un faux positif.
Cette hypothèse est centralisée dans [`configs/default.toml`](configs/default.toml)
et reste modifiable après validation métier.

## Architecture générale

```mermaid
flowchart LR
    A["Tables brutes<br/>data/raw"] --> B["Construction et EDA<br/>build_home_credit_dataset.py"]
    B --> C["Datasets Parquet<br/>data/processed"]
    C --> D["Preprocessing sklearn<br/>imputation et encodage"]
    D --> E["Benchmark<br/>modèles et sampling"]
    E --> F["Validation croisée<br/>et optimisation du seuil"]
    F --> G["Rapports<br/>Excel, courbes et SHAP"]
    F --> H["MLflow<br/>tracking et registry"]
    H --> I["Serving<br/>probabilité et décision métier"]
```

Le découpage suit un principe simple :

- `scripts/` contient les points d'entrée exécutables ;
- `app/` contient l'API FastAPI de production ;
- `src/home_credit_mlops/` contient la logique réutilisable, testable et importable.

## Rôle des briques de déploiement

Le projet contient plusieurs outils, mais chacun a un rôle distinct :

- **FastAPI** : API métier principale. Elle expose `/health`, `/predict` et
  `/monitoring/summary`, charge le modèle depuis Hugging Face Hub, applique la
  décision métier et journalise les appels.
- **Docker** : empaquette l'API FastAPI avec son environnement Python. L'image
  peut être construite localement, testée en CI et publiée dans GitHub
  Container Registry.
- **Hugging Face model repo** : stockage du modèle MLflow exporté. Le dépôt
  `mxmbrbr/home-credit-mlops` n'héberge pas l'API ; il sert de source modèle
  pour l'application.
- **Hugging Face Space** : hébergement possible d'une application web. Un Space
  Docker pourrait héberger l'API FastAPI, mais ce mode nécessite un tier payant
  sur Hugging Face. Un Space Gradio gratuit serait plutôt une démo interactive,
  pas l'API FastAPI actuelle.
- **Streamlit** : dashboard local de monitoring situé dans
  `dashboard/monitoring_app.py`. Il lit les logs SQLAlchemy pour visualiser
  latences, scores, décisions, erreurs et data drift.
- **Gradio** : non utilisé comme application dans ce dépôt. Il peut servir plus
  tard à construire une démo Hugging Face gratuite, mais aucune app Gradio Home
  Credit n'est actuellement implémentée.

Commandes principales :

```bash
# API FastAPI locale
poetry run uvicorn app.main:app --reload --port 8000

# Dashboard Streamlit local
poetry run streamlit run dashboard/monitoring_app.py

# API + PostgreSQL via Docker Compose
docker compose up -d --build

# PostgreSQL seul via Docker, API lancee avec Poetry
docker compose up -d postgres
```

## Arborescence

```text
home-credit-mlops/
|-- .github/workflows/               # Pipeline CI/CD (lint, tests, build, deploy)
|-- app/                             # API FastAPI de production
|   |-- api/                         # Routes HTTP
|   |-- core/                        # Config, sécurité, exceptions
|   |-- db/                          # SQLAlchemy et logs de prédiction
|   |-- schemas/                     # Schémas Pydantic et validateurs
|   `-- services/                    # Chargement modèle et logique métier
|-- configs/                         # Configuration TOML
|-- data/
|   |-- raw/                         # Données Kaggle non versionnées
|   |-- interim/                     # Données intermédiaires
|   `-- processed/                   # Datasets prêts pour le modèle
|-- dashboard/                       # Dashboard Streamlit de monitoring
|-- docs/                            # Documentation détaillée
|-- scripts/                         # Points d'entrée CLI
|-- src/home_credit_mlops/
|   |-- data/                        # Construction du dataset
|   |-- eda/                         # Diagnostics et visualisations
|   |-- fairness/                    # Metriques et rapports de fairness
|   |-- features/                    # Preprocessing sklearn
|   |-- modeling/                    # Modèles, métriques, SHAP et serving
|   |-- monitoring/                  # Logs production, drift et rapports
|   `-- reporting/                   # Consolidation des rapports Excel
|-- tests/                           # Tests automatisés
|-- Dockerfile                       # Image de l'API de scoring
|-- mlflow.db                        # Tracking et registry locaux, non versionnés
|-- mlartifacts/                     # Artefacts MLflow locaux, non versionnés
|-- artifacts/                       # Cache modele HF Hub local, non versionné
`-- reports/                         # Livrables générés, non versionnés
```

Une nomenclature détaillée, fichier par fichier, est disponible dans
[`docs/mode_emploi_pipeline_ml.md`](docs/mode_emploi_pipeline_ml.md).

## Prérequis et installation

- WSL 2 avec Ubuntu pour l'environnement de développement actuel ;
- Python `>=3.11,<3.13` ;
- Poetry ;
- fichiers Home Credit placés dans `data/raw/` ;
- Docker (pour construire/tester l'image de l'API) ;
- Docker Compose (pour la démo PostgreSQL + API multi-conteneurs) ;
- un compte Hugging Face (pour publier/télécharger le modèle de l'API, voir
  section "API de scoring").

Installation des dépendances depuis WSL :

```bash
cd /home/maxime/projects/home-credit-mlops
poetry install
```

`poetry install` installe tous les groupes de dépendances (entraînement,
rapports/EDA, API, tests). Le build Docker de l'API n'installe que les
groupes `main` et `api` (`poetry install --only main,api`), pour une image
plus légère.

Vérification de l'environnement :

```bash
poetry check
poetry run python --version
poetry run pytest -q
```

Les dossiers `data/raw/`, `data/processed/`, `reports/`, `mlartifacts/` et la
base `mlflow.db` sont exclus de Git. Les données Kaggle et les artefacts lourds
doivent donc être transmis séparément si une reproduction complète est attendue.

## Configuration centrale

Le fichier [`configs/default.toml`](configs/default.toml) centralise notamment :

| Section | Paramètre | Valeur par défaut | Rôle |
|---|---|---:|---|
| `dataset` | `test_size` | `0.2` | Part réservée au holdout |
| `dataset` | `random_state` | `42` | Reproductibilité des découpages |
| `business` | `fn_cost` | `10.0` | Coût d'un mauvais client prédit bon |
| `business` | `fp_cost` | `1.0` | Coût d'un bon client prédit mauvais |
| `business` | `threshold_grid_size` | `401` | Résolution minimale de la recherche de seuil |
| `training` | `cv_folds` | `5` | Nombre de plis de validation croisée |
| `training` | `n_jobs` | `1` | Nombre de processus parallèles |
| `mlflow` | `experiment_name` | `home-credit-scoring` | Nom de l'expérience MLflow |
| `serving` | `model_repo_id` | `mxmbrbr/home-credit-mlops` | Dépôt Hugging Face Hub du modèle servi par l'API |
| `serving` | `revision` | `main` | Révision du dépôt HF Hub à télécharger |

## Exécution du pipeline

### 1. Construire le dataset et les rapports EDA

```bash
poetry run python scripts/build_home_credit_dataset.py
```

Sorties principales :

- `data/processed/train_features.parquet` ;
- `data/processed/test_features.parquet` ;
- `reports/AAAAMMJJ_home_credit_data_prep/` ;
- `reports/AAAAMMJJ_home_credit_data_prep/AAAAMMJJ_home_credit_data_prep.xlsx`.

### 2. Réaliser un test de développement

```bash
poetry run python scripts/run_home_credit_experiment.py \
  --campaign-name dev_lightgbm_5k_cv3 \
  --model lightgbm \
  --sampling baseline \
  --sample-size 5000 \
  --cv-folds 3 \
  --n-jobs 1
```

Cette commande valide rapidement la chaîne complète sur un échantillon. Sous
WSL, `--n-jobs 1` limite les duplications de mémoire provoquées par la validation
croisée, le preprocessing et le sur-échantillonnage.

### 3. Comparer plusieurs modèles et stratégies de sampling

```bash
poetry run python scripts/run_home_credit_experiment.py \
  --campaign-name benchmark_models_10k_cv3 \
  --model logistic_regression \
  --model random_forest \
  --model extra_trees \
  --model lightgbm \
  --model xgboost \
  --model mlp \
  --sampling baseline \
  --sampling smote \
  --sample-size 10000 \
  --cv-folds 3 \
  --n-jobs 1
```

Modèles disponibles :

- `logistic_regression` ;
- `random_forest` ;
- `extra_trees` ;
- `lightgbm` ;
- `xgboost` ;
- `mlp` (réseau de neurones `MLPClassifier` ; standardisation automatique des
  features en amont car, contrairement aux arbres, ce modèle est sensible à
  l'échelle).

Stratégies de rééquilibrage disponibles :

- `baseline` : aucun rééchantillonnage ;
- `smote` : sur-échantillonnage SMOTE ;
- `borderline_smote` : sur-échantillonnage des observations proches de la frontière ;
- `adasyn` : génération adaptative d'observations minoritaires ;
- `smote_under` : combinaison SMOTE et sous-échantillonnage aléatoire.

### 4. Enregistrer un modèle final dans le Model Registry

```bash
poetry run python scripts/run_home_credit_experiment.py \
  --campaign-name champion_lightgbm_smote_full_cv5 \
  --model lightgbm \
  --sampling smote \
  --cv-folds 5 \
  --n-jobs 1 \
  --register-model-name home-credit-scoring
```

Le nom du modèle et la stratégie de sampling de cette commande constituent un
exemple de finalisation. Le choix définitif doit reposer sur les résultats CV de
la campagne de comparaison.

Une fois le champion sélectionné, un enregistrement plus rapide est disponible
pour créer une nouvelle version MLflow sans relancer toute la recherche
d'hyperparamètres :

```bash
poetry run python scripts/register_champion_model.py
```

Ce script relit automatiquement le champion (modèle, hyperparamètres, seuil
métier) depuis les artefacts de la dernière campagne (`campaign_metadata.json`),
réentraîne une seule fois sur le dataset préparé, puis enregistre une version
servable dans `home-credit-scoring`. Voir
[docs/mode_emploi_pipeline_ml.md](docs/mode_emploi_pipeline_ml.md#123-model-registry)
pour les options de sélection de campagne source.

### 5. Analyser la fairness du champion

```bash
poetry run python scripts/analyze_fairness.py --source-campaign lgbm_smote_full_cv5
```

Relit les prédictions holdout déjà produites par la campagne (sans rien
réentraîner), les joint à `CODE_GENDER` et `AGE_YEARS`, puis calcule par
groupe le taux de sélection, le recall, le taux de faux positifs et le coût
métier, ainsi que deux indicateurs de disparité (`disparate_impact_ratio`,
règle des 4/5e ; `equal_opportunity_difference`). Rapport écrit dans
`<dossier_campagne>/fairness/` (CSV, graphiques, `fairness.xlsx`). Sur le
champion actuel, l'analyse remonte des écarts significatifs par genre et par
tranche d'âge, non encore traités — voir
[docs/mode_emploi_pipeline_ml.md](docs/mode_emploi_pipeline_ml.md#14-phase-9bis--analyse-de-fairness-biais)
pour le détail des métriques et leurs limites (pas de croisement genre × âge,
pas de precision/F1/ROC AUC par groupe).

## Protocole d'évaluation

1. Un holdout stratifié de 20 % est isolé avant l'entraînement.
2. `GridSearchCV` recherche les hyperparamètres avec `StratifiedKFold`.
3. Des probabilités out-of-fold, dites OOF, sont recalculées sur la partie entraînement.
4. Le seuil métier est choisi sur ces probabilités OOF.
5. Les candidats sont classés par coût métier CV, puis par average precision et ROC AUC CV.
6. Le holdout sert uniquement à estimer la généralisation après sélection.
7. Le meilleur pipeline est réentraîné sur l'ensemble des données disponibles.

Le coût métier normalisé est défini par :

```text
coût métier = (10 × FN + 1 × FP) / nombre d'observations
```

Le seuil n'est donc pas fixé arbitrairement à `0.5`. Il minimise le coût métier
sur les probabilités OOF, puis sa performance est contrôlée sur le holdout.

Les métriques suivies comprennent le coût métier, ROC AUC, average precision,
accuracy, balanced accuracy, précision, rappel, F1-score, Brier score, statistique
de Kolmogorov-Smirnov et matrice de confusion.

## Rapports générés

Chaque campagne crée un dossier :

```text
reports/AAAAMMJJ_home_credit_experiments/<horodatage>_<campagne>/
```

Contenu principal :

- `summary.xlsx` : synthèse de la campagne et comparaison des candidats ;
- `cv_results/` : résultats détaillés des recherches d'hyperparamètres ;
- `diagnostics/<candidat>/` : courbes ROC, précision-rappel et matrices de confusion ;
- `predictions/` : probabilités OOF et holdout au format Parquet ;
- `threshold_optimization/` : courbes et tables coût métier contre seuil ;
- `interpretability/` : feature importance et analyses SHAP du meilleur modèle ;
- `decision_threshold.json` : seuil retenu et hypothèses de coût ;
- `campaign_metadata.json` : paramètres et traçabilité de la campagne.

Les tables et graphiques sont consolidés en classeurs Excel afin de limiter la
dispersion des fichiers. Les diagnostics sont produits pour chaque candidat ;
l'interprétabilité détaillée est réservée au modèle sélectionné.

## MLflow

Lancement de l'interface locale :

```bash
poetry run python scripts/mlflow_ui.py
```

L'interface est ensuite accessible sur <http://127.0.0.1:5000>.

MLflow assure :

- le tracking des paramètres, métriques, tags et artefacts ;
- l'organisation d'une campagne en run parent et runs candidats imbriqués ;
- la journalisation des modèles candidats et du modèle final ;
- le versionnement du modèle final dans le Model Registry ;
- le serving local du modèle enregistré.

Un test rapide sans tracking reste possible :

```bash
poetry run python scripts/run_home_credit_experiment.py \
  --model lightgbm \
  --sample-size 3000 \
  --cv-folds 3 \
  --n-jobs 1 \
  --skip-mlflow
```

## Serving de la décision métier

Le modèle final MLflow encapsule le pipeline entraîné et son seuil métier
versionné. Démarrage d'une version enregistrée :

```bash
MODEL_VERSION=3

poetry run mlflow models serve \
  --model-uri "models:/home-credit-scoring/${MODEL_VERSION}" \
  --host 127.0.0.1 \
  --port 8000 \
  --env-manager local
```

Téléchargement de l'exemple d'entrée généré par MLflow :

```bash
poetry run mlflow artifacts download \
  --artifact-uri "models:/home-credit-scoring/${MODEL_VERSION}" \
  --dst-path /tmp/home-credit-serving-demo
```

Appel de l'endpoint depuis un second terminal, avec un affichage JSON lisible :

```bash
curl -s -X POST http://127.0.0.1:8000/invocations \
  -H "Content-Type: application/json" \
  --data @/tmp/home-credit-serving-demo/serving_input_example.json \
  | python -m json.tool
```

Sans `python -m json.tool`, `curl` affiche la réponse brute sur une seule ligne,
ce qui est normal pour une API REST.

Format de réponse :

```json
{
  "predictions": [
    {
      "default_probability": 0.37,
      "business_threshold": 0.2203,
      "predicted_default": 1,
      "credit_decision": "refused"
    }
  ]
}
```

La valeur `predicted_default = 1` signifie que la probabilité estimée dépasse le
seuil métier. La décision associée est alors `refused`. Une valeur `0` produit
la décision `approved`.

## API de scoring (FastAPI) et déploiement

Le serving MLflow ci-dessus reste utile pour du débogage local rapide contre
une version précise du registry. Pour un déploiement réel (le besoin exprimé
par Chloé Dubois : "API fonctionnelle et déployable, Docker Ready"), le projet
expose une API FastAPI dédiée dans
[`app/`](app/), avec validation
des entrées, documentation Swagger automatique et chargement du modèle une
seule fois au démarrage (jamais par requête).

### Pourquoi une API séparée plutôt que `mlflow models serve`

- Contrôle fin des erreurs (422 structuré sur une entrée invalide, 500 générique
  sans fuite d'informations internes sur une erreur inattendue) ;
- validation métier explicite sur les champs sensibles (âge, revenu, montant
  du crédit, taille du foyer) — pas seulement un contrôle de type ;
- journalisation optionnelle des prédictions dans une base SQLAlchemy
  pour alimenter le monitoring ;
- image Docker autonome et déployable, sans dépendre de l'infrastructure
  MLflow locale (`mlflow.db`/`mlartifacts/`) au runtime.

### Contrat d'entrée

Le modèle attend les mêmes 548 features déjà calculées que celles utilisées à
l'entraînement (mêmes colonnes que `train_features.parquet`). Le schéma
Pydantic de la requête est construit **dynamiquement** au démarrage à partir
de la signature MLflow du modèle réellement chargé — il n'est jamais écrit à
la main, pour ne jamais diverger silencieusement du modèle servi. Au-delà du
type et du caractère obligatoire (vérifiés sur les 548 champs), une
quarantaine de champs ont en plus une règle de borne explicite :

| Champs | Règle |
|---|---|
| `AGE_YEARS` | entre 18 et 100 |
| `AMT_INCOME_TOTAL` | strictement positif |
| `AMT_CREDIT` | strictement positif |
| `CNT_CHILDREN`, `CNT_FAM_MEMBERS` | positif ou nul |
| Flags binaires (`FLAG_MOBIL`, `FLAG_DOCUMENT_2`-`21`, `REG_*_NOT_*_REGION/CITY`, ~33 champs) | 0 ou 1 |
| `EXT_SOURCE_1/2/3`, `EXT_SOURCES_MEAN/MIN/MAX` | entre 0 et 1 |
| `HOUR_APPR_PROCESS_START` | entre 0 et 23 |
| `REGION_RATING_CLIENT`, `REGION_RATING_CLIENT_W_CITY` | entre 1 et 3 |

Les ~500 colonnes restantes (agrégats bureau/previous/installments/credit
card) n'ont volontairement pas de borne : ce sont des sommes, moyennes et
ratios sans maximum métier universel, et une règle générique "positif
uniquement" serait fausse (`DAYS_EMPLOYED`, `DAYS_BIRTH`... sont
légitimement négatifs dans ce dataset).

### Lancer l'API en local

```bash
poetry run uvicorn app.main:app --reload --port 8000
```

Documentation interactive (Swagger) : <http://127.0.0.1:8000/docs>.
Vérification de santé : `curl http://127.0.0.1:8000/health`.

Par défaut, les prédictions sont journalisées dans
`artifacts/production_predictions.db`. La variable d'environnement
`PREDICTION_LOGGING_ENABLED=false` désactive cette persistance et
`API_CALL_LOGGING_ENABLED=false` désactive les logs techniques d'appels HTTP.
`PREDICTION_DB_URL` permet de choisir une autre base compatible SQLAlchemy.
`LOG_FORMAT=json` active des logs console structurés en JSON.
La base peut aussi être initialisée explicitement avec :

```bash
poetry run python scripts/init_production_db.py
```

Pour utiliser PostgreSQL en local hors Docker Compose :

```bash
export PREDICTION_DB_URL="postgresql+psycopg://home_credit:home_credit@127.0.0.1:55432/home_credit_monitoring"
poetry run python scripts/init_production_db.py
poetry run uvicorn app.main:app --reload --port 8000
```

Au premier démarrage, le modèle est téléchargé depuis un dépôt Hugging Face
Hub (`[serving]` dans `configs/default.toml`) — publié au préalable via :

```bash
export HF_TOKEN=hf_...  # jamais commité
poetry run python scripts/export_model_for_serving.py \
  --model-uri models:/home-credit-scoring/3 \
  --hf-repo-id <votre-compte-hf>/home-credit-scoring
```

### Construire et lancer l'image Docker

```bash
docker build -t home-credit-scoring-api .
docker run -p 8000:7860 -e HF_TOKEN=hf_... home-credit-scoring-api
```

Une variante production-like avec PostgreSQL est fournie via Docker Compose :

```bash
docker compose up --build
```

Elle lance deux services :

- `postgres` : base PostgreSQL persistée dans un volume Docker ;
- `api` : API FastAPI connectée à PostgreSQL via `PREDICTION_DB_URL`.

L'API reste disponible sur <http://127.0.0.1:8000/docs>.

Un exemple de requête `curl`, avec un payload réel déjà prêt dans le dépôt
(548 champs, [`tests/fixtures/sample_predict_payload.json`](tests/fixtures/sample_predict_payload.json)) :

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @tests/fixtures/sample_predict_payload.json | python -m json.tool
```

### CI/CD

Deux fichiers de workflow séparés, pour une frontière CI/CD sans ambiguïté :

**[`.github/workflows/ci.yml`](.github/workflows/ci.yml)** — sur chaque push/PR vers `main` :

1. `lint-and-test` — `ruff check` puis `pytest` (couvre aussi l'API) ;
2. `build-and-test-image` — construit l'image Docker et la teste
   **réellement** dans le runner GitHub Actions (conteneur lancé, `/health`
   interrogé jusqu'à ce que le vrai modèle — public sur Hugging Face Hub —
   soit chargé, puis `/predict` testé avec le payload d'exemple ci-dessus).
   Un test d'intégration complet, mais rien n'est déployé : le conteneur est
   détruit à la fin.

**[`.github/workflows/cd.yml`](.github/workflows/cd.yml)** — se déclenche
via `workflow_run` uniquement quand `ci.yml` **réussit sur `main`**
(`needs:` ne fonctionne qu'entre jobs d'un même fichier ; `workflow_run` +
la vérification de `conclusion == 'success'` reproduisent la même garantie
entre deux fichiers). Reconstruit l'image à partir du cache déjà chaud
(partagé via le cache GitHub Actions, `cache-from`/`cache-to: type=gha` —
donc quasi instantané, pas une vraie recompilation), puis la pousse vers le
**GitHub Container Registry** (`ghcr.io/<owner>/<repo>:latest`,
authentification via le `GITHUB_TOKEN` natif, aucun secret à créer).

⚠️ `workflow_run` ne s'active qu'une fois `cd.yml` présent sur `main` : le
tout premier push qui l'ajoute ne déclenche pas encore le CD, les pushes
suivants oui.

**Pourquoi pas un déploiement réel sur Hugging Face Spaces ?** Testé, mais
l'hébergement Docker sur le tier gratuit "cpu-basic" de Hugging Face
nécessite désormais un abonnement PRO (erreur 402 constatée en pratique) —
hors périmètre de ce projet pédagogique.

**Pourquoi `ghcr.io` plutôt qu'un service qui tourne en continu ?** Publier
un service réellement accessible en permanence demanderait un hébergeur
payant (comme HF Spaces PRO). `ghcr.io` reste gratuit et donne un vrai
artefact versionné et récupérable après chaque pipeline réussi — c'est
la partie "build → test → **publie**" du CD ; faire tourner ce conteneur
en continu quelque part resterait la suite logique si un hébergement était
disponible.

## Monitoring production et data drift

L'API journalise les données nécessaires au suivi de production dans une base
SQLAlchemy. SQLite reste le mode local par défaut
(`artifacts/production_predictions.db`), tandis que PostgreSQL est disponible
pour une démo production-like via `docker-compose.yml`.

- `api_call_logs` : méthode HTTP, endpoint, statut, latence, payload JSON,
  type d'erreur, client et user-agent pour tous les appels, y compris les
  erreurs `422` ou `500` ;
- `prediction_logs` : trace complète des prédictions réussies, avec payload
  d'entrée, payload de sortie, probabilité de défaut, décision, latence et
  timestamp ;
- `production_inputs` : snapshot des features envoyées au modèle, une ligne par
  scoring réussi ;
- `production_outputs` : réponse métier persistée, avec probabilité, seuil,
  classe prédite et décision crédit.

Le rapport automatique se lance avec :

```bash
poetry run python scripts/analyze_production_monitoring.py
```

Sorties générées dans `reports/YYYYMMDD_home_credit_monitoring/...` :

- `monitoring_summary.xlsx` : métriques opérationnelles, taux d'erreur,
  latences, distribution des décisions, synthèse du drift ;
- `monitoring_report.html` : rapport visuel consultable localement ;
- PNG : distribution des scores, décisions, latences, variables les plus
  dérivées.

La référence de drift est le dataset d'entraînement préparé
`data/processed/train_features.parquet`. Le script compare les inputs de
production aux features de référence avec des indicateurs simples et
explicables : PSI, KS test, variation de taux de valeurs manquantes. Si le
volume de production est trop faible, le niveau de drift est marqué
`insufficient_data` plutôt que sur-interprété.

### Démo locale du monitoring avec SQLite

SQLite est le mode local le plus simple : aucune base externe n'est nécessaire.
Si une variable `PREDICTION_DB_URL` PostgreSQL est encore présente dans le
terminal, elle doit être retirée avant de démarrer l'API.

1. Démarrer l'API :

```bash
unset PREDICTION_DB_URL
poetry run uvicorn app.main:app --reload --port 8000
```

2. Simuler du trafic de production depuis `test_features.parquet` :

```bash
poetry run python scripts/simulate_production_requests.py \
  --sample-size 100 \
  --invalid-requests 3
```

3. Consulter le résumé opérationnel exposé par l'API :

```bash
curl -s http://127.0.0.1:8000/monitoring/summary | python -m json.tool
```

4. Exporter les logs bruts stockés en base :

```bash
poetry run python scripts/export_production_logs.py
```

5. Ouvrir le dashboard Streamlit :

```bash
poetry run streamlit run dashboard/monitoring_app.py
```

### Démo PostgreSQL avec les quatre tables de traçabilité

Deux modes sont possibles :

- API et PostgreSQL dans Docker Compose : utiliser l'URL interne
  `postgres:5432` déjà configurée dans `docker-compose.yml` ;
- API lancée localement avec Poetry + PostgreSQL dans Docker : utiliser
  `127.0.0.1:55432`, car le hostname `postgres` n'existe que dans le réseau
  Docker Compose.

1. Lancer uniquement PostgreSQL dans Docker :

```bash
docker compose up -d postgres
docker compose ps
docker compose exec postgres pg_isready \
  -U maximebarbier \
  -d home_credit_monitoring
```

2. Démarrer l'API localement avec PostgreSQL :

```bash
export PREDICTION_DB_URL="postgresql+psycopg://maximebarbier:%40udrey29Le@127.0.0.1:55432/home_credit_monitoring"
poetry run uvicorn app.main:app --reload --port 8000
```

Le mot de passe contient un `@` ; dans une URL SQLAlchemy, ce caractère doit
être encodé en `%40`. Sans PostgreSQL démarré, l'API renverra une erreur
`Connection refused` au démarrage.

3. Simuler du trafic :

```bash
poetry run python scripts/simulate_production_requests.py \
  --sample-size 100 \
  --invalid-requests 3
```

4. Vérifier les quatre tables dans PostgreSQL :

```bash
docker compose exec postgres psql \
  -U maximebarbier \
  -d home_credit_monitoring \
  -c "SELECT 'api_call_logs' AS table_name, COUNT(*) FROM api_call_logs
      UNION ALL SELECT 'prediction_logs', COUNT(*) FROM prediction_logs
      UNION ALL SELECT 'production_inputs', COUNT(*) FROM production_inputs
      UNION ALL SELECT 'production_outputs', COUNT(*) FROM production_outputs;"
```

5. Exporter les quatre tables dans un classeur Excel :

```bash
export PREDICTION_DB_URL="postgresql+psycopg://maximebarbier:%40udrey29Le@127.0.0.1:55432/home_credit_monitoring"
poetry run python scripts/export_production_logs.py
```

## Analyse et optimisation des performances

L'étape d'optimisation s'appuie sur les logs de production déjà collectés :
latence totale API (`api_call_logs.latency_ms`), latence modèle
(`production_outputs.latency_ms`), codes HTTP et erreurs. L'objectif n'est pas
seulement de rendre l'API plus rapide, mais de prouver l'impact des choix avec
des mesures reproductibles.

Optimisations intégrées :

- le modèle reste chargé une seule fois au démarrage de l'application ;
- la prédiction est retournée avant l'écriture en base, grâce à des tâches de
  fond FastAPI ;
- les payloads valides ne sont plus dupliqués dans `api_call_logs`, car ils
  sont déjà stockés dans `production_inputs` ;
- ONNX Runtime et GPU sont documentés comme pistes non retenues à ce stade :
  le modèle tabulaire LightGBM et le preprocessing Python/MLflow rendent le
  gain incertain par rapport au risque de régression.

Flux de démonstration recommandé :

```bash
docker compose up -d postgres
curl -s http://127.0.0.1:8000/health | python -m json.tool

poetry run python scripts/simulate_production_requests.py \
  --sample-size 100 \
  --invalid-requests 3

export PREDICTION_DB_URL="postgresql+psycopg://maximebarbier:%40udrey29Le@127.0.0.1:55432/home_credit_monitoring"

poetry run python scripts/profile_api_performance.py \
  --sample-size 50 \
  --warmup-requests 5

poetry run python scripts/analyze_api_performance.py
```

Sorties générées dans `reports/YYYYMMDD_home_credit_performance/...` :

- `api_profile_summary.xlsx` : mesures client par requête et synthèse de
  latence ;
- `cprofile_top.txt` : fonctions les plus coûteuses observées côté client ;
- `performance_summary.xlsx` : latences API, latences modèle, taux d'erreur,
  goulots d'étranglement et décisions d'optimisation ;
- `performance_report.md` : rapport lisible pour expliquer les tests,
  résultats, limites et configuration finale.

## Qualité et limites

Contrôles disponibles (exécutés automatiquement en CI, voir
[`.github/workflows/ci.yml`](.github/workflows/ci.yml), sur chaque push et
pull request vers `main`) :

```bash
poetry run ruff check app dashboard scripts src tests
poetry run pytest -q
```

Limites à conserver dans l'analyse :

- le rapport de coût `FN/FP = 10` constitue une hypothèse pédagogique à valider avec le métier ;
- le tracking et le registry reposent actuellement sur une infrastructure locale ;
- le monitoring PostgreSQL reste un PoC local ; en production réelle, une
  politique de rétention, des droits d'accès, des sauvegardes et une revue RGPD
  seraient nécessaires ;
- les artefacts locaux et les données brutes ne sont pas stockés dans Git ;
- l'analyse de fairness (`scripts/analyze_fairness.py`) remonte des écarts significatifs par
  genre et par tranche d'âge sur le champion actuel, non encore traités.
