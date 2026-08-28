# Guide complet d'utilisation - Home Credit MLOps

Ce guide explique comment utiliser le projet de bout en bout : entrainement,
MLflow, API FastAPI, securite par API key, Docker, PostgreSQL, monitoring,
dashboard Streamlit et analyse des performances.

Il peut servir de support de demonstration pendant une revue technique ou une
soutenance.

## 1. Vue d'ensemble

Le projet suit la chaine suivante :

```text
data/raw
-> build_home_credit_dataset.py
-> data/processed/train_features.parquet
-> run_home_credit_experiment.py
-> MLflow + rapports de modelisation
-> export_model_for_serving.py
-> Hugging Face model repo
-> API FastAPI
-> logs SQLite ou PostgreSQL
-> monitoring + Streamlit + rapports de performance
```

Roles des briques principales :

- `scripts/` contient les commandes executables du projet.
- `src/home_credit_mlops/` contient la logique ML reutilisable.
- `app/` contient l'API FastAPI de scoring.
- `dashboard/` contient le dashboard Streamlit de monitoring.
- `Dockerfile` contient l'image de l'API.
- `docker-compose.yml` lance PostgreSQL et/ou l'API en conteneurs.
- `configs/default.toml` centralise la configuration ML et serving.

## 2. Hugging Face, Docker, FastAPI, Streamlit et Gradio

Le depot Hugging Face `mxmbrbr/home-credit-mlops` est un **Model Repo**. Il
stocke le modele MLflow exporte. Il n'heberge pas l'API.

FastAPI est l'API metier de scoring. Elle expose :

- `GET /health` pour verifier que le modele est charge ;
- `POST /predict` pour scorer un client ;
- `GET /monitoring/summary` pour consulter un resume operationnel.

Docker sert a empaqueter l'API et ses dependances pour la faire tourner de
maniere reproductible. Docker Compose permet aussi de lancer PostgreSQL.

Streamlit sert au dashboard local de monitoring. Il lit les logs produits par
l'API et affiche les volumes, erreurs, latences, scores, decisions et drift.

Gradio n'est pas utilise comme application dans ce depot. Il reste une option
possible pour une future demo Hugging Face gratuite, mais l'API principale du
projet est bien FastAPI.

## 3. Prerequis

Depuis le dossier du projet :

```bash
cd /home/maxime/projects/home-credit-mlops
```

Verifier Python et Poetry :

```bash
python --version
poetry --version
```

Installer les dependances :

```bash
poetry install
```

Verifier que l'environnement fonctionne :

```bash
poetry run python -c "import app, home_credit_mlops; print('Imports OK')"
poetry run ruff check .
poetry run pytest -q
```

## 4. Variables d'environnement importantes

### 4.1 API key

`HOME_CREDIT_API_KEY` active la protection par cle API.

Si la variable n'est pas definie, l'API reste ouverte en local. Si elle est
definie, les routes protegees exigent le header HTTP `X-API-Key`.

Generer une cle aleatoire :

```bash
openssl rand -base64 32
```

Activer la cle :

```bash
export HOME_CREDIT_API_KEY="coller-ici-la-cle-generee"
```

Desactiver la cle en local :

```bash
unset HOME_CREDIT_API_KEY
```

### 4.2 Base de logs

`PREDICTION_DB_URL` indique ou stocker les logs API.

Sans `PREDICTION_DB_URL`, l'API utilise SQLite :

```text
artifacts/production_predictions.db
```

Avec PostgreSQL local expose par Docker Compose, utiliser :

```bash
export PREDICTION_DB_URL="postgresql+psycopg://maximebarbier:<MOT_DE_PASSE_ENCODE>@127.0.0.1:55432/home_credit_monitoring"
```

Si le mot de passe contient le caractere `@`, il doit etre encode en `%40`
dans l'URL SQLAlchemy.

Exemple :

```text
mot de passe brut : @monMotDePasse
mot de passe URL  : %40monMotDePasse
```

Ne pas commiter de vraie cle API, de token Hugging Face ou de mot de passe de
production dans Git.

## 5. Lancer l'API en local avec SQLite

Ce mode est le plus simple pour verifier rapidement l'API. Il ne necessite pas
PostgreSQL.

Dans un premier terminal :

```bash
cd /home/maxime/projects/home-credit-mlops
unset PREDICTION_DB_URL
export HOME_CREDIT_API_KEY="demo-home-credit-key"
poetry run uvicorn app.main:app --reload --port 8000
```

Dans le navigateur :

```text
http://127.0.0.1:8000/docs
```

Verifier la sante de l'API :

```bash
curl -s http://127.0.0.1:8000/health | python -m json.tool
```

Resultat attendu :

```json
{
  "status": "ok",
  "model_loaded": true
}
```

Tester une prediction avec la cle API :

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: demo-home-credit-key" \
  --data @tests/fixtures/sample_predict_payload.json \
  | python -m json.tool
```

La reponse attendue ressemble a :

```json
{
  "default_probability": 0.37,
  "business_threshold": 0.220331353025222,
  "predicted_default": 1,
  "credit_decision": "refused"
}
```

Tester la securite :

```bash
curl -i -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  --data @tests/fixtures/sample_predict_payload.json
```

Sans header `X-API-Key`, la reponse doit etre `401 Unauthorized`.

## 6. Lancer PostgreSQL avec Docker et l'API avec Poetry

Ce mode est pratique pour montrer les quatre tables de tracabilite tout en
gardant l'API lancee localement.

Demarrer uniquement PostgreSQL :

```bash
docker compose up -d postgres
docker compose ps
```

Verifier PostgreSQL :

```bash
docker compose exec postgres pg_isready \
  -U maximebarbier \
  -d home_credit_monitoring
```

Configurer l'API locale pour ecrire dans PostgreSQL :

```bash
export PREDICTION_DB_URL="postgresql+psycopg://maximebarbier:<MOT_DE_PASSE_ENCODE>@127.0.0.1:55432/home_credit_monitoring"
export HOME_CREDIT_API_KEY="demo-home-credit-key"
poetry run uvicorn app.main:app --reload --port 8000
```

Point important :

```text
127.0.0.1:55432 = acces PostgreSQL depuis WSL/local
postgres:5432   = acces PostgreSQL depuis un conteneur Docker Compose
```

Si l'API affiche `Connection refused`, PostgreSQL n'est probablement pas
demarre ou le port `55432` n'est pas expose.

## 7. Lancer API et PostgreSQL entierement avec Docker Compose

Ce mode lance l'environnement complet en conteneurs.

```bash
docker compose up -d --build
docker compose ps
```

Voir les logs de l'API :

```bash
docker compose logs -f api
```

Ouvrir Swagger :

```text
http://127.0.0.1:8000/docs
```

Arreter l'affichage des logs :

```text
CTRL+C
```

Arreter les conteneurs :

```bash
docker compose down
```

Supprimer aussi les volumes PostgreSQL et cache modele :

```bash
docker compose down -v
```

Utiliser `docker compose down -v` uniquement si les donnees de demonstration
peuvent etre supprimees.

## 8. Se connecter a PostgreSQL avec pgAdmin 4

Avec PostgreSQL lance par Docker Compose :

```bash
docker compose up -d postgres
```

Parametres pgAdmin :

```text
Server name: Home Credit Local
Host name/address: 127.0.0.1
Port: 55432
Maintenance database: home_credit_monitoring
Username: maximebarbier
Password: mot de passe PostgreSQL configure dans docker-compose.yml
```

Dans pgAdmin, les tables attendues sont :

- `api_call_logs`
- `prediction_logs`
- `production_inputs`
- `production_outputs`

Verifier les volumes depuis le terminal :

```bash
docker compose exec postgres psql \
  -U maximebarbier \
  -d home_credit_monitoring \
  -c "SELECT 'api_call_logs' AS table_name, COUNT(*) FROM api_call_logs
      UNION ALL SELECT 'prediction_logs', COUNT(*) FROM prediction_logs
      UNION ALL SELECT 'production_inputs', COUNT(*) FROM production_inputs
      UNION ALL SELECT 'production_outputs', COUNT(*) FROM production_outputs;"
```

## 9. Simuler du trafic de production

Le script lit `data/processed/test_features.parquet`, retire les colonnes non
attendues et envoie des clients vers `/predict`.

Avec API key en variable d'environnement :

```bash
export HOME_CREDIT_API_KEY="demo-home-credit-key"

poetry run python scripts/simulate_production_requests.py \
  --sample-size 100 \
  --invalid-requests 3
```

Avec API key passee explicitement :

```bash
poetry run python scripts/simulate_production_requests.py \
  --sample-size 100 \
  --invalid-requests 3 \
  --api-key "demo-home-credit-key"
```

Sauvegarder les reponses dans un fichier JSONL :

```bash
poetry run python scripts/simulate_production_requests.py \
  --sample-size 100 \
  --invalid-requests 3 \
  --api-key "demo-home-credit-key" \
  --output-jsonl artifacts/production_requests_demo.jsonl
```

## 10. Exporter les logs de production

Exporter les quatre tables dans un classeur Excel :

```bash
poetry run python scripts/export_production_logs.py
```

Avec PostgreSQL local :

```bash
export PREDICTION_DB_URL="postgresql+psycopg://maximebarbier:<MOT_DE_PASSE_ENCODE>@127.0.0.1:55432/home_credit_monitoring"
poetry run python scripts/export_production_logs.py
```

Sortie attendue :

```text
reports/YYYYMMDD_home_credit_monitoring/YYYYMMDD_HHMMSS_production_logs.xlsx
```

Onglets attendus :

- `api_call_logs`
- `prediction_logs`
- `production_inputs`
- `production_outputs`

## 11. Analyser le monitoring et le data drift

Generer le rapport automatique :

```bash
poetry run python scripts/analyze_production_monitoring.py
```

Sorties attendues :

```text
reports/YYYYMMDD_home_credit_monitoring/YYYYMMDD_HHMMSS_monitoring/
```

Fichiers principaux :

- `monitoring_summary.xlsx`
- `monitoring_report.html`
- `score_distribution.png`
- `decision_distribution.png`
- `latency_distribution.png`
- `top_drift_features.png`

Le drift compare :

```text
Reference : data/processed/train_features.parquet
Production : table production_inputs
```

Indicateurs utilises :

- PSI pour mesurer le deplacement de distribution ;
- KS test pour les variables numeriques ;
- variation du taux de valeurs manquantes ;
- niveau `low`, `moderate`, `high` ou `insufficient_data`.

Le parametre `minimum de lignes pour qualifier le drift` evite de conclure sur
un echantillon trop petit. Si le volume de production est insuffisant, le
dashboard signale `insufficient_data`.

## 12. Lancer le dashboard Streamlit

Le dashboard Streamlit lit la meme base SQLAlchemy que l'API.

Avec SQLite :

```bash
unset PREDICTION_DB_URL
poetry run streamlit run dashboard/monitoring_app.py
```

Avec PostgreSQL local :

```bash
export PREDICTION_DB_URL="postgresql+psycopg://maximebarbier:<MOT_DE_PASSE_ENCODE>@127.0.0.1:55432/home_credit_monitoring"
poetry run streamlit run dashboard/monitoring_app.py
```

Onglets a montrer :

- `Operations` pour les volumes, statuts HTTP et latences ;
- `Scores` pour la distribution des probabilites et decisions ;
- `Data drift` pour les variables derivees ;
- `Logs bruts` pour inspecter les donnees stockees.

## 13. Analyser les performances post-deploiement

Profiler les appels API avec `cProfile` cote client :

```bash
poetry run python scripts/profile_api_performance.py \
  --sample-size 50 \
  --warmup-requests 5 \
  --api-key "demo-home-credit-key"
```

Generer le rapport de performance a partir des logs :

```bash
poetry run python scripts/analyze_api_performance.py
```

Avec PostgreSQL local :

```bash
export PREDICTION_DB_URL="postgresql+psycopg://maximebarbier:<MOT_DE_PASSE_ENCODE>@127.0.0.1:55432/home_credit_monitoring"
poetry run python scripts/analyze_api_performance.py
```

Sorties attendues :

```text
reports/YYYYMMDD_home_credit_performance/YYYYMMDD_HHMMSS_performance/
```

Fichiers principaux :

- `api_profile_summary.xlsx`
- `cprofile_top.txt`
- `performance_summary.xlsx`
- `performance_report.md`

Lecture du rapport :

- si la latence API est nettement superieure a la latence modele, le goulot
  vient plutot de FastAPI, Pydantic, HTTP ou PostgreSQL ;
- si la latence modele domine, l'optimisation doit plutot cibler le pipeline de
  preprocessing ou l'inference ;
- l'optimisation deja integree consiste a journaliser en tache de fond pour ne
  pas bloquer la reponse `/predict`.

## 14. Reconstruire les donnees

Les donnees brutes Kaggle doivent etre presentes dans `data/raw/`.

Lancer la preparation :

```bash
poetry run python scripts/build_home_credit_dataset.py
```

Sorties attendues :

```text
data/processed/train_features.parquet
data/processed/test_features.parquet
reports/YYYYMMDD_home_credit_data_prep/
```

Cette etape realise :

- chargement des tables brutes ;
- nettoyage ;
- agregations ;
- jointures ;
- feature engineering ;
- exports Parquet ;
- rapports EDA et qualite.

## 15. Lancer une experience de modelisation

Run rapide de demonstration :

```bash
poetry run python scripts/run_home_credit_experiment.py \
  --campaign-name demo_lightgbm_10k_cv3 \
  --model lightgbm \
  --sampling baseline \
  --sampling smote \
  --sample-size 10000 \
  --cv-folds 3 \
  --n-jobs 1
```

Run plus complet :

```bash
poetry run python scripts/run_home_credit_experiment.py \
  --campaign-name benchmark_all_models_50k_cv5 \
  --model logistic_regression \
  --model random_forest \
  --model extra_trees \
  --model lightgbm \
  --model xgboost \
  --sampling baseline \
  --sampling smote \
  --sample-size 50000 \
  --cv-folds 5 \
  --n-jobs 1
```

`--n-jobs 1` est recommande pour les runs stables sous WSL. Des valeurs plus
elevees peuvent accelerer certains entrainements mais augmenter fortement la
consommation CPU/RAM.

Sorties attendues :

```text
reports/YYYYMMDD_home_credit_experiments/YYYYMMDD_HHMMSS_<campaign>/
```

Contenu typique :

- `summary.xlsx`
- `cv_results/cv_results.xlsx`
- `diagnostics/`
- `interpretability/`
- `predictions/`
- `decision_threshold.json`
- `campaign_metadata.json`

## 16. MLflow

Lancer l'UI MLflow :

```bash
poetry run python scripts/mlflow_ui.py
```

Ouvrir :

```text
http://127.0.0.1:5000
```

MLflow permet de consulter :

- runs ;
- parametres ;
- metriques ;
- tags ;
- artefacts ;
- model registry ;
- versions de modeles.

Enregistrer un modele champion pendant un run :

```bash
poetry run python scripts/run_home_credit_experiment.py \
  --campaign-name lgbm_smote_register_demo \
  --model lightgbm \
  --sampling smote \
  --sample-size 50000 \
  --cv-folds 5 \
  --n-jobs 1 \
  --register-model-name home-credit-scoring
```

Exporter une version MLflow vers Hugging Face Hub :

```bash
export HF_TOKEN="coller-ici-le-token-hugging-face-si-necessaire"

poetry run python scripts/export_model_for_serving.py \
  --model-uri models:/home-credit-scoring/3 \
  --hf-repo-id mxmbrbr/home-credit-mlops
```

Adapter le numero de version `3` selon la version presente dans le registry
MLflow.

## 17. Tests, lint et CI/CD

Executer les controles locaux :

```bash
poetry run ruff check .
poetry run pytest -q
poetry check
```

La CI GitHub execute :

- lint avec Ruff ;
- tests Pytest ;
- build de l'image Docker ;
- lancement reel du conteneur dans le runner ;
- test `/health` ;
- test `/predict`.

Le CD publie l'image dans GitHub Container Registry si la CI reussit sur
`main`.

## 18. Commandes de demonstration conseillees

### Demonstration API rapide

```bash
unset PREDICTION_DB_URL
export HOME_CREDIT_API_KEY="demo-home-credit-key"
poetry run uvicorn app.main:app --reload --port 8000
```

Dans un second terminal :

```bash
curl -s http://127.0.0.1:8000/health | python -m json.tool

curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: demo-home-credit-key" \
  --data @tests/fixtures/sample_predict_payload.json \
  | python -m json.tool
```

### Demonstration monitoring

```bash
poetry run python scripts/simulate_production_requests.py \
  --sample-size 100 \
  --invalid-requests 3 \
  --api-key "demo-home-credit-key"

poetry run python scripts/analyze_production_monitoring.py
poetry run streamlit run dashboard/monitoring_app.py
```

### Demonstration Docker et PostgreSQL

```bash
docker compose up -d --build
docker compose ps

docker compose exec postgres psql \
  -U maximebarbier \
  -d home_credit_monitoring \
  -c "SELECT COUNT(*) FROM api_call_logs;"
```

### Demonstration performance

```bash
poetry run python scripts/profile_api_performance.py \
  --sample-size 50 \
  --warmup-requests 5 \
  --api-key "demo-home-credit-key"

poetry run python scripts/analyze_api_performance.py
```

## 19. Depannage rapide

### 401 Unauthorized

Cause probable : `HOME_CREDIT_API_KEY` est definie cote serveur, mais la
requete n'envoie pas le bon header.

Verifier la cle cote terminal :

```bash
echo "$HOME_CREDIT_API_KEY"
```

Envoyer le header :

```bash
-H "X-API-Key: valeur-exacte"
```

Redemarrer Uvicorn apres modification de la variable.

### Connection refused sur PostgreSQL

Cause probable : `PREDICTION_DB_URL` pointe vers PostgreSQL, mais le conteneur
n'est pas demarre.

```bash
docker compose up -d postgres
docker compose ps
```

Ou revenir a SQLite :

```bash
unset PREDICTION_DB_URL
```

### Host `postgres` introuvable

`postgres:5432` fonctionne uniquement entre conteneurs Docker Compose. Pour une
API lancee avec Poetry depuis WSL, utiliser `127.0.0.1:55432`.

### Swagger affiche un exemple generique

Le Request Body par defaut est genere depuis la signature MLflow dans
`app/schemas/prediction.py`. Pour un exemple realiste, utiliser :

```text
tests/fixtures/sample_predict_payload.json
```

### Le drift indique beaucoup de variables derivees

Avec un faible trafic simule, le drift peut etre instable. Le dashboard utilise
un minimum de lignes pour eviter de qualifier le drift sur un echantillon trop
petit. Pour une lecture plus robuste, augmenter le volume simule :

```bash
poetry run python scripts/simulate_production_requests.py \
  --sample-size 1000 \
  --api-key "demo-home-credit-key"
```

### WSL ralentit ou crash pendant les entrainements

Limiter le parallelisme :

```bash
--n-jobs 1
```

Eviter les gros runs complets avec SMOTE/ADASYN sur tout le dataset si la RAM
est limitee.

## 20. Checklist finale avant demonstration

Verifier l'etat Git :

```bash
git status -sb
git log --oneline -5
```

Verifier les tests :

```bash
poetry run ruff check .
poetry run pytest -q
```

Verifier l'API :

```bash
curl -s http://127.0.0.1:8000/health | python -m json.tool
```

Verifier les logs :

```bash
poetry run python scripts/export_production_logs.py
```

Verifier le dashboard :

```bash
poetry run streamlit run dashboard/monitoring_app.py
```

Verifier les rapports :

```bash
find reports -maxdepth 3 -type f | sort | tail -40
```

