# Mode d'emploi du pipeline Home Credit MLOps

## 1. Objectif du projet

Le projet sert a construire un modele de credit scoring pour predire la probabilite de defaut d'un client.

La logique metier est la suivante :
- un faux negatif coute beaucoup plus cher qu'un faux positif ;
- on entraine plusieurs modeles ;
- on compare leurs performances techniques et metier ;
- on choisit aussi le meilleur seuil de decision, pas seulement le meilleur modele.

Dans la configuration actuelle :
- cout FN = 10
- cout FP = 1

Ces valeurs sont definies dans [configs/default.toml](/home/maxime/projects/home-credit-mlops/configs/default.toml).

---

## 2. Les 3 points d'entree a connaitre

### 2.1 Construire le dataset

Script : [scripts/build_home_credit_dataset.py](/home/maxime/projects/home-credit-mlops/scripts/build_home_credit_dataset.py)

Commande :

```bash
poetry run python scripts/build_home_credit_dataset.py
```

Ce script lance [src/home_credit_mlops/data/home_credit.py](/home/maxime/projects/home-credit-mlops/src/home_credit_mlops/data/home_credit.py), qui :
- charge les fichiers bruts dans `data/raw/`
- nettoie les variables principales
- cree des variables derivees
- agrege les tables secondaires
- fusionne tout avec `SK_ID_CURR`
- exporte un dataset train et un dataset test

Sorties principales :
- `data/processed/train_features.parquet`
- `data/processed/test_features.parquet`
- `reports/home_credit_eda/`
- `reports/home_credit_eda/home_credit_eda.xlsx`

---

### 2.2 Lancer une experience ML complete

Script : [scripts/run_home_credit_experiment.py](/home/maxime/projects/home-credit-mlops/scripts/run_home_credit_experiment.py)

Commande type :

```bash
poetry run python scripts/run_home_credit_experiment.py --model lightgbm --sample-size 5000 --cv-folds 3
```

Ce script lance [src/home_credit_mlops/modeling/benchmark.py](/home/maxime/projects/home-credit-mlops/src/home_credit_mlops/modeling/benchmark.py).

C'est le coeur du pipeline ML.

---

### 2.3 Ouvrir MLflow

Script : [scripts/mlflow_ui.py](/home/maxime/projects/home-credit-mlops/scripts/mlflow_ui.py)

Commande :

```bash
poetry run python scripts/mlflow_ui.py
```

Cela ouvre l'interface MLflow pour visualiser :
- les runs
- les metriques
- les parametres
- les artefacts
- les modeles enregistres

---

## 3. La chaine de build ML, dans l'ordre

### Etape 1. Preparation des donnees

Fichier principal : [src/home_credit_mlops/data/home_credit.py](/home/maxime/projects/home-credit-mlops/src/home_credit_mlops/data/home_credit.py)

Cette couche fait le travail "data engineering / feature engineering tabulaire" :
- lecture des tables Kaggle
- nettoyage de certaines anomalies
- creation de ratios et indicateurs metier
- aggregation des historiques client
- fusion finale en un seul dataset modele

Exemples de tables agregees :
- `bureau.csv`
- `bureau_balance.csv`
- `previous_application.csv`
- `POS_CASH_balance.csv`
- `installments_payments.csv`
- `credit_card_balance.csv`

Cette etape produit deja des rapports utiles :
- distribution de la target
- missing values
- profils de tables
- coverage des jointures
- un classeur Excel qui regroupe les CSV et JSON du dossier de rapports en onglets

---

### Etape 2. Nettoyage / preprocessing modele

Fichier principal : [src/home_credit_mlops/features/preprocessing.py](/home/maxime/projects/home-credit-mlops/src/home_credit_mlops/features/preprocessing.py)

Ici, on passe de "dataset propre" a "dataset modelisable".

Le module :
- separe `X` et `y`
- identifie les colonnes numeriques et categorielles
- applique une imputation mediane sur le numerique
- applique une imputation par la modalite la plus frequente sur le categoriel
- applique un one-hot encoding sur le categoriel

Important : ce preprocessing n'est pas fait a la main avant le modele. Il est integre dans un `Pipeline` scikit-learn. Donc il est bien rejoue proprement dans la cross-validation et en inference.

---

### Etape 3. EDA et diagnostic

Fichiers principaux :
- [src/home_credit_mlops/eda/diagnostics.py](/home/maxime/projects/home-credit-mlops/src/home_credit_mlops/eda/diagnostics.py)
- [src/home_credit_mlops/eda/visualisation.py](/home/maxime/projects/home-credit-mlops/src/home_credit_mlops/eda/visualisation.py)

Cette couche sert a comprendre les donnees et a documenter leur qualite.

Elle exporte notamment :
- schema des colonnes
- resumes numeriques et categoriels
- distribution de la target
- rapports de valeurs manquantes
- variables les plus associees a la target
- modalites associees positivement ou negativement au risque

Cette etape est appelee automatiquement depuis `benchmark.py`, sauf si tu passes `--skip-eda`.

---

### Etape 4. Entrainement et comparaison des modeles

Fichier principal : [src/home_credit_mlops/modeling/benchmark.py](/home/maxime/projects/home-credit-mlops/src/home_credit_mlops/modeling/benchmark.py)

C'est la brique centrale du projet.

Elle fait :
- chargement du dataset prepare
- eventuel sous-echantillonnage pour aller plus vite
- split train / holdout
- creation du pipeline preprocessing + modele
- `GridSearchCV`
- cross-validation stratifiee
- comparaison de plusieurs modeles
- calcul des probabilites OOF
- recherche du meilleur seuil metier
- evaluation finale sur holdout
- selection du meilleur modele
- refit final sur l'ensemble des donnees

Les modeles candidats sont definis dans [src/home_credit_mlops/modeling/candidates.py](/home/maxime/projects/home-credit-mlops/src/home_credit_mlops/modeling/candidates.py).

Modeles disponibles actuellement :
- `logistic_regression`
- `random_forest`
- `extra_trees`
- `lightgbm`

---

### Etape 5. Evaluation des performances

Fichier principal : [src/home_credit_mlops/modeling/metrics.py](/home/maxime/projects/home-credit-mlops/src/home_credit_mlops/modeling/metrics.py)

Le projet ne se limite pas a l'AUC.

Metriques suivies :
- business cost
- business score
- ROC AUC
- average precision
- accuracy
- balanced accuracy
- precision
- recall
- F1
- Brier score
- KS statistic
- matrice de confusion

La logique importante est la suivante :
- pendant la CV, on optimise d'abord le score metier ;
- puis on cherche le meilleur seuil sur les probabilites OOF ;
- ensuite on evalue ce seuil sur le holdout.

C'est exactement ce que demande la consigne metier.

---

### Etape 6. Choix du seuil de decision

Toujours dans [src/home_credit_mlops/modeling/metrics.py](/home/maxime/projects/home-credit-mlops/src/home_credit_mlops/modeling/metrics.py).

Le seuil n'est pas fixe a `0.5`.

Le projet :
- balaie une grille de seuils
- calcule le cout metier pour chaque seuil
- garde celui qui minimise le cout
- departage a cout egal avec le meilleur recall

Le seuil retenu est exporte dans :
- `decision_threshold.json`

---

### Etape 7. Interpretabilite

Fichier principal : [src/home_credit_mlops/modeling/interpretability.py](/home/maxime/projects/home-credit-mlops/src/home_credit_mlops/modeling/interpretability.py)

Cette couche sert a expliquer le modele.

Elle produit :
- feature importance globale
- feature importance groupee par variable source
- SHAP global
- SHAP local sur quelques clients a risque et quelques clients peu risqu?s

C'est la partie utile pour la transparence vis-a-vis d'un charge d'etudes.

---

### Etape 8. Packaging des resultats

Fichier principal : [src/home_credit_mlops/reporting/excel.py](/home/maxime/projects/home-credit-mlops/src/home_credit_mlops/reporting/excel.py)

Le pipeline regroupe automatiquement les sorties en classeurs Excel :
- un classeur resume a la racine du run
- un classeur par dossier de sortie (`eda`, `diagnostics`, `interpretability`, etc.)

Cela evite d'avoir trop de fichiers isoles a ouvrir un par un.

---

### Etape 9. Tracking MLOps avec MLflow

Fichier principal : [src/home_credit_mlops/mlflow_utils.py](/home/maxime/projects/home-credit-mlops/src/home_credit_mlops/mlflow_utils.py)

Quand tu lances `run_home_credit_experiment.py`, MLflow peut :
- creer ou reutiliser l'experiment
- tracer les parametres
- tracer les metriques
- stocker les artefacts
- logger les modeles candidats
- logger le modele final
- enregistrer le meilleur modele dans le registry

Tu peux desactiver le tracking pour aller vite avec :

```bash
poetry run python scripts/run_home_credit_experiment.py --skip-mlflow
```

---

## 4. Lecture pratique de l'arborescence

### `scripts/`
Ce sont les points d'entree executables.

### `src/home_credit_mlops/data/`
Preparation des donnees et construction du dataset final.

### `src/home_credit_mlops/features/`
Preprocessing reutilisable par les modeles.

### `src/home_credit_mlops/eda/`
Rapports et visualisations EDA.

### `src/home_credit_mlops/modeling/`
Modeles, benchmark, metriques et interpretabilite.

### `src/home_credit_mlops/reporting/`
Exports finaux, notamment Excel.

### `configs/`
Configuration centrale du projet.

---

## 5. Workflow recommande au quotidien

### Cas 1. Tu modifies la preparation des donnees
1. Tu modifies `data/home_credit.py`
2. Tu rebuilds le dataset
3. Tu relances une experience ML
4. Tu compares les resultats avec MLflow

### Cas 2. Tu modifies un modele ou ses hyperparametres
1. Tu modifies `modeling/candidates.py`
2. Tu relances `run_home_credit_experiment.py`
3. Tu regardes `benchmark_results.csv` et MLflow

### Cas 3. Tu veux aller vite
Utilise par exemple :

```bash
poetry run python scripts/run_home_credit_experiment.py --model lightgbm --sample-size 3000 --cv-folds 3 --skip-mlflow
```

### Cas 4. Tu veux une run complete pour livrable
Utilise plutot :

```bash
poetry run python scripts/run_home_credit_experiment.py --model lightgbm --cv-folds 5 --register-model-name home-credit-scoring
```

---

## 6. Les fichiers de sortie a connaitre absolument

### Dataset
- `data/processed/train_features.parquet`
- `data/processed/test_features.parquet`

### Step 1 data prep
- `reports/home_credit_eda/table_profiles.csv`
- `reports/home_credit_eda/merge_coverage.csv`
- `reports/home_credit_eda/train_features_missingness.csv`
- `reports/home_credit_eda/dataset_metadata.json`
- `reports/home_credit_eda/home_credit_eda.xlsx`

### Experiment ML
Dans `reports/home_credit_experiments/<timestamp>/` :
- `benchmark_results.csv`
- `experiment_metadata.json`
- `decision_threshold.json`
- `best_model_test_predictions.csv` si test fourni
- dossiers `eda/`, `diagnostics/`, `interpretability/`, `predictions/`, `cv_results/`
- `summary.xlsx`

---

## 7. Comment raconter le projet a l'oral

Tu peux le presenter comme ca :

> J'ai structure le projet autour d'une seule chaine ML claire. D'abord je construis un dataset client consolide a partir des tables brutes. Ensuite j'applique un preprocessing integre au pipeline modele. Puis je compare plusieurs algorithmes avec cross-validation et une metrique metier qui penalise davantage les faux negatifs. Je n'utilise pas un seuil fixe a 0.5 : j'optimise le seuil de decision sur les probabilites OOF. Enfin, j'exporte a la fois des diagnostics de performance, des explications globales et locales avec SHAP, et je trace les experimentations avec MLflow.

---

## 8. Le point le plus important a retenir

Le coeur du projet, c'est ce trio :
- [src/home_credit_mlops/data/home_credit.py](/home/maxime/projects/home-credit-mlops/src/home_credit_mlops/data/home_credit.py)
- [src/home_credit_mlops/features/preprocessing.py](/home/maxime/projects/home-credit-mlops/src/home_credit_mlops/features/preprocessing.py)
- [src/home_credit_mlops/modeling/benchmark.py](/home/maxime/projects/home-credit-mlops/src/home_credit_mlops/modeling/benchmark.py)

Si tu comprends bien comment ces trois fichiers s'enchainent, tu comprends l'architecture du projet.
