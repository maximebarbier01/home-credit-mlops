# Image d'API de scoring credit. Le modele (~130 Mo) n'est jamais embarque
# ici : il est telecharge depuis Hugging Face Hub au demarrage du conteneur
# (voir app/services/model_service.py). Cela garde l'image
# legere et permet de promouvoir un nouveau champion sans reconstruire.

FROM python:3.12-slim AS builder
WORKDIR /app

RUN pip install --no-cache-dir poetry==2.3.4
RUN poetry config virtualenvs.in-project true

# Couche dependances seule d'abord, pour beneficier du cache Docker tant
# que pyproject.toml/poetry.lock ne changent pas.
COPY pyproject.toml poetry.lock ./
RUN poetry install --only main,api --no-root --no-interaction

# Le package racine (pyproject.toml declare readme = "README.md") a besoin
# de ces fichiers pour s'installer.
COPY README.md ./
COPY src ./src
COPY app ./app
COPY configs ./configs
RUN poetry install --only main,api --no-interaction

FROM python:3.12-slim AS runtime
WORKDIR /app

# lightgbm (et potentiellement xgboost) chargent une bibliotheque native
# compilee qui depend de libgomp (runtime OpenMP), absente de l'image slim
# par defaut.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 appuser

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src ./src
COPY --from=builder /app/app ./app
COPY --from=builder /app/configs ./configs

# Le modele est telecharge dans artifacts/hf_model_cache (chemin relatif a
# la racine du projet, /app ici) au demarrage du conteneur. Le sous-dossier
# doit exister avant le montage du volume Docker pour que le volume herite
# de droits compatibles avec l'utilisateur non-root.
RUN mkdir -p /app/artifacts/hf_model_cache/.cache/huggingface \
    && chown -R appuser:appuser /app/artifacts

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER appuser

# Convention Hugging Face Spaces (SDK Docker) : le conteneur doit ecouter
# sur le port 7860.
EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
