"""Schemas Pydantic de l'API : modèle de requête dynamique + réponse.

Le modèle MLflow attend 548 colonnes déjà calculées. Ecrire ces 548 champs
à la main serait ingérable et impossible à maintenir en phase avec le
modèle réellement chargé. Le modèle de requête est donc construit
dynamiquement au démarrage, à partir de la signature MLflow du modèle
chargé (voir build_request_model). Le type et le caractère obligatoire de
CHACUN des 548 champs viennent directement de cette signature (Pydantic
rejette donc deja tout champ requis manquant ou de mauvais type, sur les
548 colonnes, pas seulement celles listées ci-dessous).

En plus de ca, deux niveaux de validation de bornes explicites :

- `business_rule_validators` : les champs cités nommement par la consigne
  (age, revenu, montant du credit, taille du foyer) ;
- `plausible_range_validators` : bornes connues et non ambigues d'après le
  dictionnaire de données Home Credit (flags binaires, scores EXT_SOURCE
  dans [0, 1], heure de la demande 0-23, notes de region 1-3...).

Volontairement absent : une validation de bornes sur les ~500 colonnes
restantes (agregats bureau/previous/installments/credit_card - sommes,
moyennes, ratios). Contrairement aux catégories ci-dessus, elles n'ont pas
de borne universelle non ambigue (une somme de crédits ou un ratio de
paiement n'a pas de maximum métier fixe), et une règle générique "tout
numérique doit être positif" serait fausse : DAYS_EMPLOYED, DAYS_BIRTH et
les autres colonnes DAYS_* sont légitimement négatives dans ce dataset
(nombre de jours avant la demande).
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Optional

import mlflow.types
import pandas as pd
from mlflow.types import DataType
from pydantic import BaseModel, create_model, field_validator

REQUEST_MODEL_NAME = "CreditScoringRequest"

# Flags binaires (0 ou 1) d'apres le dictionnaire de donnees Home Credit.
# FLAG_OWN_CAR / FLAG_OWN_REALTY sont exclus : ce sont des chaines "Y"/"N",
# pas des entiers, dans la signature du modele.
BINARY_FLAG_COLUMNS = [
    "FLAG_MOBIL",
    "FLAG_EMP_PHONE",
    "FLAG_WORK_PHONE",
    "FLAG_CONT_MOBILE",
    "FLAG_PHONE",
    "FLAG_EMAIL",
    *[f"FLAG_DOCUMENT_{i}" for i in range(2, 22)],
    "REG_REGION_NOT_LIVE_REGION",
    "REG_REGION_NOT_WORK_REGION",
    "LIVE_REGION_NOT_WORK_REGION",
    "REG_CITY_NOT_LIVE_CITY",
    "REG_CITY_NOT_WORK_CITY",
    "LIVE_CITY_NOT_WORK_CITY",
    "DAYS_EMPLOYED_ANOM",
]

# Scores EXT_SOURCE (et leurs agregats mean/min/max, qui restent dans le
# meme intervalle) : normalises entre 0 et 1 par construction.
EXT_SOURCE_UNIT_INTERVAL_COLUMNS = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "EXT_SOURCES_MEAN",
    "EXT_SOURCES_MIN",
    "EXT_SOURCES_MAX",
]


class PredictionResponse(BaseModel):
    default_probability: float
    business_threshold: float
    predicted_default: int
    credit_decision: Literal["approved", "refused"]


def _bounded_validator(field_name: str, *, predicate: Callable[[Any], bool], message: str):
    """Construit un field_validator Pydantic pour un champ optionnel-safe."""

    def _validate(cls, value):  # noqa: ANN001 - signature imposee par pydantic
        if value is not None and not predicate(value):
            raise ValueError(message)
        return value

    return field_validator(field_name)(classmethod(_validate))


def business_rule_validators() -> list[tuple[str, str, Any]]:
    """Validateurs metier pour les champs explicitement testes par la consigne.

    Retourne une liste de (nom_de_champ_cible, nom_du_validateur, validateur).
    Le nom de champ cible sert a ne garder, dans build_request_model, que les
    validateurs dont le champ existe reellement dans la signature du modele
    charge (create_model leve une erreur si on reference un champ absent).
    """
    return [
        (
            "AGE_YEARS",
            "validate_age_years",
            _bounded_validator(
                "AGE_YEARS",
                predicate=lambda v: 18 <= v < 100,
                message="AGE_YEARS must be between 18 and 100.",
            ),
        ),
        (
            "AMT_INCOME_TOTAL",
            "validate_income",
            _bounded_validator(
                "AMT_INCOME_TOTAL",
                predicate=lambda v: v > 0,
                message="AMT_INCOME_TOTAL must be strictly positive.",
            ),
        ),
        (
            "AMT_CREDIT",
            "validate_credit_amount",
            _bounded_validator(
                "AMT_CREDIT",
                predicate=lambda v: v > 0,
                message="AMT_CREDIT must be strictly positive.",
            ),
        ),
        (
            "CNT_CHILDREN",
            "validate_children_count",
            _bounded_validator(
                "CNT_CHILDREN",
                predicate=lambda v: v >= 0,
                message="CNT_CHILDREN cannot be negative.",
            ),
        ),
        (
            "CNT_FAM_MEMBERS",
            "validate_family_members",
            _bounded_validator(
                "CNT_FAM_MEMBERS",
                predicate=lambda v: v >= 0,
                message="CNT_FAM_MEMBERS cannot be negative.",
            ),
        ),
    ]


def plausible_range_validators() -> list[tuple[str, str, Any]]:
    """Validateurs de bornes connues et non ambigues (dictionnaire de donnees).

    Complementaire de business_rule_validators : ici, des categories entieres
    de colonnes plutot que des champs isoles, generees programmatiquement
    pour eviter de dupliquer un validateur quasi identique 33 fois.
    """
    validators: list[tuple[str, str, Any]] = []

    for field_name in BINARY_FLAG_COLUMNS:
        validators.append(
            (
                field_name,
                f"validate_binary_{field_name.lower()}",
                _bounded_validator(
                    field_name,
                    predicate=lambda v: v in (0, 1),
                    message=f"{field_name} must be 0 or 1.",
                ),
            )
        )

    for field_name in EXT_SOURCE_UNIT_INTERVAL_COLUMNS:
        validators.append(
            (
                field_name,
                f"validate_unit_interval_{field_name.lower()}",
                _bounded_validator(
                    field_name,
                    predicate=lambda v: 0.0 <= v <= 1.0,
                    message=f"{field_name} must be between 0 and 1.",
                ),
            )
        )

    validators.append(
        (
            "EXT_SOURCES_NA_COUNT",
            "validate_ext_sources_na_count",
            _bounded_validator(
                "EXT_SOURCES_NA_COUNT",
                predicate=lambda v: 0 <= v <= 3,
                message="EXT_SOURCES_NA_COUNT must be between 0 and 3.",
            ),
        )
    )
    validators.append(
        (
            "HOUR_APPR_PROCESS_START",
            "validate_hour_appr_process_start",
            _bounded_validator(
                "HOUR_APPR_PROCESS_START",
                predicate=lambda v: 0 <= v <= 23,
                message="HOUR_APPR_PROCESS_START must be between 0 and 23.",
            ),
        )
    )
    for field_name in ("REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY"):
        validators.append(
            (
                field_name,
                f"validate_{field_name.lower()}",
                _bounded_validator(
                    field_name,
                    predicate=lambda v: 1 <= v <= 3,
                    message=f"{field_name} must be between 1 and 3.",
                ),
            )
        )

    return validators


def build_request_model(
    input_schema: mlflow.types.Schema,
    extra_validators: list[tuple[str, str, Any]] | None = None,
) -> type[BaseModel]:
    """Construit dynamiquement le modele Pydantic de requete depuis la signature MLflow."""
    fields: dict[str, Any] = {}
    field_names: set[str] = set()
    for col_spec in input_schema.inputs:
        python_type = col_spec.type.to_python()
        field_names.add(col_spec.name)
        if col_spec.required:
            fields[col_spec.name] = (python_type, ...)
        else:
            fields[col_spec.name] = (Optional[python_type], None)

    applicable_validators = {
        validator_name: validator
        for target_field, validator_name, validator in (extra_validators or [])
        if target_field in field_names
    }

    return create_model(
        REQUEST_MODEL_NAME,
        __base__=BaseModel,
        __validators__=applicable_validators,
        **fields,
    )


def coerce_frame_dtypes(frame: pd.DataFrame, input_schema: mlflow.types.Schema) -> pd.DataFrame:
    """Force chaque colonne numerique au dtype numpy exact attendu par MLflow.

    Necessaire car MLflow's schema enforcement (`PyFuncModel.predict`) refuse
    toute conversion qu'il juge non-sure (ex. int64 -> int32), et un
    DataFrame construit depuis un dict Python (via model_dump()) n'a pas
    naturellement ces dtypes precis. Les colonnes string ne sont pas
    recastees : le dtype "object" est deja ce que MLflow attend pour elles.
    Les colonnes numeriques optionnelles absentes (None) sont converties en
    NaN par .astype(), ce qui est le comportement souhaite (aucune des
    colonnes optionnelles de ce modele n'est de type entier, qui ne peut pas
    representer de valeur manquante).
    """
    coerced = frame.copy()
    for col_spec in input_schema.inputs:
        if col_spec.type == DataType.string:
            continue
        coerced[col_spec.name] = coerced[col_spec.name].astype(col_spec.type.to_numpy())
    return coerced
