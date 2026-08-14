"""Schemas Pydantic de l'API : modele de requete dynamique + reponse.

Le modele MLflow attend 548 colonnes deja calculees. Ecrire ces 548 champs
a la main serait ingerable et impossible a maintenir en phase avec le
modele reellement charge. Le modele de requete est donc construit
dynamiquement au demarrage, a partir de la signature MLflow du modele
charge (voir build_request_model), avec des validateurs metier explicites
uniquement sur les quelques champs pour lesquels la consigne demande des
verifications de bornes (age, revenu, montant du credit, taille du foyer).

Volontairement absent : une regle generique "tout numerique doit etre
positif" - des colonnes comme DAYS_EMPLOYED ou DAYS_BIRTH sont legitimement
negatives dans ce dataset (nombre de jours avant la demande).
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Optional

import mlflow.types
import pandas as pd
from mlflow.types import DataType
from pydantic import BaseModel, create_model, field_validator

REQUEST_MODEL_NAME = "CreditScoringRequest"


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
