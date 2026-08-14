"""Cas de validation explicitement demandes par la consigne de l'etape 2 :
champ obligatoire manquant, valeur hors limites (age -5, revenu 0), type
incorrect (texte au lieu d'un nombre)."""

from __future__ import annotations


def test_valid_payload_returns_200(api_client, valid_payload: dict) -> None:
    response = api_client.post("/predict", json=valid_payload)

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {
        "default_probability",
        "business_threshold",
        "predicted_default",
        "credit_decision",
    }


def test_missing_required_field_returns_422(api_client, valid_payload: dict) -> None:
    payload = dict(valid_payload)
    del payload["AMT_INCOME_TOTAL"]

    response = api_client.post("/predict", json=payload)

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any("AMT_INCOME_TOTAL" in error["loc"] for error in detail)


def test_negative_age_returns_422(api_client, valid_payload: dict) -> None:
    payload = dict(valid_payload, AGE_YEARS=-5)

    response = api_client.post("/predict", json=payload)

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any("AGE_YEARS" in error["loc"] for error in detail)


def test_zero_income_returns_422(api_client, valid_payload: dict) -> None:
    payload = dict(valid_payload, AMT_INCOME_TOTAL=0)

    response = api_client.post("/predict", json=payload)

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any("AMT_INCOME_TOTAL" in error["loc"] for error in detail)


def test_negative_credit_amount_returns_422(api_client, valid_payload: dict) -> None:
    payload = dict(valid_payload, AMT_CREDIT=-1)

    response = api_client.post("/predict", json=payload)

    assert response.status_code == 422


def test_wrong_type_text_instead_of_number_returns_422(api_client, valid_payload: dict) -> None:
    payload = dict(valid_payload, AGE_YEARS="not-a-number")

    response = api_client.post("/predict", json=payload)

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any("AGE_YEARS" in error["loc"] for error in detail)


def test_negative_children_count_returns_422(api_client, valid_payload: dict) -> None:
    payload = dict(valid_payload, CNT_CHILDREN=-1)

    response = api_client.post("/predict", json=payload)

    assert response.status_code == 422


def test_optional_field_can_be_omitted(api_client, valid_payload: dict) -> None:
    payload = dict(valid_payload)
    del payload["CODE_GENDER"]

    response = api_client.post("/predict", json=payload)

    assert response.status_code == 200
