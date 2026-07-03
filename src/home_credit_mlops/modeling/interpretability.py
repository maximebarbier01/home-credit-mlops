from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy import sparse
from sklearn.pipeline import Pipeline

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)


def _get_feature_name_mapping(preprocessor) -> pd.DataFrame:
    transformed_features = preprocessor.get_feature_names_out().tolist()
    transformer_columns = {
        name: list(columns)
        for name, _, columns in preprocessor.transformers_
        if name != "remainder"
    }
    numeric_columns = transformer_columns.get("numeric", [])
    categorical_columns = transformer_columns.get("categorical", [])

    source_features: list[str] = []
    feature_kinds: list[str] = []
    ordered_categorical = sorted(categorical_columns, key=len, reverse=True)

    for transformed_feature in transformed_features:
        if transformed_feature.startswith("numeric__"):
            source_features.append(transformed_feature.removeprefix("numeric__"))
            feature_kinds.append("numeric")
            continue

        if transformed_feature.startswith("categorical__"):
            encoded_name = transformed_feature.removeprefix("categorical__")
            source_feature = encoded_name
            for column in ordered_categorical:
                if encoded_name == column or encoded_name.startswith(f"{column}_"):
                    source_feature = column
                    break
            source_features.append(source_feature)
            feature_kinds.append("categorical")
            continue

        stripped = transformed_feature.split("__", maxsplit=1)[-1]
        feature_kinds.append("numeric" if stripped in numeric_columns else "unknown")
        source_features.append(stripped)

    return pd.DataFrame(
        {
            "transformed_feature": transformed_features,
            "source_feature": source_features,
            "feature_kind": feature_kinds,
        }
    )


def _build_transformed_feature_frame(
    pipeline: Pipeline,
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(features)
    if sparse.issparse(transformed):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed, dtype=np.float32)

    mapping = _get_feature_name_mapping(preprocessor)
    transformed_frame = pd.DataFrame(
        transformed,
        index=features.index,
        columns=mapping["transformed_feature"].tolist(),
    )
    return transformed_frame, mapping


def compute_feature_importance(pipeline: Pipeline) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = pipeline.named_steps["model"]
    mapping = _get_feature_name_mapping(pipeline.named_steps["preprocessor"])

    if hasattr(model, "feature_importances_"):
        importance_values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        importance_values = np.abs(np.ravel(model.coef_)).astype(float)
    else:
        raise ValueError(
            f"Feature importance export is not configured for model type {model.__class__.__name__}."
        )

    transformed_importance = mapping.assign(importance=importance_values)
    transformed_importance = transformed_importance.sort_values(
        "importance", ascending=False
    ).reset_index(drop=True)
    grouped_importance = (
        transformed_importance.groupby(["source_feature", "feature_kind"], as_index=False)[
            "importance"
        ]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return transformed_importance, grouped_importance


def export_feature_importance(
    pipeline: Pipeline,
    output_dir: str | Path,
    *,
    top_n: int = 20,
) -> dict[str, str]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    transformed_importance, grouped_importance = compute_feature_importance(pipeline)
    transformed_path = destination / "feature_importance_transformed.csv"
    grouped_path = destination / "feature_importance_grouped.csv"
    transformed_importance.to_csv(transformed_path, index=False)
    grouped_importance.to_csv(grouped_path, index=False)

    top_grouped = grouped_importance.head(top_n).sort_values("importance", ascending=True)
    plt.figure(figsize=(10, 8))
    sns.barplot(data=top_grouped, x="importance", y="source_feature", color="#4C78A8")
    plt.title("Top grouped feature importances")
    plt.xlabel("Importance")
    plt.ylabel("")
    plt.tight_layout()
    grouped_plot_path = destination / "feature_importance_grouped.png"
    plt.savefig(grouped_plot_path, dpi=150)
    plt.close()

    return {
        "transformed_csv": transformed_path.as_posix(),
        "grouped_csv": grouped_path.as_posix(),
        "grouped_plot": grouped_plot_path.as_posix(),
    }


def _build_shap_explainer(model, background_frame: pd.DataFrame):
    if hasattr(model, "booster_") or model.__class__.__module__.startswith(("lightgbm", "sklearn.ensemble")):
        return shap.TreeExplainer(model)
    if hasattr(model, "coef_"):
        background = background_frame.sample(
            n=min(500, len(background_frame)),
            random_state=42,
        )
        return shap.LinearExplainer(model, background)
    raise ValueError(
        f"SHAP analysis is not configured for model type {model.__class__.__name__}."
    )


def _extract_binary_class_values(explanation) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(explanation.values)
    base_values = np.asarray(explanation.base_values)

    if values.ndim == 3:
        values = values[:, :, 1]
        if base_values.ndim > 1:
            base_values = base_values[:, 1]
    return values, base_values


def _sample_reference_rows(
    features: pd.DataFrame,
    client_ids: pd.Series | None,
    *,
    sample_size: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series | None]:
    if sample_size <= 0 or len(features) <= sample_size:
        sampled_features = features.copy()
    else:
        sampled_features = features.sample(n=sample_size, random_state=random_state).copy()

    if client_ids is None:
        return sampled_features, None

    aligned_client_ids = client_ids.loc[sampled_features.index]
    return sampled_features, aligned_client_ids


def export_shap_analysis(
    pipeline: Pipeline,
    reference_features: pd.DataFrame,
    client_ids: pd.Series | None,
    output_dir: str | Path,
    *,
    sample_size: int = 1_500,
    local_examples: int = 3,
    max_display: int = 20,
    random_state: int = 42,
) -> dict[str, str]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    sampled_features, sampled_client_ids = _sample_reference_rows(
        reference_features,
        client_ids,
        sample_size=sample_size,
        random_state=random_state,
    )
    transformed_frame, mapping = _build_transformed_feature_frame(pipeline, sampled_features)
    model = pipeline.named_steps["model"]
    explainer = _build_shap_explainer(model, transformed_frame)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
            category=UserWarning,
        )
        explanation = explainer(transformed_frame)

    values, base_values = _extract_binary_class_values(explanation)
    shap_frame = pd.DataFrame(
        values,
        index=transformed_frame.index,
        columns=transformed_frame.columns,
    )
    probabilities = pipeline.predict_proba(sampled_features)[:, 1]

    source_lookup = mapping.set_index("transformed_feature")["source_feature"]
    feature_kind_lookup = mapping.set_index("transformed_feature")["feature_kind"]

    global_transformed = pd.DataFrame(
        {
            "transformed_feature": transformed_frame.columns,
            "source_feature": source_lookup.loc[transformed_frame.columns].values,
            "feature_kind": feature_kind_lookup.loc[transformed_frame.columns].values,
            "mean_abs_shap": shap_frame.abs().mean(axis=0).values,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    global_grouped = (
        global_transformed.groupby(["source_feature", "feature_kind"], as_index=False)[
            "mean_abs_shap"
        ]
        .sum()
        .sort_values("mean_abs_shap", ascending=False)
    )

    transformed_path = destination / "shap_global_transformed.csv"
    grouped_path = destination / "shap_global_grouped.csv"
    global_transformed.to_csv(transformed_path, index=False)
    global_grouped.to_csv(grouped_path, index=False)

    shap.summary_plot(values, transformed_frame, show=False, max_display=max_display)
    plt.tight_layout()
    beeswarm_path = destination / "shap_summary_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
    plt.close()

    shap.summary_plot(
        values,
        transformed_frame,
        plot_type="bar",
        show=False,
        max_display=max_display,
    )
    plt.tight_layout()
    bar_path = destination / "shap_summary_bar.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()

    grouped_top = global_grouped.head(max_display).sort_values("mean_abs_shap", ascending=True)
    plt.figure(figsize=(10, 8))
    sns.barplot(data=grouped_top, x="mean_abs_shap", y="source_feature", color="#F28E2B")
    plt.title("Top grouped SHAP importances")
    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("")
    plt.tight_layout()
    grouped_plot_path = destination / "shap_grouped_bar.png"
    plt.savefig(grouped_plot_path, dpi=150)
    plt.close()

    if sampled_client_ids is not None:
        client_lookup = sampled_client_ids.to_dict()
    else:
        client_lookup = {index: f"sample_{position}" for position, index in enumerate(sampled_features.index, start=1)}

    probability_frame = pd.DataFrame(
        {
            "client_id": [client_lookup[index] for index in sampled_features.index],
            "predicted_probability": probabilities,
        },
        index=sampled_features.index,
    )
    top_risky = probability_frame.nlargest(min(local_examples, len(probability_frame)), "predicted_probability")
    top_safe = probability_frame.nsmallest(min(local_examples, len(probability_frame)), "predicted_probability")
    selected_indices = pd.Index(top_risky.index.tolist() + top_safe.index.tolist()).unique()

    local_rows: list[pd.DataFrame] = []
    grouped_local_rows: list[pd.DataFrame] = []
    position_lookup = {index: position for position, index in enumerate(sampled_features.index)}

    for chart_index, row_index in enumerate(selected_indices, start=1):
        position = position_lookup[row_index]
        local_frame = pd.DataFrame(
            {
                "client_id": client_lookup[row_index],
                "predicted_probability": probabilities[position],
                "transformed_feature": transformed_frame.columns,
                "source_feature": source_lookup.loc[transformed_frame.columns].values,
                "feature_value": transformed_frame.iloc[position].values,
                "shap_value": shap_frame.iloc[position].values,
            }
        )
        local_frame["abs_shap"] = local_frame["shap_value"].abs()
        local_rows.append(
            local_frame.sort_values("abs_shap", ascending=False).head(max_display).reset_index(drop=True)
        )

        grouped_local = (
            local_frame.groupby("source_feature", as_index=False)
            .agg(
                shap_value=("shap_value", "sum"),
                abs_shap=("abs_shap", "sum"),
            )
            .sort_values("abs_shap", ascending=False)
            .head(max_display)
            .reset_index(drop=True)
        )
        grouped_local.insert(0, "predicted_probability", probabilities[position])
        grouped_local.insert(0, "client_id", client_lookup[row_index])
        grouped_local_rows.append(grouped_local)

        if chart_index <= min(3, len(selected_indices)):
            explanation_row = shap.Explanation(
                values=values[position],
                base_values=base_values[position] if np.ndim(base_values) > 0 else base_values,
                data=transformed_frame.iloc[position].values,
                feature_names=transformed_frame.columns.tolist(),
            )
            shap.plots.waterfall(explanation_row, max_display=max_display, show=False)
            plt.tight_layout()
            waterfall_path = destination / f"shap_waterfall_{client_lookup[row_index]}.png"
            plt.savefig(waterfall_path, dpi=150, bbox_inches="tight")
            plt.close()

    local_path = destination / "shap_local_transformed.csv"
    grouped_local_path = destination / "shap_local_grouped.csv"
    if local_rows:
        pd.concat(local_rows, ignore_index=True).to_csv(local_path, index=False)
        pd.concat(grouped_local_rows, ignore_index=True).to_csv(grouped_local_path, index=False)
    else:
        pd.DataFrame().to_csv(local_path, index=False)
        pd.DataFrame().to_csv(grouped_local_path, index=False)

    return {
        "global_transformed_csv": transformed_path.as_posix(),
        "global_grouped_csv": grouped_path.as_posix(),
        "summary_beeswarm": beeswarm_path.as_posix(),
        "summary_bar": bar_path.as_posix(),
        "grouped_plot": grouped_plot_path.as_posix(),
        "local_transformed_csv": local_path.as_posix(),
        "local_grouped_csv": grouped_local_path.as_posix(),
    }
