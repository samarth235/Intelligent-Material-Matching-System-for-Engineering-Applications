"""
Material selection API and React app server.

Run the backend with:
    python app.py

During development, run the React app from ui/frontend with npm run dev.
For production, build the React app and Flask will serve ui/frontend/dist.
The default API and production URL is http://127.0.0.1:5050.
"""

from __future__ import annotations

import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

from recommender import MaterialRecommender


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIST = BASE_DIR / "ui" / "frontend" / "dist"
DATASET_PATH = BASE_DIR / ("materials_large.csv" if (BASE_DIR / "materials_large.csv").exists() else "materials.csv")

app = Flask(__name__, static_folder=str(FRONTEND_DIST), static_url_path="")
recommender = MaterialRecommender(str(DATASET_PATH))
ML_DIR = BASE_DIR / "models"

FEATURE_COLUMNS = [
    "Density_g_cm3",
    "Yield_Strength_MPa",
    "Tensile_Strength_MPa",
    "Hardness_HB",
    "Corrosion_Resistance_0_10",
    "Thermal_Conductivity_W_mK",
    "Cost_Index_1_10",
    "Melting_Point_C",
    "Machinability",
]

ENGINEERED_FEATURES = ["strength_weight", "cost_perf", "thermal_strength_ratio"]
MODEL_FEATURE_COLUMNS = FEATURE_COLUMNS + ENGINEERED_FEATURES
APPLICATION_ALIASES = {
    "Aerospace Structures": "Aircraft Structures",
    "Castings": "Casting",
}


SEARCH_COLUMNS = {
    "tensile_strength": {
        "label": "Tensile Strength",
        "unit": "MPa",
        "column": "Tensile_Strength_MPa",
        "higherIsBetter": True,
    },
    "density": {
        "label": "Density",
        "unit": "g/cm3",
        "column": "Density_g_cm3",
        "higherIsBetter": False,
    },
    "corrosion": {
        "label": "Corrosion Resistance",
        "unit": "/10",
        "column": "Corrosion_Resistance_0_10",
        "higherIsBetter": True,
    },
    "cost": {
        "label": "Cost Index",
        "unit": "/10",
        "column": "Cost_Index_1_10",
        "higherIsBetter": False,
    },
    "thermal": {
        "label": "Thermal Conductivity",
        "unit": "W/mK",
        "column": "Thermal_Conductivity_W_mK",
        "higherIsBetter": True,
    },
}

DISPLAY_COLUMNS = [
    "Material",
    "Alloy_Type",
    "Density_g_cm3",
    "Yield_Strength_MPa",
    "Tensile_Strength_MPa",
    "Hardness_HB",
    "Corrosion_Resistance_0_10",
    "Thermal_Conductivity_W_mK",
    "Cost_Index_1_10",
    "Melting_Point_C",
    "Machinability",
    "Primary_Application",
]


ml_load_errors: list[str] = []


def load_pickle(path: Path) -> Any | None:
    if not path.exists():
        ml_load_errors.append(f"Missing model artifact: {path.name}")
        return None
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except (OSError, EOFError, pickle.UnpicklingError, AttributeError, ValueError) as exc:
        ml_load_errors.append(f"Unable to load {path.name}: {exc}")
        return None


ml_model = load_pickle(ML_DIR / "random_forest_model.pkl")
ml_scaler = load_pickle(ML_DIR / "scaler.pkl")
ml_label_encoders = load_pickle(ML_DIR / "label_encoders.pkl") or {}
ml_application_encoder = ml_label_encoders.get("Primary_Application")
if ml_application_encoder is None:
    ml_load_errors.append("Primary_Application encoder is unavailable")
ml_ready = all([ml_model, ml_scaler, ml_application_encoder])


def clean_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def frame_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    safe = df.replace({np.nan: None})
    return [{key: clean_value(value) for key, value in row.items()} for row in safe.to_dict(orient="records")]


def get_json_body() -> dict[str, Any]:
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def build_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df[FEATURE_COLUMNS].copy()
    features["strength_weight"] = features["Tensile_Strength_MPa"] / features["Density_g_cm3"].replace(0, np.nan)
    features["cost_perf"] = features["Tensile_Strength_MPa"] / features["Cost_Index_1_10"].replace(0, np.nan)
    features["thermal_strength_ratio"] = features["Thermal_Conductivity_W_mK"] / features["Tensile_Strength_MPa"].replace(0, np.nan)
    return features[MODEL_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0)


def add_ml_predictions(df: pd.DataFrame, target_application: str | None = None) -> pd.DataFrame:
    enriched = df.copy()
    enriched["ML_Model_Active"] = bool(ml_ready)

    if not ml_ready or enriched.empty:
        enriched["ML_Predicted_Application"] = None
        enriched["ML_Confidence"] = None
        enriched["ML_Target_Probability"] = None
        return enriched

    try:
        features = build_ml_features(enriched)
        scaled = ml_scaler.transform(features)
        probabilities = ml_model.predict_proba(scaled)
    except (KeyError, TypeError, ValueError) as exc:
        enriched["ML_Model_Active"] = False
        enriched["ML_Predicted_Application"] = None
        enriched["ML_Confidence"] = None
        enriched["ML_Target_Probability"] = None
        enriched["ML_Error"] = f"Prediction failed: {exc}"
        return enriched
    best_indexes = probabilities.argmax(axis=1)
    encoded_predictions = np.asarray(ml_model.classes_)[best_indexes].astype(int)
    classes = ml_application_encoder.inverse_transform(encoded_predictions)

    enriched["ML_Predicted_Application"] = classes
    enriched["ML_Confidence"] = probabilities.max(axis=1)

    if target_application:
        try:
            encoded_target = int(ml_application_encoder.transform([target_application])[0])
            probability_index = list(ml_model.classes_).index(encoded_target)
            enriched["ML_Target_Probability"] = probabilities[:, probability_index]
        except ValueError:
            enriched["ML_Target_Probability"] = 0.0
    else:
        enriched["ML_Target_Probability"] = enriched["ML_Confidence"]

    return enriched


def rank_with_ml(df: pd.DataFrame, target_application: str | None = None) -> pd.DataFrame:
    ranked = add_ml_predictions(df, target_application)
    if ranked.empty:
        return ranked

    weighted = ranked["Recommendation_Score"]
    score_min = weighted.min()
    score_max = weighted.max()
    if score_max == score_min:
        ranked["Weighted_Score_Norm"] = 1.0
    else:
        ranked["Weighted_Score_Norm"] = (weighted - score_min) / (score_max - score_min)

    if ml_ready and target_application:
        ranked["ML_Recommendation_Score"] = (
            ranked["ML_Target_Probability"].fillna(0) * 70
            + ranked["Weighted_Score_Norm"].fillna(0) * 30
        )
        return ranked.sort_values("ML_Recommendation_Score", ascending=False)

    ranked["ML_Recommendation_Score"] = ranked["Weighted_Score_Norm"].fillna(0) * 100
    return ranked.sort_values("Recommendation_Score", ascending=False)


def material_profile(row: pd.Series | dict[str, Any]) -> dict[str, float]:
    get = row.get
    return {
        "strength": round(float(get("Tensile_Strength_MPa_norm", 0)), 2),
        "lowDensity": round(10 - float(get("Density_g_cm3_norm", 10)), 2),
        "corrosion": round(float(get("Corrosion_Resistance_0_10_norm", 0)), 2),
        "lowCost": round(10 - float(get("Cost_Index_1_10_norm", 10)), 2),
        "thermal": round(float(get("Thermal_Conductivity_W_mK_norm", 0)), 2),
        "hardness": round(float(get("Hardness_HB_norm", 0)), 2),
    }


def selection_reasons(row: pd.Series | dict[str, Any], weights: dict[str, int] | None = None) -> list[str]:
    weights = weights or {}
    reasons: list[str] = []

    if row.get("Tensile_Strength_MPa", 0) > 600:
        reasons.append("High tensile strength")
    elif weights.get("strength", 0) >= 5 and row.get("Tensile_Strength_MPa", 0) > 400:
        reasons.append("Balanced strength")

    if row.get("Density_g_cm3", 99) < 5:
        reasons.append("Low density")
    if row.get("Corrosion_Resistance_0_10", 0) >= 8:
        reasons.append("Strong corrosion resistance")
    if row.get("Cost_Index_1_10", 10) <= 4:
        reasons.append("Cost efficient")
    if row.get("Thermal_Conductivity_W_mK", 0) > 100:
        reasons.append("High thermal conductivity")
    if row.get("Machinability", 0) >= 8:
        reasons.append("Easy to machine")

    return reasons[:4] or ["Specialized property mix"]


def material_detail(material_name: str) -> dict[str, Any] | None:
    matches = recommender.materials_df[recommender.materials_df["Material"] == material_name]
    if matches.empty:
        return None

    row = matches.iloc[0]
    ml_row = add_ml_predictions(matches).iloc[0]
    return {
        "material": row["Material"],
        "alloyType": row.get("Alloy_Type"),
        "application": row.get("Primary_Application"),
        "ml": {
            "active": bool(ml_ready),
            "predictedApplication": clean_value(ml_row.get("ML_Predicted_Application")),
            "confidence": clean_value(ml_row.get("ML_Confidence")),
        },
        "properties": {
            "density": clean_value(row.get("Density_g_cm3")),
            "yieldStrength": clean_value(row.get("Yield_Strength_MPa")),
            "tensileStrength": clean_value(row.get("Tensile_Strength_MPa")),
            "hardness": clean_value(row.get("Hardness_HB")),
            "corrosion": clean_value(row.get("Corrosion_Resistance_0_10")),
            "thermal": clean_value(row.get("Thermal_Conductivity_W_mK")),
            "cost": clean_value(row.get("Cost_Index_1_10")),
            "meltingPoint": clean_value(row.get("Melting_Point_C")),
            "machinability": clean_value(row.get("Machinability")),
        },
        "reasons": selection_reasons(row),
    }


@app.get("/api/health")
def health() -> Any:
    return jsonify(
        {
            "ok": True,
            "dataset": DATASET_PATH.name,
            "mlReady": bool(ml_ready),
            "mlModel": "Random Forest" if ml_ready else None,
            "mlErrors": ml_load_errors,
        }
    )


@app.get("/api/meta")
def meta() -> Any:
    df = recommender.materials_df
    numeric_ranges = {}
    for key, config in SEARCH_COLUMNS.items():
        col = config["column"]
        numeric_ranges[key] = {
            **config,
            "min": clean_value(df[col].min()),
            "max": clean_value(df[col].max()),
        }

    return jsonify(
        {
            "applications": recommender.get_available_applications(),
            "materials": recommender.get_material_names(),
            "searchColumns": numeric_ranges,
            "stats": {
                "totalMaterials": int(len(df)),
                "alloyTypes": int(df["Alloy_Type"].nunique()),
                "applications": int(df["Primary_Application"].nunique()),
                "dataset": DATASET_PATH.name,
                "mlReady": bool(ml_ready),
                "mlModel": "Random Forest" if ml_ready else None,
            },
        }
    )


@app.post("/api/recommend")
def recommend() -> Any:
    data = get_json_body()
    weights = data.get("weights") if isinstance(data.get("weights"), dict) else {}
    application = data.get("application")
    application_filter = None if application in (None, "", "All") else str(application)
    ml_target_application = APPLICATION_ALIASES.get(application_filter, application_filter)

    top_n = int(data.get("topN", 5))
    results = recommender.get_weighted_recommendations(
        strength_weight=int(weights.get("strength", 7)),
        weight_weight=int(weights.get("weight", 7)),
        corrosion_weight=int(weights.get("corrosion", 5)),
        cost_weight=int(weights.get("cost", 3)),
        thermal_weight=int(weights.get("thermal", 3)),
        top_n=len(recommender.materials_df),
        application_filter=None,
    )
    results = rank_with_ml(results, ml_target_application).head(top_n)

    records = frame_to_records(results)
    for record in records:
        record["profile"] = material_profile(record)
        record["reasons"] = selection_reasons(record, weights)

    return jsonify(
        {
            "recommendations": records,
            "ml": {
                "active": bool(
                    ml_ready
                    and not results.empty
                    and results["ML_Predicted_Application"].notna().all()
                ),
                "model": "Random Forest" if ml_ready else None,
                "rankingMode": "ml_application_probability" if ml_ready and application_filter else "weighted_with_ml_annotations",
                "targetApplication": application_filter,
                "modelTargetApplication": ml_target_application,
                "errors": ml_load_errors
                + (
                    results["ML_Error"].dropna().astype(str).unique().tolist()
                    if "ML_Error" in results.columns
                    else []
                ),
            },
        }
    )


@app.get("/api/materials")
def materials() -> Any:
    search = request.args.get("search", "").strip()
    application = request.args.get("application", "All")
    limit = min(int(request.args.get("limit", 500)), 1000)

    df = recommender.materials_df.copy()
    if search:
        df = df[df["Material"].str.contains(search, case=False, na=False)]
    if application and application != "All":
        df = df[df["Primary_Application"].str.contains(application, case=False, na=False)]

    columns = [col for col in DISPLAY_COLUMNS if col in df.columns]
    enriched = add_ml_predictions(df.head(limit))
    columns = [col for col in DISPLAY_COLUMNS + ["ML_Predicted_Application", "ML_Confidence"] if col in enriched.columns]
    return jsonify({"materials": frame_to_records(enriched[columns]), "total": int(len(df))})


@app.get("/api/materials/<path:material_name>")
def material(material_name: str) -> Any:
    detail = material_detail(material_name)
    if detail is None:
        return jsonify({"error": "Material not found"}), 404
    return jsonify(detail)


@app.post("/api/search")
def advanced_search() -> Any:
    data = get_json_body()
    key = str(data.get("property", "tensile_strength"))
    config = SEARCH_COLUMNS.get(key, SEARCH_COLUMNS["tensile_strength"])
    column = config["column"]
    min_value = float(data.get("min", recommender.materials_df[column].min()))
    max_value = float(data.get("max", recommender.materials_df[column].max()))

    df = recommender.materials_df[
        (recommender.materials_df[column] >= min_value)
        & (recommender.materials_df[column] <= max_value)
    ].copy()
    df = df.sort_values(column, ascending=not bool(config["higherIsBetter"]))

    columns = [col for col in ["Material", "Alloy_Type", column, "Primary_Application"] if col in df.columns]
    return jsonify(
        {
            "materials": frame_to_records(df[columns]),
            "total": int(len(df)),
            "property": config,
        }
    )


@app.post("/api/compare")
def compare() -> Any:
    data = get_json_body()
    raw_materials = data.get("materials") if isinstance(data.get("materials"), list) else []
    names = list(dict.fromkeys(str(name) for name in raw_materials if name))[:3]
    comparison = recommender.compare_materials(names)
    if not comparison.empty:
        comparison = comparison.set_index("Material").reindex(names).dropna(how="all").reset_index()

    normalized = add_ml_predictions(comparison)
    for col in [
        "Tensile_Strength_MPa",
        "Density_g_cm3",
        "Corrosion_Resistance_0_10",
        "Cost_Index_1_10",
        "Thermal_Conductivity_W_mK",
        "Hardness_HB",
    ]:
        col_min = recommender.materials_df[col].min()
        col_max = recommender.materials_df[col].max()
        normalized[col + "_norm"] = 5 if col_max == col_min else (normalized[col] - col_min) / (col_max - col_min) * 10

    records = frame_to_records(normalized)
    for record in records:
        record["profile"] = material_profile(record)
        record["reasons"] = selection_reasons(record)

    return jsonify({"materials": records})


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react(path: str) -> Any:
    if FRONTEND_DIST.exists():
        target = FRONTEND_DIST / path
        if path and target.exists():
            return send_from_directory(FRONTEND_DIST, path)
        return send_from_directory(FRONTEND_DIST, "index.html")

    return jsonify(
        {
            "message": "React frontend is not built yet.",
            "development": "Run npm install and npm run dev inside ui/frontend.",
            "production": "Run npm run build inside ui/frontend, then restart python app.py.",
        }
    )


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 5050))
    debug = os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes"}
    app.run(host=host, port=port, debug=debug, use_reloader=False)
