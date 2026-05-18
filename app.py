import json
import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS


MODEL_PATH = Path("models/agriculture_suitability_model.pkl")
DATASET_PATH = Path("uploads/konya_agriculture_training_dataset_15_features.csv")
PREDICTION_GRID_PATH = Path("static/predictions.geojson")
STUDY_AREA_BOUNDS = {
    "min_lon": 31.0,
    "min_lat": 36.8,
    "max_lon": 34.8,
    "max_lat": 39.3,
}

app = Flask(__name__)
CORS(app)

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
feature_columns = bundle["feature_columns"]

training_points = pd.read_csv(DATASET_PATH)
prediction_points = None


def _extract_coordinates(geo_json):
    geometry = json.loads(geo_json)
    lon, lat = geometry["coordinates"]
    return lat, lon


training_points = pd.read_csv("uploads/konya_agriculture_training_dataset_15_features.csv")

coords = training_points[["Latitude", "Longitude"]]

class_feature_means = (
    training_points[feature_columns + ["Label"]]
    .dropna()
    .groupby("Label")[feature_columns]
    .mean()
)

if hasattr(model, "feature_importances_"):
    feature_importance = dict(zip(feature_columns, model.feature_importances_))
elif hasattr(model, "named_steps") and hasattr(model.named_steps.get("model"), "feature_importances_"):
    feature_importance = dict(zip(feature_columns, model.named_steps["model"].feature_importances_))
else:
    feature_importance = {column: 1 / len(feature_columns) for column in feature_columns}

FEATURE_DISPLAY_NAMES = {
    "B2_Blue": "B2 Blue",
    "B3_Green": "B3 Green",
    "B4_Red": "B4 Red",
    "B8_NIR": "B8 NIR",
    "B11_SWIR1": "B11 SWIR1",
    "B12_SWIR2": "B12 SWIR2",
    "NDVI": "NDVI",
    "NDWI": "NDWI",
    "EVI": "EVI",
    "SAVI": "SAVI",
    "Elevation": "Elevation",
    "Slope": "Slope",
    "Aspect": "Aspect",
    "Annual_Rainfall": "Annual Rainfall",
    "LST_Celsius": "LST Celsius",
}

if PREDICTION_GRID_PATH.exists():
    prediction_geojson = json.loads(PREDICTION_GRID_PATH.read_text(encoding="utf-8"))
    prediction_rows = []
    for feature in prediction_geojson.get("features", []):
        lon, lat = feature["geometry"]["coordinates"]
        row = {
            "Latitude": lat,
            "Longitude": lon,
            **feature.get("properties", {}),
        }
        prediction_rows.append(row)
    if prediction_rows:
        prediction_points = pd.DataFrame(prediction_rows)


@app.route("/")
def home():
    return render_template("index.html")


def nearest_training_point(lat, lon):
    distances = (
        (training_points["Latitude"] - lat) ** 2
        + (training_points["Longitude"] - lon) ** 2
    )
    return training_points.loc[distances.idxmin()]


def nearest_prediction_point(lat, lon):
    if prediction_points is None:
        return None

    distances = (
        (prediction_points["Latitude"] - lat) ** 2
        + (prediction_points["Longitude"] - lon) ** 2
    )
    return prediction_points.loc[distances.idxmin()]


def is_inside_study_area(lat, lon):
    return (
        STUDY_AREA_BOUNDS["min_lat"] <= lat <= STUDY_AREA_BOUNDS["max_lat"]
        and STUDY_AREA_BOUNDS["min_lon"] <= lon <= STUDY_AREA_BOUNDS["max_lon"]
    )


def explain_prediction(sample_values, prediction, probability):
    predicted_label = int(prediction)
    probability_text = f"{round(float(probability or 0) * 100)}%"
    label_text = "suitable" if predicted_label == 1 else "not suitable"

    value = {column: float(sample_values[column]) for column in feature_columns}
    observations = []

    def add_observation(kind, feature, title, text, strength):
        observations.append(
            {
                "kind": kind,
                "feature": feature,
                "title": title,
                "text": text,
                "strength": strength * (1 + float(feature_importance.get(feature, 0))),
            }
        )

    ndvi = value.get("NDVI", 0)
    savi = value.get("SAVI", 0)
    evi = value.get("EVI", 0)
    elevation = value.get("Elevation", 0)
    slope = value.get("Slope", 0)
    swir1 = value.get("B11_SWIR1", 0)
    swir2 = value.get("B12_SWIR2", 0)
    rainfall = value.get("Annual_Rainfall", 0)
    lst = value.get("LST_Celsius", 0)

    if ndvi < 0.25:
        add_observation(
            "negative",
            "NDVI",
            "Low vegetation signal",
            f"NDVI is {ndvi:.2f}, which indicates weak vegetation cover or bare/sparse surface conditions.",
            4,
        )
    elif ndvi < 0.4:
        add_observation(
            "negative",
            "NDVI",
            "Moderate-low vegetation signal",
            f"NDVI is {ndvi:.2f}. This is not strong enough to indicate dense agricultural vegetation.",
            3,
        )
    elif ndvi >= 0.55:
        add_observation(
            "positive",
            "NDVI",
            "Strong vegetation signal",
            f"NDVI is {ndvi:.2f}, which suggests dense and healthy vegetation cover.",
            4,
        )

    if savi < 0.35:
        add_observation(
            "negative",
            "SAVI",
            "Weak vegetation after soil adjustment",
            f"SAVI is {savi:.2f}, supporting the interpretation of sparse vegetation or strong soil background effect.",
            3,
        )
    elif savi < 0.55:
        add_observation(
            "negative",
            "SAVI",
            "Limited vegetation after soil adjustment",
            f"SAVI is {savi:.2f}, which suggests only limited vegetation strength after reducing soil background effects.",
            2,
        )
    elif savi >= 0.65:
        add_observation(
            "positive",
            "SAVI",
            "Healthy vegetation after soil adjustment",
            f"SAVI is {savi:.2f}, which supports the presence of vegetation even after reducing soil background effects.",
            3,
        )

    if evi < 0.5:
        add_observation(
            "negative",
            "EVI",
            "Limited vegetation vigor",
            f"EVI is {evi:.2f}, which does not indicate strong vegetation vigor at this location.",
            2,
        )
    elif evi < 0.85:
        add_observation(
            "negative",
            "EVI",
            "Moderate vegetation vigor",
            f"EVI is {evi:.2f}. This indicates some vegetation response, but not a strong agricultural vegetation signal.",
            1.5,
        )
    elif evi >= 1.0:
        add_observation(
            "positive",
            "EVI",
            "High vegetation vigor",
            f"EVI is {evi:.2f}, indicating a strong vegetation response in the satellite data.",
            2,
        )

    if elevation >= 1600:
        add_observation(
            "negative",
            "Elevation",
            "High elevation",
            f"Elevation is {elevation:.0f} m. High-altitude areas are often less favorable for intensive agriculture due to cooler conditions and terrain constraints.",
            4,
        )
    elif elevation >= 1450:
        add_observation(
            "negative",
            "Elevation",
            "Relatively high elevation",
            f"Elevation is {elevation:.0f} m, which is relatively high for plain agricultural conditions around Konya.",
            3,
        )
    elif 850 <= elevation <= 1300:
        add_observation(
            "positive",
            "Elevation",
            "Favorable plain elevation",
            f"Elevation is {elevation:.0f} m, which is within a more typical range for agricultural plains around Konya.",
            3,
        )

    if slope >= 8:
        add_observation(
            "negative",
            "Slope",
            "Steep terrain",
            f"Slope is {slope:.2f} degrees. Steeper terrain can increase erosion risk and make irrigation or machinery use harder.",
            3,
        )
    elif slope >= 6:
        add_observation(
            "negative",
            "Slope",
            "Moderately sloped terrain",
            f"Slope is {slope:.2f} degrees, which is less favorable than flat terrain for irrigation, machinery, and erosion control.",
            2.5,
        )
    elif slope <= 4:
        add_observation(
            "positive",
            "Slope",
            "Gentle terrain",
            f"Slope is {slope:.2f} degrees, which is favorable for irrigation, mechanization, and lower erosion risk.",
            3,
        )

    if swir1 >= 3800 or swir2 >= 3000:
        add_observation(
            "negative",
            "B11_SWIR1",
            "Dry surface signal",
            f"SWIR reflectance is high (B11: {swir1:.0f}, B12: {swir2:.0f}), which may indicate dry soil, low moisture, or stressed surface conditions.",
            3,
        )
    elif swir1 >= 2800 or swir2 >= 2200:
        add_observation(
            "negative",
            "B11_SWIR1",
            "Moderate dry-surface signal",
            f"SWIR reflectance is noticeable (B11: {swir1:.0f}, B12: {swir2:.0f}), which can still indicate relatively dry or stressed surface conditions.",
            2.5,
        )
    elif swir1 <= 2600 and swir2 <= 2200:
        add_observation(
            "positive",
            "B11_SWIR1",
            "Lower dry-surface signal",
            f"SWIR reflectance is relatively low (B11: {swir1:.0f}, B12: {swir2:.0f}), suggesting less dry or less stressed surface conditions.",
            2,
        )

    if rainfall < 320:
        add_observation(
            "negative",
            "Annual_Rainfall",
            "Low rainfall",
            f"Annual rainfall is {rainfall:.2f} mm, which can limit agricultural suitability in a semi-arid region.",
            3,
        )
    elif rainfall >= 450:
        add_observation(
            "positive",
            "Annual_Rainfall",
            "Higher rainfall availability",
            f"Annual rainfall is {rainfall:.2f} mm, providing better moisture availability for agricultural use.",
            2,
        )

    if lst >= 35:
        add_observation(
            "negative",
            "LST_Celsius",
            "High surface temperature",
            f"Land surface temperature is {lst:.2f} C, which may indicate heat stress or dry surface conditions.",
            2,
        )
    elif 18 <= lst <= 32:
        add_observation(
            "positive",
            "LST_Celsius",
            "Moderate surface temperature",
            f"Land surface temperature is {lst:.2f} C, which is more moderate for vegetation growth than very hot surfaces.",
            2,
        )

    if predicted_label == 1:
        selected = [item for item in observations if item["kind"] == "positive"]
    else:
        selected = [item for item in observations if item["kind"] == "negative"]

    if not selected:
        if predicted_label == 0:
            limiting_parts = []
            if ndvi < 0.55:
                limiting_parts.append(f"NDVI is moderate rather than strong ({ndvi:.2f})")
            if slope > 2:
                limiting_parts.append(f"slope is not completely flat ({slope:.2f} degrees)")
            if swir1 > 2800 or swir2 > 1900:
                limiting_parts.append(
                    f"SWIR reflectance is still noticeable (B11: {swir1:.0f}, B12: {swir2:.0f})"
                )
            if elevation > 1200:
                limiting_parts.append(f"elevation is relatively high ({elevation:.0f} m)")

            limiting_text = "; ".join(limiting_parts[:3])
            if not limiting_text:
                limiting_text = (
                    "the model did not find enough strong positive evidence across the combined satellite, "
                    "terrain, and climate features"
                )

            selected.append(
                {
                    "kind": "negative",
                    "feature": "combined",
                    "title": "Mixed agricultural signal",
                    "text": (
                        f"This location has some favorable values, but the model still assigns a low suitability "
                        f"probability because {limiting_text}."
                    ),
                    "strength": 1,
                }
            )
        else:
            supporting_parts = []
            if ndvi >= 0.4:
                supporting_parts.append(f"NDVI shows usable vegetation signal ({ndvi:.2f})")
            if savi >= 0.55:
                supporting_parts.append(f"SAVI supports vegetation presence ({savi:.2f})")
            if slope <= 6:
                supporting_parts.append(f"slope is manageable ({slope:.2f} degrees)")
            if rainfall >= 350:
                supporting_parts.append(f"rainfall is not extremely low ({rainfall:.2f} mm)")

            supporting_text = "; ".join(supporting_parts[:3])
            if not supporting_text:
                supporting_text = (
                    "the combined feature pattern is closer to suitable grid cells than unsuitable ones"
                )

            selected.append(
                {
                    "kind": "positive",
                    "feature": "combined",
                    "title": "Combined favorable pattern",
                    "text": (
                        f"The model classifies this location as suitable because {supporting_text}."
                    ),
                    "strength": 1,
                }
            )

    if predicted_label == 0 and float(probability or 0) < 0.15:
        suitable_means = class_feature_means.loc[1] if 1 in class_feature_means.index else None
        existing_titles = {item["title"] for item in selected}

        def add_low_probability_reason(feature, title, text, strength):
            if title in existing_titles:
                return
            selected.append(
                {
                    "kind": "negative",
                    "feature": feature,
                    "title": title,
                    "text": text,
                    "strength": strength * (1 + float(feature_importance.get(feature, 0))),
                }
            )
            existing_titles.add(title)

        if suitable_means is not None:
            suitable_nir = float(suitable_means["B8_NIR"])
            if value["B8_NIR"] < suitable_nir * 0.85:
                add_low_probability_reason(
                    "B8_NIR",
                    "Lower near-infrared response",
                    (
                        f"B8 NIR is {value['B8_NIR']:.0f}, clearly below the suitable training average "
                        f"({suitable_nir:.0f}). This can mean the canopy/vegetation response is weaker than "
                        "typical suitable agricultural grid cells."
                    ),
                    4,
                )

            suitable_elevation = float(suitable_means["Elevation"])
            if value["Elevation"] > suitable_elevation + 120:
                add_low_probability_reason(
                    "Elevation",
                    "Higher than typical suitable elevation",
                    (
                        f"Elevation is {value['Elevation']:.0f} m, higher than the suitable training average "
                        f"({suitable_elevation:.0f} m). This can make the location less similar to the lower "
                        "plain-like agricultural areas learned by the model."
                    ),
                    3.5,
                )

            suitable_slope = float(suitable_means["Slope"])
            if value["Slope"] > suitable_slope + 2:
                add_low_probability_reason(
                    "Slope",
                    "Less flat than typical suitable terrain",
                    (
                        f"Slope is {value['Slope']:.2f} degrees, higher than the suitable training average "
                        f"({suitable_slope:.2f} degrees). This reduces similarity to flat agricultural grid cells."
                    ),
                    3,
                )

        if 0.3 <= ndvi < 0.55:
            add_low_probability_reason(
                "NDVI",
                "Vegetation is present but not strong",
                (
                    f"NDVI is {ndvi:.2f}. This shows vegetation presence, but it is not strong enough to offset "
                    "the other limiting signals in the model."
                ),
                2.5,
            )

    if len(selected) > 1:
        selected = [item for item in selected if item["feature"] != "combined"]

    selected = sorted(selected, key=lambda item: item["strength"], reverse=True)[:5]
    for item in selected:
        item.pop("strength", None)

    summary = (
        f"The model classified this location as {label_text} with {probability_text} suitability probability. "
        "This interpretation is based on the agricultural meaning of the selected location's feature values."
    )

    return {
        "summary": summary,
        "reasons": selected,
    }


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json() or {}

    try:
        lat = float(data["Latitude"])
        lon = float(data["Longitude"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Latitude and Longitude are required."}), 400

    if not is_inside_study_area(lat, lon):
        return jsonify(
            {
                "error": (
                    "Selected point is outside the study area. "
                    "Please select a point within Konya and its surroundings."
                ),
                "study_area_bounds": STUDY_AREA_BOUNDS,
            }
        ), 400

    nearest_grid = nearest_prediction_point(lat, lon)
    if nearest_grid is not None:
        sample_values = {column: nearest_grid[column] for column in feature_columns}
        prediction = int(nearest_grid["prediction"])
        probability = round(float(nearest_grid["probability"]), 4)
        return jsonify(
            {
                "prediction": prediction,
                "probability": probability,
                "best_model": bundle.get("best_model_name"),
                "nearest_point": {
                    "latitude": round(float(nearest_grid["Latitude"]), 6),
                    "longitude": round(float(nearest_grid["Longitude"]), 6),
                    "label": None,
                },
                "features": {
                    column: round(float(sample_values[column]), 4)
                    for column in feature_columns
                },
                "explanation": explain_prediction(sample_values, prediction, probability),
                "source": "prediction_grid",
            }
        )

    nearest = nearest_training_point(lat, lon)
    sample_values = {column: nearest[column] for column in feature_columns}

    # Allows future UI scenario controls to override feature values.
    for column in feature_columns:
        if column in data and data[column] not in ("", None):
            sample_values[column] = float(data[column])

    sample = pd.DataFrame([sample_values], columns=feature_columns)
    prediction = int(model.predict(sample)[0])

    probability = None
    if hasattr(model, "predict_proba"):
        probability = round(float(model.predict_proba(sample)[0][1]), 4)

    return jsonify(
        {
            "prediction": prediction,
            "probability": probability,
            "best_model": bundle.get("best_model_name"),
            "nearest_point": {
                "latitude": round(float(nearest["Latitude"]), 6),
                "longitude": round(float(nearest["Longitude"]), 6),
                "label": int(nearest["Label"]),
            },
            "features": {
                column: round(float(sample_values[column]), 4)
                for column in feature_columns
            },
            "explanation": explain_prediction(sample_values, prediction, probability),
            "source": "training_sample_fallback",
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
