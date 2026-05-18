import json
from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("models/agriculture_suitability_model.pkl")
GRID_PATH = Path("uploads/konya_prediction_grid_15_features_no_landcover.csv")
GEOJSON_PATH = Path("static/predictions.geojson")
SUMMARY_PATH = Path("predictions/grid_prediction_summary.txt")


def extract_coordinates(geo_json):
    geometry = json.loads(geo_json)
    lon, lat = geometry["coordinates"]
    return lat, lon


def probability_class(probability):
    if probability >= 0.7:
        return "High"
    if probability >= 0.4:
        return "Medium"
    return "Low"


def main():
    if not GRID_PATH.exists():
        raise FileNotFoundError(
            f"Prediction grid not found: {GRID_PATH}. "
            "Export it from Google Earth Engine first."
        )

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    grid = pd.read_csv(GRID_PATH)
    missing = [column for column in feature_columns + [".geo"] if column not in grid.columns]
    if missing:
        raise ValueError(f"Missing required columns in prediction grid: {missing}")

    grid = grid.dropna(subset=feature_columns + [".geo"]).copy()
    features = grid[feature_columns]

    predictions = model.predict(features)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[:, 1]
    else:
        probabilities = predictions.astype(float)

    coords = grid[".geo"].apply(extract_coordinates)
    grid["Latitude"] = coords.apply(lambda value: value[0])
    grid["Longitude"] = coords.apply(lambda value: value[1])
    grid["prediction"] = predictions.astype(int)
    grid["probability"] = probabilities
    grid["risk_class"] = grid["probability"].apply(probability_class)

    features_geojson = []
    for _, row in grid.iterrows():
        properties = {
            "prediction": int(row["prediction"]),
            "probability": round(float(row["probability"]), 4),
            "risk_class": row["risk_class"],
        }
        for column in feature_columns:
            properties[column] = round(float(row[column]), 4)

        features_geojson.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [
                        round(float(row["Longitude"]), 6),
                        round(float(row["Latitude"]), 6),
                    ],
                },
                "properties": properties,
            }
        )

    feature_collection = {
        "type": "FeatureCollection",
        "name": "konya_agricultural_suitability_prediction_grid",
        "features": features_geojson,
    }

    GEOJSON_PATH.write_text(json.dumps(feature_collection), encoding="utf-8")

    class_counts = grid["risk_class"].value_counts().to_dict()
    lines = [
        "Prediction Grid Summary",
        f"Input grid rows after cleaning: {len(grid)}",
        f"Model: {bundle.get('best_model_name')}",
        f"Feature count: {len(feature_columns)}",
        "",
        "Suitability probability classes:",
        f"Low (< 0.40): {class_counts.get('Low', 0)}",
        f"Medium (0.40 - 0.70): {class_counts.get('Medium', 0)}",
        f"High (>= 0.70): {class_counts.get('High', 0)}",
        "",
        f"Mean probability: {grid['probability'].mean():.4f}",
        f"Minimum probability: {grid['probability'].min():.4f}",
        f"Maximum probability: {grid['probability'].max():.4f}",
        "",
        f"GeoJSON output: {GEOJSON_PATH}",
    ]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"Prediction grid rows: {len(grid)}")
    print(f"GeoJSON saved to: {GEOJSON_PATH}")
    print(f"Summary saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
