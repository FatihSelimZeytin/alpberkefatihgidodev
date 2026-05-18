from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATASET_PATH = Path("uploads/konya_agriculture_training_dataset_15_features.csv")
INPUTS_PATH = Path("parameters/Inputs.txt")
LABEL_PATH = Path("parameters/Label.txt")
MODEL_PATH = Path("models/agriculture_suitability_model.pkl")
METRICS_PATH = Path("predictions/model_metrics.txt")

FEATURE_COLUMNS = [
    "B2_Blue",
    "B3_Green",
    "B4_Red",
    "B8_NIR",
    "B11_SWIR1",
    "B12_SWIR2",
    "NDVI",
    "NDWI",
    "EVI",
    "SAVI",
    "Elevation",
    "Slope",
    "Aspect",
    "Annual_Rainfall",
    "LST_Celsius",
]


def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    missing = [col for col in FEATURE_COLUMNS + ["Label"] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[FEATURE_COLUMNS + ["Label"]].dropna()
    X = df[FEATURE_COLUMNS]
    y = df["Label"].astype(int)
    return df, X, y


def export_inputs_and_labels(X, y):
    X.to_csv(INPUTS_PATH, sep="\t", index=False)
    y.to_csv(LABEL_PATH, sep="\t", index=False, header=["Label"])


def build_experiments():
    return [
        {
            "classifier": "Logistic Regression",
            "stage": "Baseline",
            "parameters": "max_iter=1000, C=1.0",
            "model": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
                ]
            ),
        },
        {
            "classifier": "Logistic Regression",
            "stage": "Tuned",
            "parameters": "max_iter=2000, C=0.5, class_weight=balanced",
            "model": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=2000,
                            C=0.5,
                            class_weight="balanced",
                            random_state=42,
                        ),
                    ),
                ]
            ),
        },
        {
            "classifier": "Random Forest",
            "stage": "Baseline",
            "parameters": "n_estimators=100, max_depth=None",
            "model": RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                random_state=42,
            ),
        },
        {
            "classifier": "Random Forest",
            "stage": "Tuned",
            "parameters": "n_estimators=300, max_depth=None, min_samples_split=4, class_weight=balanced",
            "model": RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=4,
                random_state=42,
                class_weight="balanced",
            ),
        },
        {
            "classifier": "Gradient Boosting",
            "stage": "Baseline",
            "parameters": "n_estimators=100, learning_rate=0.1, max_depth=3",
            "model": GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42,
            ),
        },
        {
            "classifier": "Gradient Boosting",
            "stage": "Tuned",
            "parameters": "n_estimators=200, learning_rate=0.05, max_depth=3",
            "model": GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=42,
            ),
        },
    ]


def main():
    df, X, y = load_dataset()
    export_inputs_and_labels(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    results = []
    trained_models = {}

    for experiment in build_experiments():
        model = experiment["model"]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, digits=4)

        results.append(
            {
                "name": f"{experiment['classifier']} ({experiment['stage']})",
                "classifier": experiment["classifier"],
                "stage": experiment["stage"],
                "parameters": experiment["parameters"],
                "accuracy": accuracy,
                "f1": f1,
                "confusion_matrix": cm,
                "classification_report": report,
            }
        )
        trained_models[f"{experiment['classifier']} ({experiment['stage']})"] = model

    best = max(results, key=lambda item: item["f1"])
    best_model = trained_models[best["name"]]

    joblib.dump(
        {
            "model": best_model,
            "feature_columns": FEATURE_COLUMNS,
            "target_column": "Label",
            "best_model_name": best["name"],
            "best_classifier": best["classifier"],
            "best_stage": best["stage"],
            "best_parameters": best["parameters"],
            "metrics": results,
        },
        MODEL_PATH,
    )

    lines = [
        "Agricultural Suitability Model - Konya and Surroundings",
        f"Dataset rows after cleaning: {len(df)}",
        f"Features: {len(FEATURE_COLUMNS)}",
        "",
        "Feature columns:",
        *[f"- {col}" for col in FEATURE_COLUMNS],
        "",
        "Model development, training, testing and comparison:",
        "Each classifier was trained first with baseline parameters and then trained again with different tuned parameters.",
        "",
        "Summary table:",
        "Classifier\tTraining Stage\tParameters\tAccuracy\tF1 Score",
        *[
            (
                f"{result['classifier']}\t{result['stage']}\t{result['parameters']}"
                f"\t{result['accuracy']:.4f}\t{result['f1']:.4f}"
            )
            for result in results
        ],
        "",
        "Detailed evaluation:",
    ]

    for result in results:
        lines.extend(
            [
                f"",
                f"Model: {result['name']}",
                f"Classifier: {result['classifier']}",
                f"Training Stage: {result['stage']}",
                f"Parameters: {result['parameters']}",
                f"Accuracy: {result['accuracy']:.4f}",
                f"F1 Score: {result['f1']:.4f}",
                "Confusion Matrix:",
                str(result["confusion_matrix"]),
                "Classification Report:",
                result["classification_report"],
            ]
        )

    lines.extend(
        [
            "",
            f"Best model: {best['name']}",
            f"Best parameters: {best['parameters']}",
        ]
    )
    METRICS_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"Inputs saved to: {INPUTS_PATH}")
    print(f"Labels saved to: {LABEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Best model saved to: {MODEL_PATH}")
    print(f"Best model: {best['name']} | F1: {best['f1']:.4f}")


if __name__ == "__main__":
    main()
