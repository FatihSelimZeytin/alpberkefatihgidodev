# Konya Agricultural Suitability Mapping

This project is a GIS and machine learning based agricultural suitability mapping system for Konya and its surroundings.

## Project Structure

- `app.py`: Flask web application and prediction endpoint.
- `models/`: Trained machine learning model saved as `.pkl`.
- `parameters/`: Extracted model inputs, labels, and training script.
- `predictions/`: Model comparison and evaluation results.
- `static/`: CSS and JavaScript files for the web map.
- `templates/`: HTML template for the web interface.
- `uploads/`: Google Earth Engine training dataset with 15 spatial features.

## Run

```bash
pip install -r requirements.txt
python app.py
```

Then open:

```text
http://127.0.0.1:5000/
```
or
https://alpberkefatihgidodev.onrender.com/
