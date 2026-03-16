
# AirSense — Forest Fire Risk Prediction for Northeast India

Live demo: https://huggingface.co/spaces/aaradhya233/forestfireprediction

## What it does
Predicts forest fire risk at pixel level across Northeast India by fusing 
real NASA FIRMS MODIS satellite fire detections with ERA5 atmospheric 
reanalysis variables (temperature, wind speed, humidity, precipitation).

## Method
Two versions built sequentially:

**V1:** Real NASA FIRMS fire coordinates with synthesised climatological 
weather features. Random Forest + XGBoost, AUC 0.9583.

**V2:** Same fire coordinates rebuilt with actual ERA5 reanalysis weather 
fetched per-coordinate from Open-Meteo archive API. AUC reflects real 
atmospheric data — more honest than V1.

## Key finding
The model captured monsoon-driven fire suppression without being told 
about seasonality. Fire probability dropped from 93.4% in March to 4.4% 
in June purely from ERA5 humidity and precipitation signals.

## Data sources
- Fire detections: NASA FIRMS MODIS (March 2026, real-time)
- Weather: Open-Meteo ERA5 reanalysis API

## Files
- `airsense_fire.ipynb` — full two-version pipeline notebook
- `app.py` — Gradio app (deployed on HuggingFace)
- `fused_fire_weather.csv` — ERA5-fused fire dataset
- `feature_distributions.png` — fire vs no-fire atmospheric conditions
- `prediction_vs_actual.png` — model predicted risk vs NASA FIRMS detections
- `seasonal_fire_risk.png` — fire risk across four seasons

## Author
Aaradhya Sahu 
