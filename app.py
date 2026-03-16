
import gradio as gr
import pickle
import numpy as np

# Load model
with open('fire_rf_model_v2.pkl', 'rb') as f:
    model = pickle.load(f)

def predict_fire_risk(temperature, precipitation, wind_speed, 
                       ndvi, humidity, aod, elevation, 
                       drought_index, land_cover):
    
    land_cover_map = {
        'Forest': 0, 'Grassland': 1, 
        'Agricultural': 2, 'Shrubland': 3
    }
    
    features = np.array([[
        temperature, precipitation, wind_speed,
        ndvi, aod, elevation, humidity,
        drought_index, land_cover_map[land_cover]
    ]])
    
    prob = model.predict_proba(features)[0][1]
    risk_pct = prob * 100
    
    if prob >= 0.80:
        level = "🔴 CRITICAL FIRE RISK"
        advice = "Extreme danger. Avoid all forest areas. Alert local fire authorities immediately."
        color = "#cc0000"
    elif prob >= 0.60:
        level = "🟠 HIGH FIRE RISK"
        advice = "High danger conditions. Restrict agricultural burning. Monitor actively."
        color = "#ff6600"
    elif prob >= 0.40:
        level = "🟡 MODERATE FIRE RISK"
        advice = "Elevated risk. Avoid open burning. Stay alert for fire reports."
        color = "#ffaa00"
    elif prob >= 0.20:
        level = "🟢 LOW FIRE RISK"
        advice = "Manageable conditions. Standard fire safety protocols apply."
        color = "#00aa44"
    else:
        level = "✅ MINIMAL FIRE RISK"
        advice = "Safe conditions. Likely monsoon or heavy rain suppressing fire activity."
        color = "#0066cc"
    
    # Build output
    result = f"""
## {level}

**Fire Probability: {risk_pct:.1f}%**

{advice}

---
### Input Summary
| Parameter | Value |
|---|---|
| Temperature | {temperature}°C |
| Precipitation | {precipitation} mm |
| Wind Speed | {wind_speed} m/s |
| NDVI (Vegetation Health) | {ndvi:.2f} |
| Humidity | {humidity}% |
| AOD (Aerosol/Smoke) | {aod:.2f} |
| Elevation | {elevation} m |
| Drought Index | {drought_index:.1f} |
| Land Cover | {land_cover} |

---
*Model: Random Forest | Trained on 788 real NASA FIRMS MODIS detections*  
*Northeast India | March 2026 | AUC: 0.9583*
"""
    return result

# Build the interface
with gr.Blocks(theme=gr.themes.Base(), title="AirSense — NE India Fire Risk") as app:
    
    gr.Markdown("""
    # 🔥 AirSense — Forest Fire Risk Predictor
    ### Northeast India | Trained on NASA FIRMS MODIS Satellite Data
    *Enter current weather and vegetation conditions to predict fire risk*
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🌡️ Weather Conditions")
            temperature = gr.Slider(10, 45, value=32, 
                                     label="Temperature (°C)")
            precipitation = gr.Slider(0, 500, value=25, 
                                       label="Monthly Precipitation (mm)")
            wind_speed = gr.Slider(0, 15, value=4.5, 
                                    label="Wind Speed (m/s)")
            humidity = gr.Slider(20, 100, value=52, 
                                  label="Relative Humidity (%)")
            
        with gr.Column():
            gr.Markdown("### 🌿 Vegetation & Environment")
            ndvi = gr.Slider(0.0, 1.0, value=0.40, step=0.01,
                              label="NDVI — Vegetation Health (0=bare, 1=dense)")
            aod = gr.Slider(0.0, 1.0, value=0.35, step=0.01,
                             label="AOD — Aerosol/Smoke Density")
            elevation = gr.Slider(0, 3000, value=350, 
                                   label="Elevation (meters)")
            drought_index = gr.Slider(-3.0, 3.0, value=0.5, step=0.1,
                                       label="Drought Index (-3=wet, +3=severe drought)")
            land_cover = gr.Dropdown(
                choices=['Forest', 'Grassland', 'Agricultural', 'Shrubland'],
                value='Forest',
                label="Land Cover Type"
            )
    
    predict_btn = gr.Button("🔥 Predict Fire Risk", variant="primary", size="lg")
    output = gr.Markdown()
    
    predict_btn.click(
        fn=predict_fire_risk,
        inputs=[temperature, precipitation, wind_speed, ndvi,
                humidity, aod, elevation, drought_index, land_cover],
        outputs=output
    )
    
    gr.Markdown("""
    ---
    Fire occurrence data: NASA FIRMS MODIS satellite, March 2026  
    Weather features: ERA5 reanalysis via Open-Meteo API  
    Model: Random Forest, presence-absence framework, AUC 0.9583  
    Built by Aaradhya Sahu 
    """)

app.launch()