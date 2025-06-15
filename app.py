from flask import Flask, request, jsonify
from flask_cors import CORS
import CoolProp.CoolProp as CP
import pandas as pd
import numpy as np
import pickle

# Load pre-trained LGBM model and preprocessors
with open("compressor_model.pkl", "rb") as f:
    model = pickle.load(f)  # LightGBM wrapped model (e.g. MultiOutputRegressor)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

app = Flask(__name__)
CORS(app)

@app.route('/process', methods=['POST'])
def process_data():
    data = request.json

    try:
        model_input = data.get("model")
        refrigerant = data.get("refrigerant")
        evap_temp = float(data.get("evap_temp"))
        cond_temp = float(data.get("cond_temp"))
        superheat = float(data.get("superheat"))
        speed = float(data.get("speed"))

        Te_K = evap_temp + 273.15
        Tc_K = cond_temp + 273.15

        Psuction = CP.PropsSI("P", "T", Te_K, "Q", 1, refrigerant)
        Pdischarge = CP.PropsSI("P", "T", Tc_K, "Q", 0, refrigerant)

        if superheat == 0:
            h1 = CP.PropsSI("H", "P", Psuction, "Q", 1, refrigerant)
            s1 = CP.PropsSI("S", "P", Psuction, "Q", 1, refrigerant)
            Tsuction_K = CP.PropsSI("T", "P", Psuction, "Q", 1, refrigerant)
        else:
            Tsuction_K = Te_K + superheat
            h1 = CP.PropsSI("H", "T", Tsuction_K, "P", Psuction, refrigerant)
            s1 = CP.PropsSI("S", "T", Tsuction_K, "P", Psuction, refrigerant)

        h2 = CP.PropsSI("H", "P", Pdischarge, "S", s1, refrigerant)
        h3 = CP.PropsSI("H", "P", Pdischarge, "Q", 0, refrigerant)

        T2_C = CP.PropsSI("T", "P", Pdischarge, "H", h2, refrigerant) - 273.15

        model_encoded = le.transform([model_input.strip().upper()])[0]
        input_features = [model_encoded, evap_temp, Psuction / 1e5, cond_temp, Pdischarge / 1e5, speed]
        input_scaled = scaler.transform([input_features])

        # Predict using LGBM model
        predicted_output = model.predict(input_scaled)[0]  # Ensure this returns a list or array

        flow_rate = predicted_output[0] / 3600  # kg/s
        shaft_power = predicted_output[1]
        refrigeration_effect = (flow_rate * (h1 - h3)/1000) if h1 and h3 else 0
        isentropic_work = (flow_rate * (h2 - h1)/1000) if h1 and h2 else 0
        cop = refrigeration_effect / shaft_power if shaft_power else float('inf')

        result = {
            "Compressor Model": model_input,
            "Suction Pressure (bar abs)": round(Psuction / 1e5, 2),
            "Suction Temperature (°C)": round(Tsuction_K - 273.15, 2),
            "Discharge Pressure (bar abs)": round(Pdischarge / 1e5, 2),
            "Discharge Temperature (°C)": round(predicted_output[4], 2),
            "Speed (RPM)": round(speed, 2),
            "Volume Flow Rate (kg/hr)": round(predicted_output[0], 2),
            "Compressor Shaft Power (kW)": round(shaft_power, 2),
            "Adiabatic Efficiency (%)": round(predicted_output[2], 2),
            "Volumetric Efficiency (%)": round(predicted_output[3], 2),
            "Isentropic Work (kW)": round(isentropic_work, 2),
            "Refrigeration Effect (kW)": round(refrigeration_effect, 2),
            "Coefficient of Performance (COP)": round(cop, 2)
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run()
