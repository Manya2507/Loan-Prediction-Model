import os
import gradio as gr
import pandas as pd
import joblib

# Load model
model = joblib.load("titanic_prediction_model.pkl")

def predict(age, pclass, fare):
    data = pd.DataFrame({
        "age": [age],
        "pclass": [pclass],
        "fare": [fare]
    })

    pred = model.predict(data)[0]

    try:
        prob = model.predict_proba(data)[0][1]
        probability = f"\nSurvival Probability: {prob:.2%}"
    except Exception:
        probability = ""

    if pred == 1:
        return "✅ Passenger is likely to survive" + probability
    else:
        return "❌ Passenger is not likely to survive" + probability

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Age"),
        gr.Dropdown([1, 2, 3], value=3, label="Passenger Class"),
        gr.Number(label="Fare")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Titanic Survival Prediction"
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
