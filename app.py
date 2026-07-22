import os
import joblib
import pandas as pd
import gradio as gr

# ==========================================================
# Load Model
# ==========================================================

try:
    model = joblib.load("loan_prediction_model.pkl")
except Exception as e:
    print("Model Loading Error:", e)
    model = None

# ==========================================================
# Prediction Function
# ==========================================================

def predict_loan(
    no_of_dependents,
    education,
    self_employed,
    income_annum,
    loan_amount,
    loan_term,
    cibil_score,
    residential_assets_value,
    commercial_assets_value,
    luxury_assets_value,
    bank_asset_value,
):

    if model is None:
        return "❌ Model not loaded. Please check Loan_Prediction_Model.pkl"

    try:

        # Encode categorical values
        education = 1 if education == "Graduate" else 0
        self_employed = 1 if self_employed == "Yes" else 0

        input_df = pd.DataFrame([{
            " no_of_dependents": int(no_of_dependents),
            " education": education,
            " self_employed": self_employed,
            " income_annum": float(income_annum),
            " loan_amount": float(loan_amount),
            " loan_term": int(loan_term),
            " cibil_score": int(cibil_score),
            " residential_assets_value": float(residential_assets_value),
            " commercial_assets_value": float(commercial_assets_value),
            " luxury_assets_value": float(luxury_assets_value),
            " bank_asset_value": float(bank_asset_value),
        }])

        prediction = model.predict(input_df)[0]

        if prediction == 1 or str(prediction).strip().lower() == "approved":
            return """✅ LOAN APPROVED

Congratulations!

The Random Forest model predicts that the loan application is likely to be APPROVED.
"""

        else:
            return """❌ LOAN REJECTED

The Random Forest model predicts that the loan application is likely to be REJECTED.
"""

    except Exception as e:
        return f"Prediction Error:\n\n{e}"


# ==========================================================
# Description
# ==========================================================

DESCRIPTION = """
# 🏦 Loan Approval Prediction System

This application predicts whether a loan application will be *Approved* or *Rejected* using a trained *Random Forest Classifier*.

---

# 👩‍💻 Developer Details

*Name:* Manya

*College:* Panipat Institute of Engineering and Technology

---

# 📌 Project

Loan Approval Prediction using Machine Learning (Random Forest Classifier)

---

# 🛠️ Technology Stack

- Python
- Pandas
- Scikit-Learn
- Random Forest Classifier
- Joblib
- Gradio

---

# 📊 Input Features

- Number of Dependents
- Education
- Self Employed
- Annual Income
- Loan Amount
- Loan Term
- CIBIL Score
- Residential Assets Value
- Commercial Assets Value
- Luxury Assets Value
- Bank Asset Value

---

# 🎯 Output

Predicts whether the loan application is:

✅ Approved

or

❌ Rejected
"""

# ==========================================================
# Gradio Interface
# ==========================================================

demo = gr.Interface(
    fn=predict_loan,
    inputs=[
        gr.Number(label="Number of Dependents", value=2),
        gr.Dropdown(
            choices=["Graduate", "Not Graduate"],
            value="Graduate",
            label="Education"
        ),
        gr.Dropdown(
            choices=["Yes", "No"],
            value="No",
            label="Self Employed"
        ),
        gr.Number(label="Annual Income (₹)", value=5000000),
        gr.Number(label="Loan Amount (₹)", value=1500000),
        gr.Number(label="Loan Term", value=15),
        gr.Number(label="CIBIL Score", value=750),
        gr.Number(label="Residential Assets Value (₹)", value=4000000),
        gr.Number(label="Commercial Assets Value (₹)", value=2000000),
        gr.Number(label="Luxury Assets Value (₹)", value=1000000),
        gr.Number(label="Bank Assets Value (₹)", value=3000000),
    ],
    outputs=gr.Textbox(
        label="Prediction",
        lines=6
    ),
    title="🏦 Loan Approval Prediction",
    description=DESCRIPTION,
    theme=gr.themes.Soft(),
)

# ==========================================================
# Launch
# ==========================================================

if _name_ == "_main_":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
