import os
import gradio as gr
import joblib
import pandas as pd

# ==========================================
# Load Model
# ==========================================
model = joblib.load("loan_prediction_model.pkl")

# ==========================================
# Prediction Function
# ==========================================
def predict_loan(
    dependents,
    income,
    loan_amount,
    loan_term,
    cibil,
    residential_assets,
    commercial_assets,
    luxury_assets,
    bank_assets
):

    data = pd.DataFrame({
        " no_of_dependents":[dependents],
        " income_annum":[income],
        " loan_amount":[loan_amount],
        " loan_term":[loan_term],
        " cibil_score":[cibil],
        " residential_assets_value":[residential_assets],
        " commercial_assets_value":[commercial_assets],
        " luxury_assets_value":[luxury_assets],
        " bank_asset_value":[bank_assets]
    })

    prediction = model.predict(data)[0]

    if prediction == 1 or str(prediction).lower() == "approved":
        return "✅ Loan Status : APPROVED"
    else:
        return "❌ Loan Status : REJECTED"


# ==========================================
# CSS
# ==========================================
css = """
.gradio-container{
background-image:url("https://images.unsplash.com/photo-1554224155-6726b3ff858f?q=80&w=2070&auto=format&fit=crop");
background-size:cover;
background-position:center;
background-attachment:fixed;
}

.glass{
background:rgba(255,255,255,.95);
padding:25px;
border-radius:20px;
box-shadow:0 0 20px rgba(0,0,0,.25);
}

footer{
visibility: not hidden;
}
"""

# ==========================================
# Interface
# ==========================================
with gr.Blocks(css=css, title="Loan Approval Prediction") as demo:

    with gr.Column(elem_classes="glass"):

        gr.Markdown("""
# 🏦 Loan Approval Prediction System

Predict whether a loan application will be **Approved** or **Rejected** using a trained **Random Forest Classifier**.
""")

        with gr.Row():

            # Left Side
            with gr.Column(scale=2):

                dependents = gr.Number(label="Number of Dependents", value=2)

                income = gr.Number(label="Annual Income (₹)", value=5000000)

                loan_amount = gr.Number(label="Loan Amount (₹)", value=1500000)

                loan_term = gr.Number(label="Loan Term (Years)", value=15)

                cibil = gr.Number(label="CIBIL Score", value=750)

                residential = gr.Number(
                    label="Residential Assets Value (₹)",
                    value=4000000
                )

                commercial = gr.Number(
                    label="Commercial Assets Value (₹)",
                    value=2000000
                )

                luxury = gr.Number(
                    label="Luxury Assets Value (₹)",
                    value=1000000
                )

                bank = gr.Number(
                    label="Bank Assets Value (₹)",
                    value=3000000
                )

                btn = gr.Button("Predict Loan Status", variant="primary")

                output = gr.Textbox(
                    label="Prediction"
                )

            # Right Side
            with gr.Column(scale=1):

                gr.Markdown("""
## 👩‍💻 Developer

**Name:** Manya Singla

**College:**  
Panipat Institute of Engineering and Technology

---

## 📌 Project

Loan Approval Prediction using Random Forest Classifier

---

### Technology Stack

- Python
- Pandas
- Scikit-Learn
- Random Forest Classifier
- Joblib
- Gradio

---

### Input Features

- Dependents
- Annual Income
- Loan Amount
- Loan Term
- CIBIL Score
- Residential Assets
- Commercial Assets
- Luxury Assets
- Bank Assets

---

### Output

Predicts whether the loan application will be **Approved** or **Rejected**.
""")

        btn.click(
            predict_loan,
            inputs=[
                dependents,
                income,
                loan_amount,
                loan_term,
                cibil,
                residential,
                commercial,
                luxury,
                bank
            ],
            outputs=output
        )

# ==========================================
# Launch
# ==========================================
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
