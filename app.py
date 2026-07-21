import os
import gradio as gr
import joblib
import pandas as pd

# =====================================================
# Load Trained Model
# =====================================================
model = joblib.load("loan_prediction_model1.pkl")


# =====================================================
# Prediction Function
# =====================================================
def predict_loan(
    dependents,
    education,
    self_employed,
    income,
    loan_amount,
    loan_term,
    cibil,
    residential_assets,
    commercial_assets,
    luxury_assets,
    bank_assets
):

    input_data = pd.DataFrame({
        " no_of_dependents": [dependents],
        " education": [education],
        " self_employed": [self_employed],
        " income_annum": [income],
        " loan_amount": [loan_amount],
        " loan_term": [loan_term],
        " cibil_score": [cibil],
        " residential_assets_value": [residential_assets],
        " commercial_assets_value": [commercial_assets],
        " luxury_assets_value": [luxury_assets],
        " bank_asset_value": [bank_assets]
    })

    prediction = model.predict(input_data)[0]

    if prediction == 1 or str(prediction).lower() == "approved":
        return "✅ Loan Status : APPROVED"
    else:
        return "❌ Loan Status : REJECTED"


# =====================================================
# CSS
# =====================================================
css = """
.gradio-container{
background-image:url("https://images.unsplash.com/photo-1554224155-6726b3ff858f?q=80&w=2070&auto=format&fit=crop");
background-size:cover;
background-position:center;
background-repeat:no-repeat;
background-attachment:fixed;
}

.glass{
background:rgba(255,255,255,0.96);
padding:25px;
border-radius:20px;
box-shadow:0 0 25px rgba(0,0,0,0.25);
}

/* Make ALL markdown text black */
.gr-markdown,
.gr-markdown p,
.gr-markdown li,
.gr-markdown h1,
.gr-markdown h2,
.gr-markdown h3,
.gr-markdown strong{
color:black !important;
}

/* Button */
button{
font-size:18px !important;
}

footer{
visibility:hidden;
}
"""


# =====================================================
# Interface
# =====================================================
with gr.Blocks(
    css=css,
    title="Loan Approval Prediction"
) as demo:

    with gr.Column(elem_classes="glass"):

        gr.Markdown("""
# 🏦 Loan Approval Prediction System

Predict whether a loan application will be **Approved** or **Rejected**
using a trained **Random Forest Classifier**.
""")

        with gr.Row():

            # ================= Left =================

            with gr.Column(scale=2):

                dependents = gr.Number(
                    label="Number of Dependents",
                    value=2
                )

                education = gr.Dropdown(
                    ["Graduate", "Not Graduate"],
                    value="Graduate",
                    label="Education"
                )

                self_employed = gr.Dropdown(
                    ["Yes", "No"],
                    value="No",
                    label="Self Employed"
                )

                income = gr.Number(
                    label="Annual Income (₹)",
                    value=5000000
                )

                loan_amount = gr.Number(
                    label="Loan Amount (₹)",
                    value=1500000
                )

                loan_term = gr.Number(
                    label="Loan Term",
                    value=15
                )

                cibil = gr.Number(
                    label="CIBIL Score",
                    value=750
                )

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

                btn = gr.Button(
                    "Predict Loan Status",
                    variant="primary"
                )

                output = gr.Textbox(
                    label="Prediction"
                )

            # ================= Right =================

            with gr.Column(scale=1):

                gr.Markdown("""

# 👩‍💻 Developer Details

**Name:** Manya Singla

**College:**  
Panipat Institute of Engineering and Technology

---

# 📌 Project

Loan Approval Prediction using Random Forest Classifier

---

# 🛠 Technology Used

- Python
- Pandas
- Scikit-Learn
- Random Forest
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
- Residential Assets
- Commercial Assets
- Luxury Assets
- Bank Assets

---

# 🎯 Output

Predicts whether the loan will be

✅ Approved

or

❌ Rejected

""")

        btn.click(
            fn=predict_loan,
            inputs=[
                dependents,
                education,
                self_employed,
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


# =====================================================
# Launch
# =====================================================
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
