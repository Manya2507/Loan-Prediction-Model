import os
import gradio as gr
import joblib
import pandas as pd

# ==========================
# Load Trained Model
# ==========================
model = joblib.load("loan_prediction_model.pkl")

# ==========================
# Prediction Function
# ==========================
def predict_loan(
    no_of_dependents,
    income_annum,
    loan_amount,
    loan_term,
    cibil_score,
    residential_assets_value,
    commercial_assets_value,
    luxury_assets_value,
    bank_asset_value,
):

    input_data = pd.DataFrame(
        [[
            no_of_dependents,
            income_annum,
            loan_amount,
            loan_term,
            cibil_score,
            residential_assets_value,
            commercial_assets_value,
            luxury_assets_value,
            bank_asset_value
        ]],
        columns=[
            " no_of_dependents",
            " income_annum",
            " loan_amount",
            " loan_term",
            " cibil_score",
            " residential_assets_value",
            " commercial_assets_value",
            " luxury_assets_value",
            " bank_asset_value"
        ]
    )

    prediction = model.predict(input_data)[0]

    return f"Predicted Loan Status : {prediction}"


# ==========================
# CSS
# ==========================

custom_css = """
.gradio-container{
background-image:url('https://images.unsplash.com/photo-1554224155-6726b3ff858f');
background-size:cover;
background-position:center;
background-attachment:fixed;
}

.glass{
background:rgba(255,255,255,0.95);
padding:20px;
border-radius:15px;
}

.gr-button{
background:#2563eb;
color:black;
}

"""


# ==========================
# Gradio Interface
# ==========================

with gr.Blocks(css=custom_css, title="Loan Approval Prediction System") as demo:

    with gr.Column(elem_classes="glass"):

        gr.Markdown(
            """
# 🏦 Loan Approval Prediction System

Predict whether a loan application is **Approved** or **Rejected**
using a Machine Learning Random Forest Classifier.
"""
        )

        with gr.Row():

            with gr.Column():

                dependents = gr.Number(label="Number of Dependents", value=0)

                income = gr.Number(label="Annual Income")

                loan_amount = gr.Number(label="Loan Amount")

                loan_term = gr.Number(label="Loan Term")

                cibil = gr.Number(label="CIBIL Score")

                residential = gr.Number(label="Residential Assets Value")

                commercial = gr.Number(label="Commercial Assets Value")

                luxury = gr.Number(label="Luxury Assets Value")

                bank = gr.Number(label="Bank Asset Value")

                predict_btn = gr.Button("Predict Loan Status")

                output = gr.Textbox(label="Prediction")

            

with gr.Column(scale=1):

    gr.Markdown("## 👩‍💻 About the Developer")

    gr.Markdown("""
**Name:** Manya Singla

**College:** Panipat Institute of Engineering and Technology

**Project:** AI Based Loan Approval Prediction System

**Machine Learning Model:** Random Forest Classifier

📧 Email: manyasingla25@gmail.com

📸 Instagram: @manya_singla_25

### 🛠️ Tools Used

- Python
- Gradio
- Scikit-Learn
- Pandas
- Joblib
""")

        predict_btn.click(
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
                bank,
            ],
            outputs=output,
        )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
