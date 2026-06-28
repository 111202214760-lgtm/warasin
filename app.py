import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# =========================
# LOAD MODEL
# =========================
model_path = "./model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# =========================
# PREDICT FUNCTION
# =========================
def predict(text):

    if text.strip() == "":
        return """
#  Upss...

Tulis dulu isi hatimu yaa 
"""

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)

    score, predicted_class = torch.max(probs, dim=1)

    confidence = round(float(score.item()) * 100, 2)

    # Label
    if predicted_class.item() == 1:
        label = " Terindikasi Depresi"
        mood = " Kamu nggak sendirian. Pelan-pelan yaa, semuanya bakal lewat."
    else:
        label = " Tidak Terindikasi Depresi"
        mood = " Bagus banget! Jangan lupa tetap jaga kesehatan mentalmu ya."

    return f"""
# {label}

## Confidence Score
# {confidence}%

<div style='text-align:center; font-size:18px; margin-top:20px;'>
{mood}
</div>

---
Jika merasa membutuhkan bantuan, hubungi psikolog atau orang terpercaya.
"""

# =========================
# CUSTOM CSS
# =========================
custom_css = """
body {
    background: linear-gradient(135deg, #0b1120, #1e293b, #334155) !important;
}

.gradio-container {
    font-family: 'Poppins', sans-serif !important;
}

textarea {
    border-radius: 20px !important;
    border: none !important;
    padding: 18px !important;
    font-size: 16px !important;
    box-shadow: 0 8px 25px rgba(0,0,0,0.08);
}

button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    border-radius: 18px !important;
    font-weight: bold !important;
    height: 55px !important;
    font-size: 16px !important;
    transition: 0.3s !important;
}

button:hover {
    transform: scale(1.03);
}

footer {
    visibility: hidden;
}

h1, h2, h3 {
    text-align: center;
}

.message-wrap {
    border-radius: 18px !important;
}
"""

# =========================
# UI
# =========================
with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="violet",
        secondary_hue="pink"
    ),
    css=custom_css
) as demo:

    gr.Markdown("""

# 🌿 Waras.in

<div style='text-align:center; font-size:18px;'>

> “gapapa capek, yang penting jangan nyerah yaa ”
""")

    with gr.Column():

        input_text = gr.Textbox(
            lines=6,
            placeholder="contoh: akhir-akhir ini aku capek banget sama hidup...",
            label=" Ceritain isi hatimu"
        )

        predict_btn = gr.Button(" Analisis Sekarang")

        output = gr.Markdown()

        predict_btn.click(
            fn=predict,
            inputs=input_text,
            outputs=output
        )

    gr.Markdown("""

""")

# =========================
# RUN
# =========================
demo.launch(share=True)