import gradio as gr
import joblib
import re
import numpy as np
import nltk
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download dependencies
nltk.download('stopwords')
nltk.download('wordnet')

# ---------------------- LOAD ARTIFACTS ----------------------
LSTM_MODEL_PATH = "models/lstm_best_model.h5"
TOKENIZER_PATH = "models/tokenizer.pkl"
LABEL_ENCODER_PATH = "models/label_encoder_lstm.pkl"

lstm_model = load_model(LSTM_MODEL_PATH)
tokenizer = joblib.load(TOKENIZER_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

# ---------------------- CLEAN TEXT ----------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(words)

# ---------------------- PREDICTION ----------------------
def predict_news_category(news_text):
    if not news_text.strip():
        return "<p style='color:#666;'>⚠️ Please enter some text.</p>"

    cleaned = clean_text(news_text)
    seq = tokenizer.texts_to_sequences([cleaned])

    max_len = lstm_model.input_shape[1]
    padded = pad_sequences(seq, maxlen=max_len, padding="post")

    probs = lstm_model.predict(padded)[0]
    idx = np.argmax(probs)
    category = label_encoder.inverse_transform([idx])[0]

    return f"""
    <div style='text-align:center; padding:15px;'>
        <p style='font-size:18px; color:#333; margin-bottom:5px;'>Predicted Category</p>
        <p style='font-size:30px; font-weight:700; color:#007bff; margin-top:0px;'>
            {category.upper()}
        </p>
    </div>
    """

# ---------------------- NEWS BACKGROUND CSS ----------------------
news_css = """
body {
    background: url('https://images.pexels.com/photos/518543/pexels-photo-518543.jpeg') 
                no-repeat center center fixed !important;
    background-size: cover !important;
    font-family: 'Segoe UI', sans-serif;
}

.gradio-container {
    max-width: 750px !important;
    margin: 60px auto;
    padding: 30px 40px;
    border-radius: 14px;
    background: rgba(255,255,255,0.90) !important;
    backdrop-filter: blur(6px);
    box-shadow: 0 0 20px rgba(0,0,0,0.25);
}

h1 {
    text-align: center;
    color: #007bff;
    font-size: 28px;
    font-weight: 700;
}

textarea {
    background-color: #ffffff !important;
    color: #000 !important;
    border: 1px solid #ccc !important;
    border-radius: 8px !important;
    font-size: 15px !important;
}

button {
    background-color: #007bff !important;
    color: #fff !important;
    border-radius: 6px !important;
    padding: 10px 16px !important;
    font-weight: 600;
}

button:hover {
    background-color: #0066d6 !important;
}
"""

# ---------------------- UI ----------------------
with gr.Blocks(css=news_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("<h1>🧠 HuffPost News Classifier (BiLSTM + GloVe)</h1>")
    gr.Markdown("Paste any news headline or article below.")

    news_input = gr.Textbox(
        lines=8,
        placeholder="Type or paste news content here...",
        label="📰 Enter News Text"
    )

    predict_button = gr.Button("Predict Category")
    output = gr.HTML(label="Prediction")

    predict_button.click(fn=predict_news_category, inputs=news_input, outputs=output)

# ---------------------- LAUNCH ----------------------
if __name__ == "__main__":
    demo.launch(share=True)


