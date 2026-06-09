import gradio as gr
import joblib
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download dependencies
nltk.download('stopwords')
nltk.download('wordnet')

# Load model artifacts
model = joblib.load("best_lr.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Preprocessing setup
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return ' '.join(words)

# Prediction Function
def predict_news_category(news_text):
    if not news_text.strip():
        return "<p style='color:#aaa;'>⚠️ Please enter some text.</p>"

    cleaned = clean_text(news_text)
    vectorized = vectorizer.transform([cleaned])
    pred_label = model.predict(vectorized)[0]
    category = label_encoder.inverse_transform([pred_label])[0]

    # Simple, calm output
    result = f"""
    <div style='text-align:center; padding:10px;'>
        <p style='font-size:16px; color:#cccccc; margin-bottom:5px;'>Predicted Category</p>
        <p style='font-size:28px; font-weight:700; color:#5bc0de; margin-top:0px;'>
            {category.upper()}
        </p>
    </div>
    """
    return result

# --- Simple, Modern Dark Theme CSS ---
custom_css = """
body {
    background-color: #121212;
    color: #e0e0e0;
    font-family: 'Segoe UI', sans-serif;
}

.gradio-container {
    max-width: 750px !important;
    margin: 60px auto;
    padding: 30px 40px;
    border-radius: 12px;
    background-color: #1e1e1e;
    box-shadow: 0 0 12px rgba(0,0,0,0.4);
}

h1 {
    text-align: center;
    color: #5bc0de;
    font-size: 26px;
    font-weight: 600;
    margin-bottom: 10px;
}

.gr-markdown {
    color: #bfbfbf;
    font-size: 15px;
    text-align: center;
}

textarea {
    background-color: #2a2a2a !important;
    color: #ffffff !important;
    border: 1px solid #3a3a3a !important;
    border-radius: 8px !important;
    font-size: 14px !important;
}

button {
    background-color: #5bc0de !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 10px 16px !important;
    font-weight: 600;
    transition: all 0.2s ease-in-out;
}

button:hover {
    background-color: #4aaac4 !important;
}
"""

# --- Interface Layout ---
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("<h1>🧠 HuffPost News Classifier</h1>")
    gr.Markdown(
        "Enter a HuffPost news article and get its predicted category using a Logistic Regression model trained on TF-IDF features."
    )

    news_input = gr.Textbox(
        lines=8,
        placeholder="Type or paste news content here...",
        label="📰 Enter News Text"
    )

    predict_button = gr.Button("Predict Category")

    output = gr.HTML(label="Predicted Category")

    predict_button.click(fn=predict_news_category, inputs=news_input, outputs=output)

# --- Launch App ---
if __name__ == "__main__":
    demo.launch(share=True)
