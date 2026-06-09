# This is a Multiple Class classification model for HuffPost_new_data.
# Used BiLSTM + GloVe Embeddings + Random Search hyperparameter tuning....

# Step- 1 INSTALL NEEDED LIBRARIES
# pip install pandas numpy nltk matplotlib seaborn scikit-learn tensorflow keras wordcloud

# Step-2 IMPORT LIBRARIES
import os
import io
import zipfile
import requests
import random
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import joblib

# Step-3 DOWNLOAD NLTK DATA
nltk.download("stopwords")
nltk.download("wordnet")

# Ensure directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Step-4 LOAD DATA
df = pd.read_csv("huffpost_news_data.csv")
print("Data Loaded Successfully!")

print(df.head(), "\n")
print("Shape:", df.shape)
print("Missing values:", df.isnull().sum())

df["short_description"] = df["short_description"].fillna("")
df["raw_description "] = df["raw_description"].fillna("")
df["description"] = df["description"].fillna("")
print("Missing Values Filled.\n")

# Step-5 DROP UNNEEDED COLUMNS
df = df[["headline", "short_description", "article_section", "description"]].dropna()
df["text"] = df["headline"] + " " + df["short_description"] + " " + df["description"]

print("Cleaned Columns Sample:")
print(df.head())
print("Shape after column cleanup:", df.shape, "\n")

# Step-6 SELECT TOP N CATEGORIES (interactive as in LR script)
TOP_N_CATEGORIES = int(input("Enter number of top categories you want to train on (e.g., 3, 5, 7, 10): "))

top_categories = df["article_section"].value_counts().nlargest(TOP_N_CATEGORIES).index
df = df[df["article_section"].isin(top_categories)]
print(f"\n Top {TOP_N_CATEGORIES} Categories:", list(top_categories))

# Step-7 EDA
# 1) CATEGORY DISTRIBUTION
plt.figure(figsize=(8,5))
sns.countplot(
    data=df,
    y="article_section",
    hue="article_section",
    order=df["article_section"].value_counts().index,
    palette="viridis",
    legend=False
)
plt.title(f"Top {TOP_N_CATEGORIES} Article Categories")
plt.xlabel("Count"); plt.ylabel("Category")
plt.tight_layout()
plt.savefig("reports/lstm_category_distribution.png")
plt.show()

# 2) TEXT LENGTH DISTRIBUTION
df["text_length"] = df["text"].apply(lambda x: len(str(x).split()))
plt.figure(figsize=(6,4))
sns.histplot(df["text_length"], bins=40, kde=True)
plt.title("Distribution of Text Lengths")
plt.tight_layout()
plt.savefig("reports/lstm_text_length_distribution.png")
plt.show()

# 3) WORD CLOUD FOR EACH CATEGORY
for cat in top_categories:
    plt.figure(figsize=(8,6))
    cat_text = " ".join(df[df["article_section"] == cat]["text"].astype(str))
    wc = WordCloud(width=800, height=400, background_color="black", colormap="viridis").generate(cat_text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"☁️ WordCloud for '{cat}'")
    plt.tight_layout()
    safe_cat = "".join(c if c.isalnum() else "_" for c in cat)[:50]
    plt.savefig(f"reports/lstm_wordcloud_{safe_cat}.png")
    plt.show()

# Step-8 CLEAN TEXT
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(words)

df["clean_text"] = df["text"].apply(clean_text)
print("\n Sample Cleaned Text:")
print(df["clean_text"].head(), "\n")

# Step-9 LABEL ENCODING
le = LabelEncoder()
df["label_encoded"] = le.fit_transform(df["article_section"])

# Step-10 TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["label_encoded"],
    test_size=0.2, random_state=42, stratify=df["label_encoded"]
)

print("\n Training Set Size:", X_train.shape[0])
print(" Test Set Size:", X_test.shape[0])

# Step-11 TOKENIZATION + PADDING
vocab_size = 20000
max_len = 120

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len, padding="post")
X_test_pad = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_len, padding="post")

num_classes = len(le.classes_)

# Step-12 LOAD GLOVE EMBEDDINGS (300D)
glove_file = "glove.6B.300d.txt"

if not os.path.exists(glove_file):
    print(" Downloading GloVe 300D...")
    url = "http://nlp.stanford.edu/data/glove.6B.zip"
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(".")
    print("GloVe Downloaded!\n")

embedding_dim = 300
embeddings_index = {}
with open(glove_file, encoding="utf8") as f:
    for line in f:
        values = line.split()
        embeddings_index[values[0]] = np.asarray(values[1:], dtype="float32")

embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    if i < vocab_size and word in embeddings_index:
        embedding_matrix[i] = embeddings_index[word]

# Step-13 BASE LSTM MODEL (train and evaluate)
def build_model(lstm_units=128, dropout_rate=0.3, lr=3e-4):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, weights=[embedding_matrix],
                  input_length=max_len, trainable=True),
        Bidirectional(LSTM(lstm_units, dropout=dropout_rate, recurrent_dropout=0.2)),
        Dense(128, activation="relu"),
        Dropout(0.4),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=Adam(learning_rate=lr),
                  metrics=["accuracy"])
    return model

print("\n  Training Base LSTM Model...")
base_model = build_model(lstm_units=128, dropout_rate=0.3, lr=3e-4)
es = EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)
history = base_model.fit(
    X_train_pad, y_train,
    validation_split=0.1,
    epochs=8,
    batch_size=128,
    callbacks=[es],
    verbose=1
)

base_loss, base_acc = base_model.evaluate(X_test_pad, y_test, verbose=0)
print("\nBaseline LSTM Accuracy:", round(base_acc, 4))

# Step-14 RANDOM SEARCH HYPERPARAMETER TUNING (n_trials)
param_grid = {
    "lstm_units": [96, 128, 160, 192],
    "dropout": [0.2, 0.3, 0.4],
    "lr": [1e-4, 3e-4, 5e-4],
    "batch_size": [64, 96, 128]
}

n_trials = 5
best_acc = 0.0
best_params = None
best_model = None

print("\n  Starting Random Search Tuning (%d trials)..." % n_trials)
for t in range(n_trials):
    params = {k: random.choice(v) for k, v in param_grid.items()}
    print(f"\n Trial {t+1}/{n_trials} -> {params}")
    model_t = build_model(lstm_units=params["lstm_units"], dropout_rate=params["dropout"], lr=params["lr"])
    es_t = EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)
    model_t.fit(
        X_train_pad, y_train,
        validation_split=0.1,
        epochs=10,
        batch_size=params["batch_size"],
        callbacks=[es_t],
        verbose=0
    )
    loss_t, acc_t = model_t.evaluate(X_test_pad, y_test, verbose=0)
    print(f" Test Accuracy: {acc_t:.4f}")

    if acc_t > best_acc:
        best_acc = acc_t
        best_params = params
        best_model = model_t

print("\n Best Random-Search Accuracy: %.4f" % best_acc)
print(" Best Hyperparameters:", best_params)

# Step-15 SAVE BEST MODEL, TOKENIZER & LABEL ENCODER
if best_model is not None:
    best_model.save("models/lstm_best_model.h5")
    print("Saved best_model -> models/lstm_best_model.h5")
else:
    # Fallback: save base model if tuning found nothing better
    base_model.save("models/lstm_base_model.h5")
    print("No tuned model saved; base model saved to models/lstm_base_model.h5")

joblib.dump(tokenizer, "models/tokenizer.pkl")
joblib.dump(le, "models/label_encoder_lstm.pkl")
print("Saved tokenizer.pkl and label_encoder_lstm.pkl in models/")

# Step-16 EVALUATION & METRICS (use best_model if present)
model_to_eval = best_model if best_model is not None else base_model
y_pred = np.argmax(model_to_eval.predict(X_test_pad), axis=1)

print("\n Classification Report:")
report_text = classification_report(y_test, y_pred, target_names=le.classes_)
print(report_text)
with open("reports/lstm_classification_report.txt", "w") as f:
    f.write(report_text)

plt.figure(figsize=(7,5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d",
            xticklabels=le.classes_, yticklabels=le.classes_, cmap="coolwarm")
plt.title("Confusion Matrix - BiLSTM + GloVe (Tuned)")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.tight_layout()
plt.savefig("reports/lstm_confusion_matrix.png")
plt.show()

print("\nAll done. Models and reports saved under ./models and ./reports.")



