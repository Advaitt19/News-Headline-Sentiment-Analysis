### HuffPost News Classification ###
Logistic Regression (TF-IDF) + LSTM (GloVe) + Gradio Apps
This project provides a complete end-to-end NLP pipeline for classifying HuffPost news articles using:
Traditional Machine Learning → TF-IDF + Logistic Regression
Deep Learning → BiLSTM with pretrained GloVe embeddings
Interactive Gradio applications for real-time predictions

Dataset Overview
The HuffPost News Category Dataset includes:
Headline
Short Description
Full Description
Article Section (Category Label)
The project supports selecting Top-N most frequent categories for model training.

Text Preprocessing Pipeline
Both models use structured preprocessing, including:
General Text Cleaning
Lowercasing
Regex-based noise removal
Lemmatization

(Logistic Regression Specific- Stopword removal)

(TF-IDF bigram feature extraction, LSTM + GloVe Specific, SpaCy-based keyword extraction, Sentiment tagging using TextBlob, Tokenization + sequence padding, Integration of GloVe 300D pretrained embeddings)

Models Implemented
 Logistic Regression (TF-IDF)
A lightweight, fast, and reliable baseline model:
TF-IDF (1–2 grams, up to 10,000 features)
Multinomial Logistic Regression
Hyperparameter tuning with GridSearchCV

 BiLSTM + GloVe Embeddings
A deep learning-based model:
Bi-directional LSTM layer
Pretrained GloVe 300D embedding matrix
Random Search hyperparameter tuning
Early stopping for optimal training

Generated Reports

The following visual and analytical reports are produced:
Category distribution
Text length distribution
Word clouds (per category)
Confusion matrix
Classification report (precision, recall, F1)
All .png and .txt reports are saved inside the reports/ directory.

Future Improvements
Planned enhancements include:
Integration of BERT / DistilBERT for improved accuracy
Deployment of the Gradio apps on HuggingFace Spaces
Adding Top-5 predictions with probability charts
Live news prediction via a real-time news API
