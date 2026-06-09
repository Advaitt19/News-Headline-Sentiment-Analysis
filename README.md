# News Headline Sentiment Analysis

## Project Overview

This project uses Natural Language Processing (NLP) and Machine Learning techniques to analyze and classify news headlines into sentiment categories. The system processes textual news headlines and predicts their sentiment using both traditional machine learning and deep learning approaches.

## Business Problem

News platforms generate thousands of headlines daily. Manually categorizing sentiment is time-consuming and inconsistent. This project automates sentiment analysis, enabling faster content categorization and trend analysis.

## Dataset

* Dataset: HuffPost News Category Dataset
* Source: Publicly available news headline dataset
* Features:

  * Headline
  * Category
  * Authors
  * Date
  * Link
  * Short Description

## Project Architecture

1. Data Collection
2. Data Cleaning & Preprocessing
3. Text Vectorization
4. Model Training
5. Sentiment Prediction
6. Performance Evaluation

## Models Implemented

### Logistic Regression

* TF-IDF Vectorization
* Fast training and inference
* Baseline model

### LSTM Deep Learning Model

* Word Embeddings (GloVe)
* Long Short-Term Memory Network
* Improved contextual understanding

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* TensorFlow / Keras
* NLTK
* Matplotlib
* Seaborn
  
## Results

* Successfully trained Logistic Regression and LSTM models.
* Automated sentiment classification of news headlines.
* Compared traditional machine learning and deep learning performance.

## How to Run

1. Clone Repository

git clone https://github.com/Advaitt19/News-Headline-Sentiment-Analysis.git

2. Install Dependencies

pip install -r Requirments_libraries.txt

3. Run Training Scripts

python train.py

4. Run Prediction Application

python app.py

## Future Improvements

* Transformer Models (BERT)
* Real-Time News Sentiment Dashboard
* API Deployment using FastAPI
* Streamlit Web Application

## Author

Advait Tandel

Chemical Engineer | Data Science & AI Enthusiast

LinkedIn: Add Your LinkedIn URL
GitHub: https://github.com/Advaitt19
