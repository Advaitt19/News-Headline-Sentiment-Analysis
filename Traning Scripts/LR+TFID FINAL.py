


# This is a Multiple Class classification model for HuffPost_new_data.
# Used Logistic regression and TFIDVectorization....


# Step- 1 INSTALL NEEDED LIBRARIES

# pip install pandas numpy nltk matplotlib seaborn scikit-learn wordcloud

# Step-2 IMPORTING THE REQUIRED LIBRARIES

import pandas as pd
import numpy as np
import re, string, nltk, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

# Step- 3 DOWNLOAD THE REQUIRED NLTK DATA
nltk.download('stopwords')
nltk.download('wordnet')

# Step- 4 LOADING FILE FOR ANALYSIS
df = pd.read_csv("huffpost_news_data.csv")
print("Data Loaded Successfully and we can go further!")

# Step-5 Check data the headings,Shape check for missing values
print(df.head(), " Data Head")
print("Shape:", df.shape)
print("Missing values:", df.isnull().sum())
df['short_description'] = df['short_description'].fillna('')
df['raw_description '] = df['raw_description'].fillna('')
df['description'] = df['description'].fillna('')
print("Missing values filled")

# Step - 6 DROPPING THE COLUMNS WHICH ARE NOT NEEDED
df = df[['headline', 'short_description', 'article_section', 'description']].dropna()
df['text'] = df['headline'] + ' ' + df['short_description'] + ' ' + df['description']
print("\n Cleaning Columns done:")

print("CHecking again the head :",df.head())
print("Shape of the table after removing unwanted columns:", df.shape)

# Step - 7 SELECTING THE NUMBER OF CATEGORIES
TOP_N_CATEGORIES = int(input("Enter number of top categories you want to train on (e.g., 3, 5, 7, 10): "))

# Step - 8  FILTERING THE TOP SELECTED NUMBER OF CATEGORIES
top_categories = df['article_section'].value_counts().nlargest(TOP_N_CATEGORIES).index
df = df[df['article_section'].isin(top_categories)]
print(f"\n Top {TOP_N_CATEGORIES} Categories:", list(top_categories))

# Step - 9 LETS CARRY OUT EDA WITH THE DATASET

# 1) CATEGORY DISTRIBUTION
plt.figure(figsize=(8,5))
sns.countplot(
    data=df,
    y='article_section',
    hue='article_section',
    order=df['article_section'].value_counts().index,
    palette='viridis',
    legend=False
)
plt.title(f"Top {TOP_N_CATEGORIES} Article Categories")
plt.xlabel("Count")
plt.ylabel("Category")
plt.show()

# 2) TEXT LENGTH DISTRIBUTION
df['text_length'] = df['text'].apply(lambda x: len(x.split()))
plt.figure(figsize=(6,4))
sns.histplot(df['text_length'], bins=40, kde=True, color='teal')
plt.title("Distribution of Text Lengths")
plt.show()

# 3) WORD CLOUD FOR EACH CATEGORY
for cat in top_categories:
    plt.figure(figsize=(8,6))
    cat_text = ' '.join(df[df['article_section'] == cat]['text'])
    wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='viridis').generate(cat_text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"☁️ WordCloud for '{cat}'")
    plt.show()

# Step - 10 CLEANING THE DATA INCREASING ACCURACY
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return ' '.join(words)

df['clean_text'] = df['text'].apply(clean_text)
print("\n Sample Cleaned Text:")
print(df['clean_text'].head())

# CHECKING THE DATASET
print("Data set after cleaning:",df.head())

# Step - 11 ASSIGNING NUMERICAL VALUE TO THE CATERGORIES SEECTION FOR SIMPLE CLASSIFICATION
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['article_section'])

# Step - 12 SPLIT THE DATA FOR TESTING AND TRAINING

X_train, X_test, y_train, y_test = train_test_split( df['clean_text'], df['label_encoded'],test_size=0.2, random_state=42, stratify=df['label_encoded'])

print("\n Training Set Size:", X_train.shape[0])
print(" Test Set Size:", X_test.shape[0])

# Step - 13 USING TF-IDF VECTORIZATION TO VECTORIZE WORDS
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Step - 13   BASIC LOGISTICS REGRESSION + TFIDF MODEL TRAINING
lr = LogisticRegression(max_iter=500, multi_class='multinomial')
lr.fit(X_train_tfidf, y_train)
y_pred_lr = lr.predict(X_test_tfidf)

# accuracy for Logistic regression base model
acc_lr = accuracy_score(y_test, y_pred_lr)
print("\n Baseline Logistic Regression Accuracy:", round(acc_lr, 4))

# Step - 14  HYPERPARAMETER TUNING WITH GRID SEARCH ON THE BASE MODEL
param_grid = {
    'C': [0.1, 1, 3],
    'solver': ['lbfgs', 'saga'],
    'max_iter': [500, 1000]}
# Fitting the GridSearch for finding best parameters
grid = GridSearchCV(LogisticRegression(multi_class='multinomial'), param_grid, cv=3, n_jobs=-1, verbose=1)
grid.fit(X_train_tfidf, y_train)


best_lr = grid.best_estimator_
y_pred_grid = best_lr.predict(X_test_tfidf)

# accuracy for Logistic regression Best tuned model
acc_lr_grid = accuracy_score(y_test, y_pred_grid)
print("\n Best Logistic Regression Accuracy + GRID SEARCH :", round(acc_lr_grid, 4))
print(" Best Parameters:", grid.best_params_)

# Step - 15 Check for Metrics Evaluation

# EVALUATION REPORT
print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred_grid, target_names=le.classes_))

# Confusion Matrix
plt.figure(figsize=(7,5))
sns.heatmap(confusion_matrix(y_test, y_pred_grid), annot=True, fmt='d',
            xticklabels=le.classes_, yticklabels=le.classes_, cmap='coolwarm')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# SAVING TRAINED MODEL, TF-IDF VECTORIZER & LABEL ENCODER
import joblib
# Save best logistic regression model
joblib.dump(best_lr, "best_lr.pkl")

# Save TF-IDF vectorizer
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

# Save Label Encoder
joblib.dump(le, "label_encoder.pkl")

print("\n All model artifacts saved successfully!")
print(" Saved Files:")
print(" best_lr.pkl")
print(" tfidf_vectorizer.pkl")
print(" label_encoder.pkl")


