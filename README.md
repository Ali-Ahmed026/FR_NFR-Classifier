# FR vs NFR Classifier

This project classifies software requirements as **Functional (FR)** or **Non-Functional (NFR)** using a **Logistic Regression** model and a **TF-IDF** vectorizer. A **Streamlit chatbot web app** is included for real-time classification.

---

## Web App Link:
*https://requirements-classifier.streamlit.app/*

---

## Project Overview

- **Domain:** Software Engineering / Requirements Analysis  
- **Objective:** Automatically classify software requirements as FR or NFR  
- **Model Used:** Logistic Regression  
- **Frontend:** Streamlit Chatbot Web App  
- **Programming Language:** Python  
- **Tools & Libraries:** Pandas, Scikit-Learn, NLTK, Joblib, Streamlit  

---

## Project Features

- Data loading & preprocessing  
- Text cleaning (lowercasing, punctuation removal, stopword removal, lemmatization)  
- TF-IDF feature extraction (5,000 features, unigrams + bigrams)  
- Machine learning model training  
- Model evaluation  
- Model & vectorizer saving using Joblib  
- Real-time classification via a Streamlit chatbot interface  
- Prediction logging to a CSV file  

---

## What is FR vs NFR?

| Type | Description | Example |
|------|-------------|---------|
| **FR** (Functional Requirement) | Describes *what* the system should do — a specific behaviour, action, or function | "The system shall allow users to reset their password via email." |
| **NFR** (Non-Functional Requirement) | Describes *how well* the system should behave — quality attributes like performance, security, or usability | "The system shall respond to all requests within 2 seconds." |

---

## Machine Learning Pipeline

- **Text Preprocessing:** Lowercase → remove punctuation/numbers → remove stopwords → lemmatize  
- **Feature Extraction:** TF-IDF Vectorizer (5,000 features, unigrams + bigrams)  
- **Algorithm:** Logistic Regression  
- **Dataset:** 6,000+ labelled software requirements  
- **Why Logistic Regression?**  
  - Well-suited for text classification  
  - Outputs class probabilities (FR % vs NFR %)  
  - Fast, interpretable, and reliable for NLP tasks  

---

## Streamlit Web App Features

- Chat-style interface for entering requirements  
- Instant **FR / NFR** classification with confidence score  
- FR probability vs NFR probability breakdown  
- Colour-coded confidence progress bar  
- Sidebar with model information  
- All predictions automatically logged to `predictions_log.csv`  
- Downloadable CSV prediction log from the sidebar  
- Clear chat button  

---

## Project Files

| File | Description |
|------|-------------|
| `app.py` | Streamlit chatbot application |
| `IDS NFR_FR Classifier.ipynb` | Training notebook (EDA, preprocessing, model training) |
| `classifier_model.pkl` | Saved trained Logistic Regression model |
| `tfidf_vectorizer.pkl` | Saved TF-IDF vectorizer |
| `FR_NFR_Dataset.xlsx` | Labelled dataset of software requirements |
| `predictions_log.csv` | Auto-generated log of all predictions made via the app |
| `requirements.txt` | Python dependencies |

---

## How to Run Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train the model by running `IDS NFR_FR Classifier.ipynb` (generates `classifier_model.pkl` and `tfidf_vectorizer.pkl`)

3. Launch the app:
   ```bash
   streamlit run app.py
   ```

---
