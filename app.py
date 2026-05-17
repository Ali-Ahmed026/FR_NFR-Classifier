"""
Software Requirements Classifier — Streamlit Chatbot
Classifies a requirement as Functional (FR) or Non-Functional (NFR)
and appends every prediction to a CSV log file.

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib
import re
import os
import datetime

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ── Download NLTK data silently if not already present ──────────────────────
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ── File paths ───────────────────────────────────────────────────────────────
MODEL_FILE      = 'classifier_model.pkl'
VECTORIZER_FILE = 'tfidf_vectorizer.pkl'
CSV_LOG_FILE    = 'predictions_log.csv'


# ── Text cleaning (must match exactly what was done during training) ──────────
def clean_text(raw_text):
    """
    Cleans a single requirement string the same way the training notebook did.
    lowercase -> remove punctuation/numbers -> remove stopwords -> lemmatize
    """
    english_stopwords = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    text = raw_text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    cleaned_words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in english_stopwords
    ]
    return ' '.join(cleaned_words)


# ── Load the saved model and vectorizer (cached so they load only once) ───────
@st.cache_resource
def load_model_and_vectorizer():
    """Loads and returns the trained classifier and TF-IDF vectorizer from disk."""
    classifier = joblib.load(MODEL_FILE)
    vectorizer  = joblib.load(VECTORIZER_FILE)
    return classifier, vectorizer


# ── Prediction logic ──────────────────────────────────────────────────────────
def predict_requirement(raw_text, classifier, vectorizer):
    """
    Given a raw requirement string, returns:
      - predicted_label : 'FR' or 'NFR'
      - confidence      : model's confidence as a percentage (0-100)
      - fr_prob         : probability it is FR
      - nfr_prob        : probability it is NFR
    """
    cleaned  = clean_text(raw_text)
    features = vectorizer.transform([cleaned])

    predicted_label = classifier.predict(features)[0]
    probabilities   = classifier.predict_proba(features)[0]
    class_labels    = classifier.classes_          # e.g. ['FR', 'NFR']

    # Map each class to its probability
    prob_dict = dict(zip(class_labels, probabilities))
    fr_prob   = prob_dict.get('FR', 0) * 100
    nfr_prob  = prob_dict.get('NFR', 0) * 100
    confidence = max(fr_prob, nfr_prob)

    return predicted_label, confidence, fr_prob, nfr_prob


# ── CSV logging ───────────────────────────────────────────────────────────────
def save_prediction_to_csv(requirement_text, predicted_label, confidence):
    """
    Appends a new row to the CSV log file.
    Creates the file with a header if it does not exist yet.
    """
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_row   = {
        'Timestamp':        timestamp,
        'Requirement Text': requirement_text,
        'Predicted Type':   predicted_label,
        'Confidence (%)':   round(confidence, 2),
    }

    # If the CSV already exists, append without writing the header again
    file_exists = os.path.isfile(CSV_LOG_FILE)
    row_df = pd.DataFrame([new_row])
    row_df.to_csv(CSV_LOG_FILE, mode='a', header=not file_exists, index=False)


# ── Streamlit page configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title='Requirements Classifier',
    page_icon='🤖',
    layout='centered'
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title('🤖 Software Requirements Classifier')
st.markdown(
    'Type a software requirement below and the AI will classify it as '
    '**Functional (FR)** or **Non-Functional (NFR)**.'
)
st.divider()

# ── Check that model files exist before trying to load them ──────────────────
if not os.path.isfile(MODEL_FILE) or not os.path.isfile(VECTORIZER_FILE):
    st.error(
        '**Model files not found.**\n\n'
        'Please run the `analysis.ipynb` notebook first to train and save '
        f'`{MODEL_FILE}` and `{VECTORIZER_FILE}`.'
    )
    st.stop()  # Stop rendering the rest of the page

# Load model once (cached by Streamlit)
classifier, vectorizer = load_model_and_vectorizer()

# ── Initialise chat history in Streamlit session state ────────────────────────
# session_state persists across reruns (like a page refresh triggered by user input)
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# ── Display existing chat messages ───────────────────────────────────────────
for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# ── Chat input box at the bottom of the page ─────────────────────────────────
user_input = st.chat_input('Enter a software requirement here...')

if user_input:
    # Trim leading/trailing whitespace from input
    user_input = user_input.strip()

    # Show the user's message in the chat
    with st.chat_message('user'):
        st.markdown(user_input)

    # Save user message to chat history
    st.session_state.chat_history.append({
        'role':    'user',
        'content': user_input
    })

    # ── Validate input: don't classify empty or very short text ──────────────
    if len(user_input.split()) < 3:
        bot_reply = (
            'Please enter a complete requirement sentence (at least 3 words) '
            'so I can classify it accurately.'
        )
        with st.chat_message('assistant'):
            st.markdown(bot_reply)
        st.session_state.chat_history.append({
            'role':    'assistant',
            'content': bot_reply
        })

    else:
        # ── Run the classifier ────────────────────────────────────────────────
        predicted_label, confidence, fr_prob, nfr_prob = predict_requirement(
            user_input, classifier, vectorizer
        )

        # ── Save the result to CSV ────────────────────────────────────────────
        save_prediction_to_csv(user_input, predicted_label, confidence)

        # ── Build the bot's response message ─────────────────────────────────
        if predicted_label == 'FR':
            label_display = '✅ **Functional Requirement (FR)**'
            explanation = (
                'This requirement describes **what the system should do** — '
                'a specific behaviour, action, or function that the system must perform.'
            )
        else:
            label_display = '⚙️ **Non-Functional Requirement (NFR)**'
            explanation = (
                'This requirement describes **how the system should behave** — '
                'quality attributes like performance, security, usability, or reliability.'
            )

        bot_reply = (
            f'**Classification:** {label_display}\n\n'
            f'**Confidence:** {confidence:.1f}%\n\n'
            f'**FR probability:** {fr_prob:.1f}%  |  '
            f'**NFR probability:** {nfr_prob:.1f}%\n\n'
            f'{explanation}\n\n'
            f'*This prediction has been saved to `{CSV_LOG_FILE}`.*'
        )

        with st.chat_message('assistant'):
            st.markdown(bot_reply)

        # ── Colour-coded confidence bar ───────────────────────────────────────
        with st.chat_message('assistant'):
            st.progress(int(confidence), text=f'Model confidence: {confidence:.1f}%')

        # Save assistant messages to chat history
        st.session_state.chat_history.append({
            'role':    'assistant',
            'content': bot_reply
        })

# ── Sidebar: information and CSV log viewer ───────────────────────────────────
with st.sidebar:
    st.header('ℹ️ About')
    st.markdown(
        '**Model:** Logistic Regression\n\n'
        '**Features:** TF-IDF (5000 features, unigrams + bigrams)\n\n'
        '**Dataset:** 6,000+ labelled software requirements\n\n'
        '---\n'
        '**FR** — Functional Requirement: describes *what* the system does.\n\n'
        '**NFR** — Non-Functional Requirement: describes *how well* the system does it.'
    )

    st.divider()

    # ── Clear chat button ────────────────────────────────────────────────────
    if st.button('🗑️ Clear Chat'):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()

    # ── Show the CSV log if it exists ─────────────────────────────────────────
    st.header('📋 Prediction Log')
    if os.path.isfile(CSV_LOG_FILE):
        log_df = pd.read_csv(CSV_LOG_FILE)
        st.markdown(f'**{len(log_df)} predictions saved**')
        st.dataframe(log_df, use_container_width=True)

        # Download button so the user can export the CSV
        csv_data = log_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label='⬇️ Download CSV',
            data=csv_data,
            file_name='predictions_log.csv',
            mime='text/csv'
        )
    else:
        st.info('No predictions yet. Start classifying requirements above!')
