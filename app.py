import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer

# Initialising Porter Stemmer
ps = PorterStemmer()

# Function to preprocess text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load the vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit app title and description with new styling
st.set_page_config(page_title="Email Spam Classifier", page_icon="📧", layout="wide")

st.markdown(
    """
    <style>
    body {
        background-color: #f4f7fc;
        font-family: 'Roboto', sans-serif;
    }
    .main-title {
        font-size: 3.5em;
        color: #4A90E2;
        text-align: center;
        font-weight: bold;
        padding-top: 50px;
    }
    .sub-heading {
        text-align: center;
        color: #7B7D7D;
        font-size: 1.3em;
        margin-top: 5px;
        margin-bottom: 20px;
    }
    .result-spam {
        background-color: #FF6F61;
        color: white;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        font-size: 1.2em;
        margin-top: 30px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .result-notspam {
        background-color: #48C774;
        color: white;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        font-size: 1.2em;
        margin-top: 30px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .input-box {
        margin: 0 auto;
        display: block;
        width: 80%;
        max-width: 800px;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #D3D8E0;
        font-size: 1.2em;
        margin-top: 20px;
        background-color: white;
    }
    .button {
        background-color: #4A90E2;
        color: white;
        padding: 10px 20px;
        font-size: 1.2em;
        border-radius: 8px;
        cursor: pointer;
        border: none;
        transition: background-color 0.3s;
    }
    .button:hover {
        background-color: #357ABD;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">Email Spam Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-heading">Use machine learning to detect spam in emails!</div>', unsafe_allow_html=True)

# Input text area with new style
input_sms = st.text_area("Type or paste your email message below:", height=150, key="input_sms", placeholder="Enter your email message here...", label_visibility="collapsed", max_chars=1000, help="Max characters: 1000", )

# Predict button styled with new design
predict_button = st.button("Check if Spam", key="predict_button", help="Click to check if the email is spam or not.", use_container_width=True)

# Button actions and prediction logic
if predict_button:
    if input_sms.strip() == "":
        st.warning("Please enter a message to analyse!")
    else:
        # 1. Preprocess the input text
        transformed_sms = transform_text(input_sms)
        # 2. Vectorise the input text
        vector_input = tfidf.transform([transformed_sms])
        # 3. Predict the result using the model
        result = model.predict(vector_input)[0]
        # 4. Display the result
        if result == 1:
            st.markdown('<div class="result-spam">🚨 This email is SPAM!</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-notspam">✅ This email is NOT SPAM!</div>', unsafe_allow_html=True)

# Optionally, include a footer or additional information at the bottom
st.markdown(
    """
    <footer style="text-align:center; font-size:1em; color:#A2A6B1; margin-top:50px;">
    <p>Built with python by your friendly neighbourhood spam classifier 🤖</p>
    </footer>
    """,
    unsafe_allow_html=True,
)