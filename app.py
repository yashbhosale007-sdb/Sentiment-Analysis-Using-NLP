import streamlit as st
import pickle
import numpy as np

# Page configuration for a "good frontend"
st.set_page_config(
    page_title="Sentiment Analysis Pro",
    page_icon="😊",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to load the model
@st.cache_resource
def load_model():
    try:
        with open('nlp_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to load the vectorizer (Required to transform text to 16 features)
@st.cache_resource
def load_vectorizer():
    try:
        # Replace 'vectorizer.pkl' with your actual vectorizer file name
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return vectorizer
    except FileNotFoundError:
        st.warning("⚠️ 'vectorizer.pkl' not found. Please ensure your vectorizer is in the same directory.")
        return None

# App Header
st.title("😊 Sentiment Analysis Tool")
st.write("Enter text below to determine if the sentiment is **Positive** or **Negative**.")

# Sidebar info
st.sidebar.title("Model Details")
st.sidebar.info("""
**Model:** Multinomial Naive Bayes
**Features:** 16 Specific Keywords/Features
**Classes:** Negative, Positive
""")

# Load assets
model = load_model()
vectorizer = load_vectorizer()

# Main input area
user_input = st.text_area("Analyze Text:", placeholder="Type something here...", height=150)

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    elif model is None:
        st.error("Model failed to load.")
    elif vectorizer is None:
        st.error("Text cannot be processed without a vectorizer.")
    else:
        # 1. Transform text using the vectorizer
        transformed_input = vectorizer.transform([user_input])
        
        # 2. Make prediction
        prediction = model.predict(transformed_input)
        probability = model.predict_proba(transformed_input)
        
        # 3. Display results with style
        sentiment = prediction[0]
        confidence = np.max(probability) * 100
        
        st.divider()
        if sentiment == 'positive':
            st.success(f"### Result: **POSITIVE** (Confidence: {confidence:.2f}%)")
            st.balloons()
        else:
            st.error(f"### Result: **NEGATIVE** (Confidence: {confidence:.2f}%)")
            
# Footer
st.markdown("---")
st.caption("Powered by Scikit-Learn 1.6.1 and Streamlit")
