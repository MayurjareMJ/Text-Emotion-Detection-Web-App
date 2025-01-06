!pip install joblib
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib

# Load the pre-trained model
pipe_lr = joblib.load(open("model/text_emotion.pkl", "rb"))

# Emoji dictionary for emotions
emotions_emoji_dict = {
    "anger": "😡",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😊",
    "joy": "😁",
    "neutral": "😐",
    "sad": "😢",
    "sadness": "😭",
    "shame": "😳",
    "surprise": "😲",
}

# Function to predict emotion
def predict_proba(docx):
    result = pipe_lr.predict([docx])
    return result[0]

# Function to get prediction probabilities
def get_prediction_proba(docx):
    result = pipe_lr.predict_proba([docx])
    return result

# Styling
st.set_page_config(page_title="Emotion Detector", page_icon="😊", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f5f5;
        font-family: 'Arial', sans-serif;
    }
    .main-title {
        font-size: 2.5rem;
        color: #4CAF50;
        text-align: center;
        font-weight: bold;
    }
    .subheader {
        color: #555555;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 20px;
    }
    .result-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Main Streamlit app
def main():
    st.markdown("<h1 class='main-title'>Text Emotion Detection</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subheader'>Enter your text below to analyze its emotion!</p>", unsafe_allow_html=True)

    # User input form
    with st.form(key="my_form"):
        raw_text = st.text_area("Type your text here 👇", height=150, placeholder="Enter text to detect emotion...")
        submit_text = st.form_submit_button(label="Analyze Emotion")

    if submit_text:
        # Predictions
        prediction = predict_proba(raw_text)
        probability = get_prediction_proba(raw_text)

        # Display results with styling
        st.markdown("<h3 style='color: #4CAF50;'>Prediction Results</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.subheader("Predicted Emotion")
            emoji_icon = emotions_emoji_dict.get(prediction, "❓")
            st.markdown(f"<h1 style='text-align: center;'>{emoji_icon}</h1>", unsafe_allow_html=True)
            st.write(f"**Emotion:** {prediction}")
            st.write(f"**Confidence:** {np.max(probability):.2f}")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.subheader("Prediction Probability")
            prob_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            prob_df_clean = prob_df.T.reset_index()
            prob_df_clean.columns = ["Emotions", "Probability"]

            # Plot probabilities with Altair
            fig = alt.Chart(prob_df_clean).mark_bar().encode(
                x=alt.X("Emotions", sort="-y"),
                y="Probability",
                color="Emotions",
            ).configure_mark(
                opacity=0.8,
                color="steelblue"
            )
            st.altair_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
