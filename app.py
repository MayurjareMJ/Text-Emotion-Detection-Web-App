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
    result = pipe_lr.predict_proba([docx])  # Corrected method
    return result

# Main Streamlit app
def main():
    st.title("Text Emotion Detection")
    st.subheader("Detect Emotion in Text")

    # Form for user input
    with st.form(key="my_form"):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label="Submit")

    if submit_text:
        # Predictions
        prediction = predict_proba(raw_text)
        probability = get_prediction_proba(raw_text)

        # Display results in columns
        col1, col2 = st.columns(2)

        with col1:
            st.success("Prediction")
            emoji_icon = emotions_emoji_dict.get(prediction, "❓")
            st.write(f"{prediction}: {emoji_icon}")
            st.write(f"Confidence: {np.max(probability):.2f}")

        with col2:
            st.success("Prediction Probability")
            prob_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            prob_df_clean = prob_df.T.reset_index()
            prob_df_clean.columns = ["emotions", "probability"]

            # Plot probabilities with Altair
            fig = alt.Chart(prob_df_clean).mark_bar().encode(
                x="emotions",
                y="probability",
                color="emotions",
            )
            st.altair_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
