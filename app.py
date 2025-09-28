import streamlit as st
from model import LR, DTC, RFC, GBC, predict_label

st.title("Fake News Classifier Web App")
st.write("Enter a news article below to check if it's real or fake:")

user_input = st.text_area("News Article")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text to classify!")
    else:
        st.subheader("Predictions:")
        st.write(f"Logistic Regression: {predict_label(LR, user_input)}")
        st.write(f"Decision Tree: {predict_label(DTC, user_input)}")
        st.write(f"Random Forest: {predict_label(RFC, user_input)}")
        st.write(f"Gradient Boosting: {predict_label(GBC, user_input)}")
