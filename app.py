import streamlit as st
import pickle
from scripts.train import preprocess

st.title('Spam Shield')

message = st.text_input("Message")

if st.button("Analyze"):
    vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))
    model = pickle.load(open('models/sms_classifier_model.pkl', 'rb'))

    cleaned_message = preprocess(message)
    vector_message = vectorizer.transform([cleaned_message])
    prediction = model.predict(vector_message)
    if prediction[0] == 1:
        verdict = 'Spam'
    else:
        verdict = 'Not Spam'
    st.write(f"Your message is: {verdict}")