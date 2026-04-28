import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load files
model = load_model("lstm_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("max_sequence_len.pkl", "rb") as f:
    max_sequence_len = pickle.load(f)


# Function to predict next word
def predict_next_word(text):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')

    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]

    # Convert index to word
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word

    return ""

def generate_text(seed_text, next_words=3):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')

        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=1)[0]

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                output_word = word
                break

        seed_text += " " + output_word

    return seed_text


# Streamlit UI
st.set_page_config(page_title="Next Word Predictor", page_icon="🤖")

st.title("📝 Next Word Prediction using LSTM")
st.write("Enter a sentence and get the next predicted word.")
num_words = st.slider("Number of words to generate:", 1, 10, 3)

input_text = st.text_input("Enter text:", "")

if st.button("Predict"):
    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        next_word = generate_text(input_text, next_words=num_words)
        st.success(f"👉 Next word: **{next_word}**")