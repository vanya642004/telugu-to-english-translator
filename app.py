import streamlit as st
from transformers import MarianTokenizer, MarianMTModel

@st.cache_resource
def load_model():
    model_name = "Helsinki-NLP/opus-mt-te-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

st.title("🌐 Telugu to English Translator")
text = st.text_area("Enter Telugu Text:")

if st.button("Translate"):
    if text.strip():
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        translated = model.generate(**inputs)
        english = tokenizer.decode(translated[0], skip_special_tokens=True)
        st.success("**English Translation:**")
        st.write(english)
    else:
        st.warning("Please enter some Telugu text.")
