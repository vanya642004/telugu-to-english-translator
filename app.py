import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

@st.cache_resource
def load_model():
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

st.title("🌐 Telugu to English Translator (NLLB)")

text = st.text_area("Enter Telugu Text", height=150)

if st.button("Translate"):
    if text.strip():
        inputs = tokenizer(text, return_tensors="pt", src_lang="tel_Telu")
        translated = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"])
        result = tokenizer.decode(translated[0], skip_special_tokens=True)
        st.success("**English Translation:**")
        st.write(result)
    else:
        st.warning("Please enter Telugu text.")
