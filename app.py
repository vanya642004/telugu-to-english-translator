import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model and tokenizer once
@st.cache_resource
def load_model():
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Streamlit UI
st.title("🗣️ Telugu to English Translator")

telugu_text = st.text_area("Enter Telugu Text", height=150, placeholder="ఇక్కడ తెలుగు వాక్యాన్ని నమోదు చేయండి...")

if st.button("Translate"):
    if telugu_text.strip() == "":
        st.warning("Please enter some Telugu text.")
    else:
        inputs = tokenizer(telugu_text, return_tensors="pt", src_lang="tel_Telu")
        generated_tokens = model.generate(
            **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"]
        )
        english_translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        st.success("**Translation:**")
        st.write(english_translation)
