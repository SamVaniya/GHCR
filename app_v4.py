import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
import joblib
import cv2
import pandas as pd
import logging
from cachetools import cached, TTLCache

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure cache for model loading
cache = TTLCache(maxsize=10, ttl=3600)  # Cache up to 10 items for 1 hour

# Gujarati consonants and vowels dictionaries
gujarati_consonants_dict = {
    'k': 'ક', 'kh': 'ખ', 'g': 'ગ', 'gh': 'ઘ', 'ng': 'ઙ',
    'ch': 'ચ', 'chh': 'છ', 'j': 'જ', 'z': 'ઝ',
    'at': 'ટ', 'ath': 'ઠ', 'ad': 'ડ', 'adh': 'ઢ', 'an': 'ણ',
    't': 'ત', 'th': 'થ', 'd': 'દ', 'dh': 'ધ', 'n': 'ન',
    'p': 'પ', 'f': 'ફ', 'b': 'બ', 'bh': 'ભ', 'm': 'મ',
    'y': 'ય', 'r': 'ર', 'l': 'લ', 'v': 'વ', 'sh': 'શ',
    'shh': 'ષ', 's': 'સ', 'h': 'હ', 'al': 'ળ', 'ks': 'ક્ષ',
    'gn': 'જ્ઞ'
}

gujarati_vowels_dict = {
    'a': 'આ', 'i': 'ઇ', 'ee': 'ઈ', 'u': 'ઉ',
    'oo': 'ઊ', 'ri': 'ઋ', 'rii': 'ૠ', 'e': 'એ', 'ai': 'ઐ',
    'o': 'ઓ', 'au': 'ઔ', 'amn': 'અં', 'ah': 'અઃ', "ru": "અૃ", "ra": "અ્ર",
    'ar': "્રઅ"
}

# Load the CSV for character combination lookup
df = pd.read_csv("barakshari.csv", index_col=0)

# Cache model loading
@cached(cache)
def load_cached_model(model_path):
    return load_model(model_path)

# Load models
character_model = load_cached_model("Character_model_gray_v2.h5")
character_label_dencoder = joblib.load("Character_label_encoder_gray_v2.joblib")
consonant_model = load_cached_model("Consonant_model_gray_v2.h5")
consonant_label_dencoder = joblib.load("Consonant_label_encoder_gray_v2.joblib")
vowel_model = load_cached_model("Vowel_model_gray_v2.h5")
vowel_label_dencoder = joblib.load("Vowel_label_encoder_gray_v2.joblib")

def analyze_character(character):
    # Define sets for consonants and vowels
    consonants = ['K', 'KH', 'G', 'GH', 'CH', 'CHH', 'J', 'Z', 'AT', 'ATH', 'AD', 'ADH', 'AN', 'T', 'TH', 'D', 'DH', 'N', 'P', 'F', 'B', 'BH', 'M', 'Y', 'R', 'L', 'V', 'SH', 'SHH', 'S', 'H', 'AL', 'KS', 'GN']
    vowels = ["A", "I", "EE", "U", "OO", "E", "O", "AI", "AU", "AMN", "AH", "RA", "AR", "RU"]

    # Initialize variables to store consonant and vowel parts
    consonant_part = ""
    vowel_part = ""
    character = character.upper()

    if character[0:2] == "AR" and character[2:-1] in consonants:
        consonant_part = character[2:-1]
        vowel_part = character[0:2]
    else:
    # Check each possible length of the consonant part
        for i in range(1, len(character)):
            
            if character[:i] in consonants and character[i:] in vowels:
                consonant_part = character[:i]
                vowel_part = character[i:]
                break

    return consonant_part, vowel_part

def main():
    # Placeholder for heading
    st.title("Gujarati Handwritten Character Recognizer")

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    # Make predictions
    if st.button("Predict"):
        predict_image(uploaded_file)

def predict_image(uploaded_file):
    if uploaded_file is not None:
        try:
            # Preprocess the image
            image = Image.open(uploaded_file)
            image = image.resize((50, 50))
            image = image.convert('L')
            image = np.array(image)
            image = cv2.GaussianBlur(image, (3, 3), 0)
            image_array = image / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            # Make predictions with each model
            consonant_prediction = consonant_model.predict(image_array)
            consonant_predicted_class = np.argmax(consonant_prediction)
            consonant_label = consonant_label_dencoder.inverse_transform([consonant_predicted_class])[0]
            consonant_guj_label = get_gujarati_label(consonant_label, gujarati_consonants_dict)

            vowel_prediction = vowel_model.predict(image_array)
            vowel_predicted_class = np.argmax(vowel_prediction)
            vowel_label = vowel_label_dencoder.inverse_transform([vowel_predicted_class])[0]
            vowel_guj_label = get_gujarati_label(vowel_label, gujarati_vowels_dict)

            character_prediction = character_model.predict(image_array)
            character_predicted_class = np.argmax(character_prediction)
            character_label = character_label_dencoder.inverse_transform([character_predicted_class])[0]
            consonant_part, vowel_part = analyze_character(character_label)
            consonant_part_guj, vowel_part_guj = get_gujarati_label(consonant_part, gujarati_consonants_dict), get_gujarati_label(vowel_part, gujarati_vowels_dict)
            character_guj_label = df.loc[consonant_part_guj.strip(), vowel_part_guj.strip()]

            char_conf = character_prediction[0][character_predicted_class] * 100
            con_conf = consonant_prediction[0][consonant_predicted_class] * 100
            vow_conf = vowel_prediction[0][vowel_predicted_class] * 100

            avg_vc_conf = (con_conf + vow_conf) / 2

            if avg_vc_conf < char_conf:
                label = character_label
                guj_label = character_guj_label
                confidence = char_conf
            else:
                label = consonant_label + vowel_label
                guj_label = df.loc[consonant_guj_label.strip(), vowel_guj_label.strip()]
                confidence = avg_vc_conf

            # Display results
            st.subheader("Prediction Results")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.markdown(f"<p style='font-size:40px'><b>Character Prediction:</b> <br>{consonant_label} + {vowel_label} = {label}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:40px'><b>In Gujarati:</b> <br>{consonant_guj_label} + {vowel_guj_label} = {guj_label}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:40px'><b>Confidence:</b> <br>{confidence:.2f}%</p>", unsafe_allow_html=True)

        except Exception as e:
            # Log error and display user feedback
            logging.error("Error during prediction: %s", str(e))
            st.error("An error occurred during prediction. Please try again with a different image.")
    else:
        st.warning("Please upload an image before predicting.")

def get_gujarati_label(class_label, gujarati_dict):
    guj_class_label = ""
    if class_label.lower() in gujarati_dict.keys():
        guj_class_label = gujarati_dict[class_label.lower()]

    return guj_class_label

if __name__ == "__main__":
    main()
