import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
import joblib
import cv2
import pandas as pd

# Gujarati character dictionaries
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

df = pd.read_csv("barakshari.csv", index_col=0)

def main():
    # Placeholder for heading
    st.title("Gujarati Handwritten Character Recognizer")

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    # Load all models and label encoders
    try:
        character_model, character_label_dencoder = load_selected_model("Character Model")
        consonant_model, consonant_label_dencoder = load_selected_model("Consonant Model")
        vowel_model, vowel_label_dencoder = load_selected_model("Vowel Model")
    except Exception as e:
        st.error(f"Error loading models or label encoders: {e}")
        return

    # Make predictions
    if st.button("Predict"):
        if uploaded_file is not None:
            try:
                predict_image(uploaded_file, character_model, character_label_dencoder,
                              consonant_model, consonant_label_dencoder,
                              vowel_model, vowel_label_dencoder)
            except Exception as e:
                st.error(f"Error making predictions: {e}")
        else:
            st.warning("Please upload an image before predicting.")

def load_selected_model(selected_model):
    model_path = ""
    label_encoder_path = ""
    
    if selected_model == "Character Model":
        model_path = "Character_model_gray_v2.h5"
        label_encoder_path = "Character_label_encoder_gray_v2.joblib"
    elif selected_model == "Consonant Model":
        model_path = "Consonant_model_gray_v2.h5"
        label_encoder_path = "Consonant_label_encoder_gray_v2.joblib"
    elif selected_model == "Vowel Model":
        model_path = "Vowel_model_gray_v2.h5"
        label_encoder_path = "Vowel_label_encoder_gray_v2.joblib"

    try:
        model = load_model(model_path)
        print(f"Loaded {selected_model} model from {model_path}")
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        raise

    try:
        label_dencoder = joblib.load(label_encoder_path)
        print(f"Loaded {selected_model} label encoder from {label_encoder_path}")
    except Exception as e:
        print(f"Error loading label encoder {label_encoder_path}: {e}")
        raise

    return model, label_dencoder

def predict_image(uploaded_file, character_model, character_label_dencoder,
                  consonant_model, consonant_label_dencoder,
                  vowel_model, vowel_label_dencoder):
    # Preprocess the image
    image = Image.open(uploaded_file)
    image = image.resize((50, 50))
    image = image.convert('L')
    image = np.array(image)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image_array = image / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Print shape for debugging
    print(f"Image array shape: {image_array.shape}")

    # Make predictions with each model
    try:
        consonant_prediction = consonant_model.predict(image_array)
        consonant_predicted_class = np.argmax(consonant_prediction)
        consonant_label = consonant_label_dencoder.inverse_transform([consonant_predicted_class])[0]
        consonant_guj_label = get_gujarati_label(consonant_label, gujarati_consonants_dict)
    except Exception as e:
        print(f"Error predicting consonant: {e}")
        consonant_label = "Error"
        consonant_guj_label = "Error"

    try:
        vowel_prediction = vowel_model.predict(image_array)
        vowel_predicted_class = np.argmax(vowel_prediction)
        vowel_label = vowel_label_dencoder.inverse_transform([vowel_predicted_class])[0]
        vowel_guj_label = get_gujarati_label(vowel_label, gujarati_vowels_dict)
    except Exception as e:
        print(f"Error predicting vowel: {e}")
        vowel_label = "Error"
        vowel_guj_label = "Error"

    try:
        character_prediction = character_model.predict(image_array)
        character_predicted_class = np.argmax(character_prediction)
        character_label = character_label_dencoder.inverse_transform([character_predicted_class])[0]
        character_guj_label = df.loc[consonant_guj_label.strip(), vowel_guj_label.strip()]
    except Exception as e:
        print(f"Error predicting character: {e}")
        character_label = "Error"
        character_guj_label = "Error"

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

def get_gujarati_label(class_label, gujarati_dict):
    return gujarati_dict.get(class_label.lower(), "")

if __name__ == "__main__":
    main()
