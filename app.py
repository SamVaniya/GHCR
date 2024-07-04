import streamlit as st
import numpy as np
from PIL import Image,ImageFilter
from keras.models import load_model
import joblib
# import cv2
import pandas as pd

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

gujarati_vowels_dict = {'a': 'આ', 'i': 'ઇ', 'ee': 'ઈ', 'u': 'ઉ',
    'oo': 'ઊ', 'ri': 'ઋ', 'rii': 'ૠ', 'e': 'એ', 'ai': 'ઐ',
    'o': 'ઓ', 'au': 'ઔ', 'amn': 'અં', 'ah': 'અઃ',"ru" : "અૃ","ra" : "અ્ર",
    'ar' : "્રઅ"
}

df = pd.read_csv("barakshari.csv",index_col=0)

def main():
    # Placeholder for heading
    st.title("Gujarati Handwritten Character Recognizer")

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    # Load all models and label encoders
    character_model, character_label_dencoder = load_selected_model("Character Model")
    consonant_model, consonant_label_dencoder = load_selected_model("Consonant Model")
    vowel_model, vowel_label_dencoder = load_selected_model("Vowel Model")

    # Make predictions
    if st.button("Predict"):
        predict_image(uploaded_file, character_model, character_label_dencoder,
                      consonant_model, consonant_label_dencoder,
                      vowel_model, vowel_label_dencoder)

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

    model = load_model(model_path)
    label_dencoder = joblib.load(label_encoder_path)

    return model, label_dencoder

def predict_image(uploaded_file, character_model, character_label_dencoder,
                  consonant_model, consonant_label_dencoder,
                  vowel_model, vowel_label_dencoder):
    if uploaded_file is not None:
        # Preprocess the image
        image = Image.open(uploaded_file)
        image = image.resize((50, 50))
        image = image.convert('L')
        image = image.filter(ImageFilter.GaussianBlur(radius=1.7))
        image = np.array(image)
        # image = cv2.GaussianBlur(image, (3, 3),0)
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
        character_guj_label = df.loc[consonant_guj_label.strip(), vowel_guj_label.strip()]

        char_conf = character_prediction[0][character_predicted_class] * 100
        con_conf = consonant_prediction[0][consonant_predicted_class] * 100
        vow_conf = vowel_prediction[0][vowel_predicted_class] * 100

        avg_vc_conf = (con_conf+vow_conf)/2

        if avg_vc_conf<char_conf:
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
        # st.write("**Character Prediction:", f"{label}**", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:51px'><b>Character Prediction:</b> {label}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:51px'><b>In Gujarati:</b> {guj_label}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:51px'><b>Confidence:</b> {confidence:.2f}%</p>", unsafe_allow_html=True)
        # st.write("**In Gujarati:", f"{guj_label}**", unsafe_allow_html=True)
        # st.write("**Confidence:", f"{confidence:.2f}%**", unsafe_allow_html=True)
        # st.write("Consonant Prediction:", f"{consonant_label}")
        # st.write("In Gujarati:", f"{consonant_guj_label}")
        # st.write("Confidence:", f"{consonant_prediction[0][consonant_predicted_class] * 100:.2f}%")
        # st.write("Vowel Prediction:", f"{vowel_label}")
        # st.write("In Gujarati:", f"{vowel_guj_label}")
        # st.write("Confidence:", f"{vowel_prediction[0][vowel_predicted_class] * 100:.2f}%")
    else:
        st.warning("Please upload an image before predicting.")

def get_gujarati_label(class_label, gujarati_dict):
    guj_class_label = ""
    if class_label.lower() in gujarati_dict.keys():
        guj_class_label = gujarati_dict[class_label.lower()]

    return guj_class_label

if __name__ == "__main__":
    main()

