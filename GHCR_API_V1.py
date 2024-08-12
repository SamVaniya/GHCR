from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
from PIL import Image
from keras.models import load_model
import joblib
import cv2
import logging
from cachetools import cached, TTLCache

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize FastAPI
app = FastAPI()

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

def get_gujarati_label(class_label, gujarati_dict):
    guj_class_label = ""
    if class_label.lower() in gujarati_dict.keys():
        guj_class_label = gujarati_dict[class_label.lower()]

    return guj_class_label

def preprocess_image(uploaded_file):
    # Preprocess the image
    image = Image.open(uploaded_file.file)
    image = image.resize((50, 50))
    image = image.convert('L')
    image = np.array(image)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image_array = image / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict_label(image_array, model, label_encoder, gujarati_dict):
    try:
        prediction = model.predict(image_array)
        predicted_class = np.argmax(prediction)
        english_label = label_encoder.inverse_transform([predicted_class])[0]
        if model == character_model:
            consonant_part, vowel_part = analyze_character(english_label)
            consonant_part_guj, vowel_part_guj = get_gujarati_label(consonant_part, gujarati_consonants_dict), get_gujarati_label(vowel_part, gujarati_vowels_dict)
            gujarati_label = df.loc[consonant_part_guj.strip(), vowel_part_guj.strip()]
        else:
            gujarati_label = get_gujarati_label(english_label, gujarati_dict)
        confidence = prediction[0][predicted_class] * 100
        return english_label, gujarati_label, confidence
    except Exception as e:
        logging.error("Error during prediction: %s", str(e))
        raise HTTPException(status_code=500, detail="Prediction error")

@app.post("/predict/character")
async def predict_character(uploaded_file: UploadFile = File(...)):
    try:
        image_array = preprocess_image(uploaded_file)
        english_label, gujarati_label, confidence = predict_label(image_array, character_model, character_label_dencoder, df)
        return JSONResponse(content={
            "english_label": english_label,
            "gujarati_label": gujarati_label,
            "confidence": confidence
        })
    except Exception as e:
        logging.error("Error during character prediction: %s", str(e))
        raise HTTPException(status_code=400, detail="Invalid image or prediction error")

@app.post("/predict/consonant")
async def predict_consonant(uploaded_file: UploadFile = File(...)):
    try:
        image_array = preprocess_image(uploaded_file)
        english_label, gujarati_label, confidence = predict_label(image_array, consonant_model, consonant_label_dencoder, gujarati_consonants_dict)
        return JSONResponse(content={
            "english_label": english_label,
            "gujarati_label": gujarati_label,
            "confidence": confidence
        })
    except Exception as e:
        logging.error("Error during consonant prediction: %s", str(e))
        raise HTTPException(status_code=400, detail="Invalid image or prediction error")

@app.post("/predict/vowel")
async def predict_vowel(uploaded_file: UploadFile = File(...)):
    try:
        image_array = preprocess_image(uploaded_file)
        english_label, gujarati_label, confidence = predict_label(image_array, vowel_model, vowel_label_dencoder, gujarati_vowels_dict)
        return JSONResponse(content={
            "english_label": english_label,
            "gujarati_label": gujarati_label,
            "confidence": confidence
        })
    except Exception as e:
        logging.error("Error during vowel prediction: %s", str(e))
        raise HTTPException(status_code=400, detail="Invalid image or prediction error")
