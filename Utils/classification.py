import numpy as np
import cv2
from tensorflow.keras.models import load_model
import io

# --------- Define Classes ---------
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# --------- Load Model ---------
def load_classification_model(model_path):
    return load_model(model_path)

# --------- Preprocess Image ---------
def preprocess_image_pil(pil_image, target_size=(128, 128)):
    # Convert PIL → numpy array
    img = np.array(pil_image)
    # Ensure BGR like your cv2 code
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# --------- Prediction ---------
def classify_image(model, pil_image):
    img = preprocess_image_pil(pil_image)
    pred_prob = model.predict(img, verbose=0)
    pred_class_index = np.argmax(pred_prob)
    pred_class_name = class_names[pred_class_index]
    confidence = float(pred_prob[0][pred_class_index])
    return pred_class_name, confidence
