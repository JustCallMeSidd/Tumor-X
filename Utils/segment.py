import tensorflow as tf
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt

IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# ---------------- Custom Loss & Metrics ----------------
def bce_dice_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)

def dice_coef(y_true, y_pred):
    return 1.0

def iou_score(y_true, y_pred):
    return 1.0

# ---------------- Load Model ----------------
def load_segmentation_model(model_path):
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"bce_dice_loss": bce_dice_loss,
                        "dice_coef": dice_coef,
                        "iou_score": iou_score}
    )
    return model

# ---------------- Preprocess Image ----------------
def preprocess_image_pil(pil_image):
    img_bytes = io.BytesIO()
    pil_image.save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()

    img = tf.image.decode_image(img_bytes, channels=1, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, IMG_SIZE, method="bilinear")
    return img

# ---------------- Segmentation Prediction ----------------
def segment_image(model, pil_image):
    img = preprocess_image_pil(pil_image)
    img_in = tf.expand_dims(img, 0)

    # Predict
    pred = model.predict(img_in, verbose=0)[0]   # (H,W,1)
    pred_bin = (pred[...,0] > 0.5).astype(np.float32)

    # Create transparent red overlay
    overlay = np.zeros((IMG_HEIGHT, IMG_WIDTH, 4))
    overlay[..., 0] = 1.0             # red channel
    overlay[..., 3] = pred_bin * 0.4  # alpha where mask=1

    # ----- Reproduce matplotlib layering -----
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(img[...,0], cmap="gray")
    ax.imshow(overlay)
    ax.axis("off")

    # Convert matplotlib figure → PIL
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)
