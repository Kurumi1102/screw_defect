import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model
import requests

def download_file(url, local_path):
    if os.path.exists(local_path) and os.path.getsize(local_path) > 1024:
        print("File already exists:", local_path)
        return
    print("Downloading:", url)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print("Download complete:", local_path)

# URL of the raw screw_resnet.h5 file from GitHub LFS
url = "https://github.com/Kurumi1102/screw_defect/raw/main/screw_resnet.h5"

# Path where you want to save the model
local_path = os.path.join(os.getcwd(), "screw_resnet.h5")

# Download automatically
download_file(url, local_path)

# Load the model
from tensorflow.keras.models import load_model
model = load_model(local_path)
print("Model loaded successfully!")

# Class names
class_names = [
    "good",
    "manipulated_front",
    "scratch_head",
    "scratch_neck",
    "thread_side",
    "thread_top"
]


# Open file dialog
root = tk.Tk()
root.withdraw()
img_path = filedialog.askopenfilename(
    title="Select Image",
    filetypes=[("Image Files", "*.jpg *.png *.jpeg *.bmp")]
)

if not img_path:
    print("No image selected.")
    exit()

print("Selected:", img_path)

# Load and preprocess image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction)

print("Result:")
print("Prediction:", predicted_class)
print("Confidence:", confidence)

# Show image
plt.imshow(image.load_img(img_path))
plt.title(f"{predicted_class} ({confidence:.2f})")
plt.axis("off")
plt.show()
