import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model


def find_model(root_folder, filename="screw_resnet.h5"):
    for dirpath, dirnames, filenames in os.walk(root_folder):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    return None

root_folder = "."  # current folder
model_file = find_model(root_folder)

if model_file:
    print("Model found:", model_file)
    model = load_model(model_file)
else:
    print("Model not found!")

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
