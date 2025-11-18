import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

# Load model
model = tf.keras.models.load_model("*/screw_resnet.h5")

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
