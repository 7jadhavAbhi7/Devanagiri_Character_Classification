# 
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model


model =load_model("best_model.keras")

class_labels=['character_10_yna', 'character_11_taamatar', 'character_12_thaa', 'character_13_daa', 'character_14_dhaa', 'character_15_adna', 'character_16_tabala', 'character_17_tha', 'character_18_da', 'character_19_dha', 'character_1_ka', 'character_20_na', 'character_21_pa', 'character_22_pha', 'character_23_ba', 'character_24_bha', 'character_25_ma', 'character_26_yaw', 'character_27_ra', 'character_28_la', 'character_29_waw', 'character_2_kha', 'character_30_motosaw', 'character_31_petchiryakha', 'character_32_patalosaw', 'character_33_ha', 'character_34_chhya', 'character_35_tra', 'character_36_gya', 'character_3_ga', 'character_4_gha', 'character_5_kna', 'character_6_cha', 'character_7_chha', 'character_8_ja', 'character_9_jha', 'digit_0', 'digit_1', 'digit_2', 'digit_3', 'digit_4', 'digit_5', 'digit_6', 'digit_7', 'digit_8', 'digit_9']



def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if not file_path:
        return

    # Load and display the image
    img = Image.open(file_path)
    img_resized = img.resize((200, 200)) 
    img_tk = ImageTk.PhotoImage(img_resized)
    image_label.config(image=img_tk)
    image_label.image = img_tk
    predict_character(file_path)
def predict_character(image_path):
    img = Image.open(image_path).convert("L")  
    img = img.resize((32, 32)) 
    img_np = np.array(img) / 255.0  # Normalize
    img_np = np.stack((img_np,)*3, axis=-1)  
    img_np = img_np.reshape(1, 32, 32, 3) 
    predictions = model.predict(img_np)
    predicted_class_idx = np.argmax(predictions)
    predicted_label = class_labels[predicted_class_idx]
    result_label.config(text=f"Predicted: {predicted_label}")

# Create GUI window
root = tk.Tk()
root.title("Devnagari Character Recognition")
root.geometry("400x500")

# Upload Button
upload_btn = tk.Button(root, text="Upload Image", command=upload_image, font=("Arial", 14))
upload_btn.pack(pady=10)

# Label to display uploaded image
image_label = tk.Label(root)
image_label.pack()

# Prediction result label
result_label = tk.Label(root, text="Predicted: ", font=("Arial", 16, "bold"))
result_label.pack(pady=10)

# Run the GUI
root.mainloop()
