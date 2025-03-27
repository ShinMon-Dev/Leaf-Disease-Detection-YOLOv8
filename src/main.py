from ultralytics import YOLO
import cv2
import torch
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog


def detect_leaf_disease(image_path, model_path='models/weights.pt', conf=0.5):
    # Load YOLOv8 model
    model = YOLO(model_path)

    # Load image
    image = cv2.imread(image_path)

    # Run inference
    results = model(image, conf=conf)

    # Draw bounding boxes
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            class_id = int(box.cls[0])
            label = f"{model.names[class_id]}: {confidence:.2f}"

            # Draw rectangle and label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Convert BGR to RGB for Matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Show image using Matplotlib
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.show()


def upload_and_detect():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        detect_leaf_disease(file_path, 'models/weights.pt')


# Create a simple UI
root = tk.Tk()
root.title("Leaf Disease Detection")
button = tk.Button(root, text="Upload Image", command=upload_and_detect)
button.pack(pady=20)
root.mainloop()