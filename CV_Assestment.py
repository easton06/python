import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# |%%--%%| <pIi2T5Bf6x|lwWYcJsbo8>

SCRIPT_DIR = "/home/jesh/experimental/python/cv"
DATA_DIR = SCRIPT_DIR + "/images/MY_data"
IMAGE_SIZE = 64

def load_images_from_folder(folder, label):
    images = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        img = cv2.imread(filepath)
        if img is None:
            print(f"Skipping file (could not read): {filepath}")
            continue
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        images.append((img, label))
    return images
calculator_images = load_images_from_folder(DATA_DIR + "/train/apple", 0)
frog_images = load_images_from_folder(DATA_DIR + "/train/banana", 1)

data = calculator_images + frog_images
np.random.shuffle(data)

X = np.array([item[0] for item in data]) / 255.0
y = np.array([item[1] for item in data])

y_cat = to_categorical(y, 2)
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# |%%--%%| <lwWYcJsbo8|pY0EIveHVj>

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))
model.save("calcfrog.h5")

# |%%--%%| <pY0EIveHVj|kAYhbdEnZV>

model = load_model("calcfrog.h5")
labels = ["Calculator", "Frog"]

def predict_image(image_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)) / 255.0
    img_array = np.expand_dims(img_resized, axis=0)
    prediction = model.predict(img_array)[0]
    predicted_label = labels[np.argmax(prediction)]

    # Display the image with prediction
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Predicted: {predicted_label}")
    plt.axis('off')
    plt.show()

predict_image(DATA_DIR + "/predict/")
predict_image("/content/drive/MyDrive/Colab Notebooks/YOLO Stuff/Sample Image/data/test/test2.jpg")
