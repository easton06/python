
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# |%%--%%| <5JX5OeK7Yc|TrmUpI5muu>

SCRIPT_DIR = "/home/jesh/experimental/python/cv"
IMAGE_SIZE = 64

def zoom(image, zoom_factor=2):
    h, w = image.shape[:2]
    new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)

    # Crop the center
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    cropped = image[top:top+new_h, left:left+new_w]

    # Resize back to original size
    return cv2.resize(cropped, (IMAGE_SIZE, IMAGE_SIZE))

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

        #do FLIPZ
        imgflip = cv2.flip(img, 1)
        images.append((imgflip, label))
        imgflip = cv2.flip(img, 0)
        images.append((imgflip, label))
        imgflip = cv2.flip(img, -1)
        images.append((imgflip, label))

        #apply zoomz
        for i in np.arange(0.1, 2, 0.1):
          imgzoomed = zoom(img, i)
          images.append((imgzoomed,label))

        #Making them rotate, weeeeee
        for i in range(1, 360, 10):
          rotation_fac = cv2.getRotationMatrix2D((IMAGE_SIZE/2,IMAGE_SIZE/2),i,1)
          imgrot = cv2.warpAffine(img,rotation_fac,(IMAGE_SIZE,IMAGE_SIZE))
          images.append((imgrot,label))

    return images

calculator_images = load_images_from_folder(SCRIPT_DIR + '/images/data/calculator', 0)
frog_images = load_images_from_folder(SCRIPT_DIR + '/images/data/frog', 1)

data = calculator_images + frog_images
np.random.shuffle(data)

X = np.array([item[0] for item in data]) / 255.0
y = np.array([item[1] for item in data])

y_cat = to_categorical(y, 2)
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# |%%--%%| <TrmUpI5muu|NXSkJGXnnQ>

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
model.save("augmented_calcfrog.h5")

# |%%--%%| <NXSkJGXnnQ|iTO6E0DJ5r>

model = load_model("augmented_calcfrog.h5")
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

predict_image(SCRIPT_DIR + "/images/data/test/test3.jpg")
predict_image(SCRIPT_DIR + "/images/data/test/test4.jpg")
