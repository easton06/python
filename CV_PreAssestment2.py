
model_local = "/content/drive/MyDrive/Colab Notebooks/YOLO Stuff/carbikeperson.pt" #set to where u wanna save the model on your google drive, but include "/model.pt" at the end
data_local = "/content/drive/MyDrive/Colab Notebooks/YOLO Stuff/Fish.v2i.yolov11/data.yaml"
dataset = "/content/drive/MyDrive/Colab Notebooks/YOLO Stuff/CarBikePerson"


# |%%--%%| <vd4quhPqzX|EizcR85VpB>

!pip install ultralytics

# |%%--%%| <EizcR85VpB|B4SYqTsvow>

from ultralytics import YOLO
import torch
from google.colab import drive
import cv2
import matplotlib.pyplot as plt

# |%%--%%| <B4SYqTsvow|5wGQSBnfob>

drive.mount('/content/drive')
model = YOLO("yolov5su.pt")

# |%%--%%| <5wGQSBnfob|WvUdCOIlBW>
r"""°°°
To Run:
°°°"""
# |%%--%%| <WvUdCOIlBW|0RUwI5BMWM>

from google.colab import drive
drive.mount('/content/drive')

# |%%--%%| <0RUwI5BMWM|sQ2rB9A1qo>

final = model.predict(source= dataset, save=True, conf=0.25, stream=True)
for r in final:
    boxes = r.boxes  # Boxes object for bbox outputs
    # `r.plot()` returns an image (numpy array) with bounding boxes drawn
    im_bgr = r.plot()  # BGR image with detections
    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib

    plt.figure(figsize=(10, 8))
    plt.imshow(im_rgb)
    plt.axis('off')
    plt.title('Detected Objects')
    plt.show()

# |%%--%%| <sQ2rB9A1qo|2CSnff88FV>


