from PIL import Image
import tensorflow as tf
import cv2
import os

DATAPATH = 'dataset/'
num_classes = len(os.listdir(DATAPATH))
class_mode = "binary" if num_classes == 2 else "categorical"

def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"Ошибка: файл не найден по пути: {image_path}")
        return
    try:
        img = Image.open(image_path)
        img.verify()
    except (IOError, IOError):
        print(f"Ошибка: Поврежденное изображение - {image_path}")
        return
    model = tf.keras.models.load_model("image_classifier.h5")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Ошибка: Не удалось прочитать изображение - {image_path}")
        return
    img = cv2.resize(img, (128, 128))
    img = tf.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_names = os.listdir(DATAPATH)
    if class_mode == "binary":
        predicted_class = class_names[int(bool(prediction[0] > 0.5))]
    else:
        predicted_class = class_names[tf.argmax(prediction, axis=-1).numpy()[0]]
    print(f"Модель определила: {predicted_class}")

predict_image("images.jpg")