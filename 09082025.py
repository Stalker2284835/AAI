from tensorflow.keras.proccesing.image import ImageDataGenerator
import os

DATASET_PATH = "dataset/"

num_classes = len(os.listdir(DATASET_PATH))

class_mode = "binary" if num_classes == 2 else "categorical"
train_datagen = ImageDataGenerator(rescale=1/255, validation_split = 0.2)

train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(128,128),
    batch_size=32,
    class_mode=class_mode,
    subset="training"
)

val_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(128,128),
    batch_size=32,
    class_mode=class_mode,
    subset="validation"
)