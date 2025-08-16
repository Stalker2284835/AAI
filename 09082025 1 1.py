from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import os


DATAPATH = 'dataset/'
if not os.path.exists(DATAPATH):
    raise ValueError("Папка dataset/ не найдена. Проверьте путь.")

num_classes = len(os.listdir(DATAPATH))
if num_classes < 3:
    raise ValueError("В dataset/ должно быть минимум 3 класса.")


class_mode = "categorical"
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

try:
    train_data = train_datagen.flow_from_directory(
        DATAPATH, target_size=(128, 128), batch_size=32, class_mode=class_mode, subset="training"
    )
    val_data = train_datagen.flow_from_directory(
        DATAPATH, target_size=(128, 128), batch_size=32, class_mode=class_mode, subset="validation"
    )
except Exception as e:
    raise ValueError(f"Ошибка при загрузке данных: {str(e)}")


model = Sequential([
    Input(shape=(128, 128, 3)),
    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.4),
    Dense(num_classes, activation="softmax")
])


model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
]


try:
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=20,
        callbacks=callbacks
    )
except Exception as e:
    raise ValueError(f"Ошибка при обучении модели: {str(e)}")


try:
    test_loss, test_accuracy = model.evaluate(val_data)
    print(f"Точность модели на валидационных данных: {test_accuracy:.2f}")
except Exception as e:
    raise ValueError(f"Ошибка при оценке модели: {str(e)}")

# Сохранение модели
model.save("image_classifier.keras")  # Изменено на .keras
print("Модель сохранена как image_classifier.keras")