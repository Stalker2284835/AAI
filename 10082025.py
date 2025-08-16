from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.regularizers import l2
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import os

# Проверка данных
DATAPATH = 'dataset/'
if not os.path.exists(DATAPATH):
    raise ValueError("Папка dataset/ не найдена. Проверьте путь.")

num_classes = len(os.listdir(DATAPATH))
if num_classes < 3:
    raise ValueError("В dataset/ должно быть минимум 3 класса.")

# Аугментация данных
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

try:
    train_data = train_datagen.flow_from_directory(
        DATAPATH, target_size=(128, 128), batch_size=32, class_mode="categorical", subset="training"
    )
    val_data = train_datagen.flow_from_directory(
        DATAPATH, target_size=(128, 128), batch_size=32, class_mode="categorical", subset="validation"
    )
except Exception as e:
    raise ValueError(f"Ошибка при загрузке данных: {str(e)}")

# Балансировка классов
class_weights = class_weight.compute_class_weight(
    'balanced', classes=np.unique(train_data.classes), y=train_data.classes
)
class_weights = dict(enumerate(class_weights))

# Модель
model = Sequential([
    Input(shape=(128, 128, 3)),
    Conv2D(32, (3, 3), activation="relu", kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation="relu", kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation="relu", kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(256, activation="relu"),
    BatchNormalization(),
    Dropout(0.4),
    Dense(num_classes, activation="softmax")
])

# Компиляция модели
model.compile(
    optimizer=AdamW(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Коллбэки
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6),
    ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)
]

# Обучение
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=50,
    callbacks=callbacks,
    class_weight=class_weights
)

# Оценка
test_loss, test_accuracy = model.evaluate(val_data)
print(f"Точность модели на валидационных данных: {test_accuracy:.2f}")

# Метрики
val_predictions = model.predict(val_data)
val_pred_classes = np.argmax(val_predictions, axis=1)
val_true_classes = val_data.classes
print(classification_report(val_true_classes, val_pred_classes, target_names=val_data.class_indices.keys()))

# Визуализация
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Accuracy')
plt.legend()
plt.show()

# Сохранение модели
model.save("image_classifier.keras")
print("Модель сохранена как image_classifier.keras")