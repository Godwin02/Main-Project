import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image



# seed_value = 42
# np.random.seed(seed_value)
# tf.random.set_seed(seed_value)

# Load your dataset (replace with your actual dataset loading code)
image_paths = ["C:/Users/user/PycharmProjects/pythonProject/Seminar/img.png",
               "C:/Users/user/PycharmProjects/pythonProject/Seminar/img_1.png",
               "C:/Users/user/PycharmProjects/pythonProject/Seminar/img_2.png",
               "C:/Users/user/PycharmProjects/pythonProject/Seminar/img_3.png",
               "C:/Users/user/PycharmProjects/pythonProject/Seminar/img_4.png",
               "C:/Users/user/PycharmProjects/pythonProject/Seminar/img_5.png",
               "C:/Users/user/PycharmProjects/pythonProject/Seminar/img_6.png",
               "C:/Users/user/PycharmProjects/pythonProject/Seminar/img_7.png",
               "C:/Users/user/PycharmProjects/pythonProject/Seminar/img_8.png",
               "C:/Users/user/PycharmProjects/pythonProject/Seminar/img_9.png",
               "C:/Users/user/PycharmProjects/pythonProject/Seminar/img_10.png",
               "C:/Users/user/PycharmProjects/pythonProject/Seminar/img_11.png",
               "C:/Users/user/PycharmProjects/pythonProject/Seminar/img_12.png",
               "C:/Users/user/PycharmProjects/pythonProject/Seminar/img_13.png",
               "C:/Users/user/PycharmProjects/pythonProject/Seminar/img_14.png",
               "C:/Users/user/PycharmProjects/pythonProject/Seminar/img_15.png",
               "C:/Users/user/PycharmProjects/pythonProject/Seminar/img_16.png",
               "C:/Users/user/PycharmProjects/pythonProject/Seminar/img_17.png",
               "C:/Users/user/PycharmProjects/pythonProject/Seminar/img_18.png",
               "C:/Users/user/PycharmProjects/pythonProject/Seminar/img_19.png",
               "C:/Users/user/PycharmProjects/pythonProject/Seminar/img_20.png",
               "C:/Users/user/PycharmProjects/pythonProject/Seminar/img_21.png",
               "C:/Users/user/PycharmProjects/pythonProject/Seminar/img_22.png",
               "C:/Users/user/PycharmProjects/pythonProject/Seminar/img_23.png",
               "C:/Users/user/PycharmProjects/pythonProject/Seminar/img_24.png",
               "C:/Users/user/PycharmProjects/pythonProject/Seminar/img_25.png",
               "C:/Users/user/PycharmProjects/pythonProject/Seminar/img_26.png",
               "C:/Users/user/PycharmProjects/pythonProject/Seminar/img_27.png",
               "C:/Users/user/PycharmProjects/pythonProject/Seminar/img_28.png",
               "C:/Users/user/PycharmProjects/pythonProject/Seminar/img_29.png",
               "C:/Users/user/PycharmProjects/pythonProject/Seminar/img_30.png",]
elevations = [7000, 7200, 6900,8849,8611,8586,8516,8485,8188,8167,8163,8126,8091,8080,7946,7937,7932,7863,7821,7816,7694,7665,7649,7577,7476,7434,7410,7294,7281,7276,7227]  # Corresponding elevations in feet

# Preprocess images
def preprocess_image(file_path):
    img = Image.open(file_path)
    img = img.resize((224, 224))  # Resize to a consistent size
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    return img_array

X_train_images = np.array([preprocess_image(path) for path in image_paths])
y_train_elevations = np.array(elevations)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train_images, y_train_elevations, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2,
                             height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, fill_mode='nearest')

# Define the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1, activation='linear'))  # Output layer for regression

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model with data augmentation
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    epochs=10, validation_data=(X_val, y_val))

# Evaluate the model on a test set (you'll need a separate test set)
# test_loss, test_mae = model.evaluate(X_test, y_test)

# Make predictions on new images (replace 'path/to/new_image.jpg' with the actual path)



# new_image_path = 'C:/Users/user/PycharmProjects/pythonProject/Seminar/img_14.png'
# new_image = np.expand_dims(preprocess_image(new_image_path), axis=0)
# prediction = model.predict(new_image)
# print("Predicted Elevation:", prediction[0])
