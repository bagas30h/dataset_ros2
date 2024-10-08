import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, Dropout, Flatten, Dense, BatchNormalization, Activation, InputLayer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
import pandas as pd
import ntpath
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load data
columns = ['right', 'center', 'left', 'linear_velocity', 'angular_velocity']
data = pd.read_csv(os.path.join('velocity_data.csv'), names=columns)

# Process image paths
data['right'] = data['right'].apply(lambda x: ntpath.basename(x))
data['center'] = data['center'].apply(lambda x: ntpath.basename(x))
data['left'] = data['left'].apply(lambda x: ntpath.basename(x))

# Clean angular velocity
data['angular_velocity'] = pd.to_numeric(data['angular_velocity'], errors='coerce')
data.dropna(subset=['angular_velocity'], inplace=True)

# Load images and angular velocity with focus on the center camera
def load_img_angular_velocity(datadir, df):
    image_path = []
    angular_velocity = []
    for i in range(len(df)):
        indexed_data = df.iloc[i]
        camera_choice = 'center' if random.random() < 0.8 else random.choice(['left', 'right'])
        image_file = indexed_data[camera_choice]
        correction = 0.0
        if camera_choice == 'left':
            correction = 0.15  # Adjust for left camera
        elif camera_choice == 'right':
            correction = -0.15  # Adjust for right camera
            
        image_path.append(os.path.join(datadir, image_file.strip()))
        angular_velocity.append(float(indexed_data['angular_velocity']) + correction)
    
    return np.asarray(image_path), np.asarray(angular_velocity)

image_path, angular_velocities = load_img_angular_velocity('D:/Github/dataset_ros2/data', data)

# Split dataset
x_train, x_valid, y_train, y_valid = train_test_split(image_path, angular_velocities, test_size=0.2, random_state=6)

# Preprocess images with augmentation
def random_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + (0.5 - np.random.rand()) * 0.5
    hsv[:,:,2] = hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def add_random_shadow(image):
    top_y = 320 * np.random.rand()
    top_x = 0
    bot_x = 160
    bot_y = 320 * np.random.rand()
    shadow_mask = 0 * image[:,:,1]
    X_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][1]
    shadow_mask[((Y_m - top_y)*(bot_x - top_x) - (bot_y - top_y)*(X_m - top_x) >= 0)] = 1
    if np.random.rand() > 0.5:
        random_bright = .5
        cond1 = shadow_mask == 1
        image[:,:,0][cond1] = image[:,:,0][cond1] * random_bright
        image[:,:,1][cond1] = image[:,:,1][cond1] * random_bright
        image[:,:,2][cond1] = image[:,:,2][cond1] * random_bright
    return image

# Function to preprocess image, cutting from bottom to middle and visualizing the process
def img_preprocess(img, show_images=False):
    img = mpimg.imread(img)
    height = img.shape[0]
    
    # Step 1: Crop image from bottom to middle
    cropped_img = img[height // 2:height, :, :]
    if show_images:
        plt.imshow(cropped_img)
        plt.title("Cropped Image")
        plt.show()

    # Step 2: Convert to YUV
    yuv_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2YUV)
    if show_images:
        plt.imshow(yuv_img)
        plt.title("YUV Image")
        plt.show()

    # Step 3: Random brightness adjustment
    bright_img = random_brightness(yuv_img)
    if show_images:
        plt.imshow(bright_img)
        plt.title("Brightness Adjusted Image")
        plt.show()

    # Step 4: Add random shadow
    shadow_img = add_random_shadow(bright_img)
    if show_images:
        plt.imshow(shadow_img)
        plt.title("Shadow Added Image")
        plt.show()

    # Step 5: Apply Gaussian blur
    blur_img = cv2.GaussianBlur(shadow_img, (1, 1), 0)
    if show_images:
        plt.imshow(blur_img)
        plt.title("Blurred Image")
        plt.show()

    # Step 6: Resize image
    final_img = cv2.resize(blur_img, (200, 66))
    if show_images:
        plt.imshow(final_img)
        plt.title("Resized Image")
        plt.show()

    # Step 7: Normalize image
    final_img = final_img / 255.0

    return final_img

# Example to show preprocessing steps for a single image
img_preprocess(x_train[0], show_images=True)

# Apply preprocessing to all training and validation images
x_train = np.array([img_preprocess(img) for img in x_train])
x_valid = np.array([img_preprocess(img) for img in x_valid])

# Model CIL
def cil_model_improved():
    model = Sequential()

    # Input Layer
    model.add(InputLayer(input_shape=(66, 200, 3)))
    
    # CNN layers
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'))
    model.add(BatchNormalization())

    for _ in range(5):  # Reduce the number of layers for simplicity
        kernel_size = (3, 3)
        stride = (2, 2) if (_ % 2 == 0) else (1, 1)  # Adjust stride pattern
        model.add(Conv2D(64, kernel_size=kernel_size, strides=stride, padding='same', activation='relu'))
        model.add(BatchNormalization())

    model.add(Flatten())
    
    # FCN layers with dropout
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))  # Dropout to prevent overfitting
    model.add(BatchNormalization())

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    
    model.add(Dense(1))  # Output layer for angular velocity
    optimizer = Adam(learning_rate=1e-3)
    model.compile(loss='mse', optimizer=optimizer)
    
    return model

model = cil_model_improved()
print(model.summary())

# Train the model
history = model.fit(x_train, y_train, epochs=50, validation_data=(x_valid, y_valid), batch_size=100, verbose=1, shuffle=True)

# Save the model
model.save('model_mobil_bes.h5')
print('Model saved')

# Prediksi pada data validasi
y_pred = model.predict(x_valid)

# Evaluasi menggunakan MSE dan MAE
mse = mean_squared_error(y_valid, y_pred)
mae = mean_absolute_error(y_valid, y_pred)

print(f"Mean Squared Error (MSE) on validation set: {mse}")
print(f"Mean Absolute Error (MAE) on validation set: {mae}")

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

def visualize_predictions(y_true, y_pred):
    plt.figure(figsize=(15, 5))
    plt.plot(y_true, label='True Angular Velocity', color='blue')
    plt.plot(y_pred, label='Predicted Angular Velocity', color='red')
    plt.title('True vs Predicted Angular Velocity')
    plt.xlabel('Sample Index')
    plt.ylabel('Angular Velocity')
    plt.legend()
    plt.show()

visualize_predictions(y_valid, y_pred)
