import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, Dropout, Flatten, Dense
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
            correction = -0.15 # Adjust for left camera
        elif camera_choice == 'right':
            correction = 0.15  # Adjust for right camera
            
        image_path.append(os.path.join(datadir, image_file.strip()))
        angular_velocity.append(float(indexed_data['angular_velocity']) + correction)
    
    return np.asarray(image_path), np.asarray(angular_velocity)

image_path, angular_velocities = load_img_angular_velocity('D:/Github/dataset_ros2/data', data)

# Split dataset
x_train, x_valid, y_train, y_valid = train_test_split(image_path, angular_velocities, test_size=0.2, random_state=6)

# Preprocess images
def img_preprocess(img):
    img = mpimg.imread(img)
    height = img.shape[0]
    img = img[height - 135:height - 60, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (1, 1), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255.0
    return img

x_train = np.array([img_preprocess(img) for img in x_train])
x_valid = np.array([img_preprocess(img) for img in x_valid])

def nvidia_model():
    model = Sequential()
    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), input_shape=(66, 200, 3), activation='relu'))
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='elu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='elu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))  # Output layer for angular velocity
    optimizer = Adam(learning_rate=1e-3)
    model.compile(loss='mse', optimizer=optimizer)
    return model

model = nvidia_model()
print(model.summary())

# Train the model
history = model.fit(x_train, y_train, epochs=30, validation_data=(x_valid, y_valid), batch_size=100, verbose=1, shuffle=True)

# Save the model
model.save('model_mobil_besar.h5')
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