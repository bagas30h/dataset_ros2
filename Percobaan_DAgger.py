import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, Dropout, Flatten, Dense
from sklearn.model_selection import train_test_split
import cv2
import pandas as pd
import ntpath
import random

# Load data awal dari demonstrasi manusia
columns = ['right', 'center', 'left', 'linear_velocity', 'angular_velocity']
data = pd.read_csv(os.path.join('velocity_data.csv'), names=columns)

# Proses image paths
data['right'] = data['right'].apply(lambda x: ntpath.basename(x))
data['center'] = data['center'].apply(lambda x: ntpath.basename(x))
data['left'] = data['left'].apply(lambda x: ntpath.basename(x))

# Clean angular velocity
data['angular_velocity'] = pd.to_numeric(data['angular_velocity'], errors='coerce')
data.dropna(subset=['angular_velocity'], inplace=True)

# Load images and angular velocity
def load_img_angular_velocity(datadir, df):
    image_path = []
    angular_velocity = []
    for i in range(len(df)):
        indexed_data = df.iloc[i]
        camera_choice = 'center' if random.random() < 0.8 else random.choice(['left', 'right'])
        image_file = indexed_data[camera_choice]
        correction = 0.0
        if camera_choice == 'left':
            correction = 0.25
        elif camera_choice == 'right':
            correction = -0.25
            
        image_path.append(os.path.join(datadir, image_file.strip()))
        angular_velocity.append(float(indexed_data['angular_velocity']) + correction)
    
    return np.asarray(image_path), np.asarray(angular_velocity)

# Ambil data awal dari demonstrasi manusia
image_path, angular_velocities = load_img_angular_velocity('D:/Github/dataset_ros2/data', data)

# Split dataset awal
x_train, x_valid, y_train, y_valid = train_test_split(image_path, angular_velocities, test_size=0.2, random_state=6)

# Preprocess images
def img_preprocess(img):
    img = mpimg.imread(img)
    height = img.shape[0]
    img = img[height - 135:height - 60, :, :]  # Crop image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (1, 1), 0)
    img = cv2.resize(img, (200, 66))  # Resize to 200x66
    img = img / 255.0
    return img

# Preprocess training dan validasi data
x_train = np.array([img_preprocess(img) for img in x_train])
x_valid = np.array([img_preprocess(img) for img in x_valid])

# Define the model
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

# Inisialisasi model
model = nvidia_model()
print(model.summary())

# Train the model untuk iterasi pertama dengan data demonstrasi awal
history = model.fit(x_train, y_train, epochs=40, validation_data=(x_valid, y_valid), batch_size=100, verbose=1, shuffle=True)

# Simpan model pertama
model.save('dagger_model_iter1.h5')
print('Model Iterasi Pertama disimpan')
