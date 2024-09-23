import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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

# Histogram for angular velocity
finite_values = data['angular_velocity'][np.isfinite(data['angular_velocity'])]
num_bins = 25
samples_per_bin = 650
hist, bins = np.histogram(finite_values, num_bins)
center = (bins[:-1] + bins[1:]) * 0.5
plt.bar(center, hist, width=0.05)
plt.plot((np.min(finite_values), np.max(finite_values)), (samples_per_bin, samples_per_bin))
plt.show()

# Remove samples to balance the dataset
remove_list = []
for j in range(num_bins - 1):
    list_ = [i for i in range(len(data['angular_velocity'])) if bins[j] <= data['angular_velocity'].iloc[i] < bins[j + 1]]
    list_ = shuffle(list_)
    list_ = list_[samples_per_bin:]
    remove_list.extend(list_)

remove_list = remove_list[:len(data)]
data.drop(data.index[remove_list], inplace=True)

# Load images and angular velocity
def load_img_angular_velocity(datadir, df):
    image_path = []
    angular_velocity = []
    for i in range(len(df)):
        indexed_data = df.iloc[i]
        camera_choice = random.choice(['center', 'left', 'right'])  # Augment with different camera angles
        image_file = indexed_data[camera_choice]
        correction = 0.0  # Steering correction
        if camera_choice == 'left':
            correction = 0.25  # Adjust for left camera
        elif camera_choice == 'right':
            correction = -0.25  # Adjust for right camera

        image_path.append(os.path.join(datadir, image_file.strip()))
        angular_velocity.append(float(indexed_data['angular_velocity']) + correction)  # Correct steering
    return np.asarray(image_path), np.asarray(angular_velocity)

image_path, angular_velocities = load_img_angular_velocity('D:/Github/dataset_ros2/data', data)

# Split dataset
x_train, x_valid, y_train, y_valid = train_test_split(image_path, angular_velocities, test_size=0.2, random_state=6)
print('Training Samples: {}\nValid Samples: {}'.format(len(x_train), len(x_valid)))

# Preprocess images
def img_preprocess(img):
    img = mpimg.imread(img)
    height = img.shape[0]
    img = img[height - 135:height - 60, :, :]  # Mengambil bagian bawah gambar
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (1, 1), 0)
    img = cv2.resize(img, (200, 66))  # Mengubah ukuran gambar menjadi 200x66
    img = img / 255.0
    return img

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

model = nvidia_model()
print(model.summary())

# Train the model
history = model.fit(x_train, y_train, epochs=40, validation_data=(x_valid, y_valid), batch_size=100, verbose=1, shuffle=True)

# Plot training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# Save the model
model.save('model40.h5')
print('Model saved')


# Prediksi pada data validasi
y_pred = model.predict(x_valid)

# Evaluasi menggunakan MSE dan MAE
mse = mean_squared_error(y_valid, y_pred)
mae = mean_absolute_error(y_valid, y_pred)

print(f"Mean Squared Error (MSE) on validation set: {mse}")
print(f"Mean Absolute Error (MAE) on validation set: {mae}")