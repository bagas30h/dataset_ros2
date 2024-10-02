import numpy as np
import matplotlib.image as mpimg
import cv2
from keras.models import load_model
from sklearn.model_selection import train_test_split

# Load model dari iterasi pertama
model = load_model('dagger_model_iter1.h5')

# Global variables for training data
# Pastikan x_train, y_train, x_valid, y_valid sudah didefinisikan dengan benar
# Misalnya, ambil dari dataset awal yang sama
# Atau jika Anda sudah memiliki dataset baru, gunakan dataset tersebut

# Fungsi untuk preprocess gambar
def img_preprocess(img):
    img = mpimg.imread(img)
    height = img.shape[0]
    img = img[height - 135:height - 60, :, :]  # Crop image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (1, 1), 0)
    img = cv2.resize(img, (200, 66))  # Resize to 200x66
    img = img / 255.0
    return img

# Fungsi untuk menambah dataset baru dari demonstrasi manusia
def augment_dataset_with_human_feedback(new_images, new_angular_velocities):
    global x_train, y_train
    # Preprocess new images and augment dataset
    new_images_processed = np.array([img_preprocess(img) for img in new_images])
    x_train = np.concatenate((x_train, new_images_processed))
    y_train = np.concatenate((y_train, new_angular_velocities))
    print(f"Dataset diperbarui dengan {len(new_images)} contoh baru dari feedback manusia.")

# Simulasi robot untuk mengumpulkan feedback manusia
# Gantikan bagian ini dengan proses nyata pengumpulan data dari robot
new_images_from_feedback = ['path_to_image1.jpg', 'path_to_image2.jpg']  # Ganti dengan path sebenarnya
new_angular_velocities_from_feedback = [0.5, -0.2]  # Ganti dengan angular velocity sebenarnya

# Tambahkan data ke dataset
augment_dataset_with_human_feedback(new_images_from_feedback, new_angular_velocities_from_feedback)

# Split dataset untuk validasi setelah augmentasi
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=6)

# Retrain model dengan dataset baru
history = model.fit(x_train, y_train, epochs=20, validation_data=(x_valid, y_valid), batch_size=100, verbose=1, shuffle=True)

# Simpan model hasil DAgger
model.save('dagger_model_iter2.h5')
print('Model Iterasi Kedua disimpan')
