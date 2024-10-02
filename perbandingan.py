import pandas as pd
import matplotlib.pyplot as plt

# Fungsi untuk membaca jalur dari file CSV
def read_path_from_csv(file_path):
    # Membaca data dari file CSV tanpa header
    data = pd.read_csv(file_path, header=None, names=['timestamp', 'x', 'y', 'z', 'color'])
    
    # Konversi kolom 'x' dan 'y' menjadi float
    data['x'] = pd.to_numeric(data['x'], errors='coerce')
    data['y'] = pd.to_numeric(data['y'], errors='coerce')
    
    return data['x'], data['y']

# Mengurangi jumlah titik yang dipetakan dengan downsampling
def downsample_data(x, y, factor=10):
    return x[::factor], y[::factor]

# Baca jalur teleoperation, otonom, dan path ketiga
teleop_x, teleop_y = read_path_from_csv('robot_path2.csv')
autonomous_x, autonomous_y = read_path_from_csv('robot_path.csv')
third_path_x, third_path_y = read_path_from_csv('robot_path3.csv')

# Cek ukuran data yang dibaca
print(f'Teleoperation Path Size: {len(teleop_x)}')
print(f'Autonomous Path Size: {len(autonomous_x)}')
print(f'Autonomous without dropout Size: {len(third_path_x)}')

# Downsampling data untuk visualisasi
teleop_x, teleop_y = downsample_data(teleop_x, teleop_y, factor=10)
autonomous_x, autonomous_y = downsample_data(autonomous_x, autonomous_y, factor=10)
third_path_x, third_path_y = downsample_data(third_path_x, third_path_y, factor=10)

# Visualisasi jalur
plt.figure(figsize=(10, 8))

if not teleop_x.empty and not teleop_y.empty:
    plt.plot(teleop_x, teleop_y, label='Teleoperation Path', color='blue', alpha=0.5)
else:
    print("Teleoperation Path is empty!")

if not autonomous_x.empty and not autonomous_y.empty:
    plt.plot(autonomous_x, autonomous_y, label='Autonomous Path', color='green', alpha=0.5)
else:
    print("Autonomous Path is empty!")

if not third_path_x.empty and not third_path_y.empty:
    plt.plot(third_path_x, third_path_y, label='Autonomous without dropout', color='red', alpha=0.5)
else:
    print("Third Path is empty!")

plt.title('Comparison of Teleoperation, Autonomous, and Third Paths (Downsampled)')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.grid()
plt.axis('equal')  # Menjaga skala sumbu agar seimbang
plt.show()
