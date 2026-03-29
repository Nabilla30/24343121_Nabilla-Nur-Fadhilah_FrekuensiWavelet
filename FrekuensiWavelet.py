import numpy as np
import cv2
import matplotlib.pyplot as plt
import pywt
import time

# ===============================
# 1. LOAD CITRA
# ===============================
img = cv2.imread('Gunung.jpg', 0)

# ===============================
# 2. TAMBAH NOISE PERIODIK
# ===============================
rows, cols = img.shape
x = np.arange(cols)
y = np.arange(rows)
X, Y = np.meshgrid(x, y)

noise = 50 * np.sin(2 * np.pi * X / 20)
noisy_img = img + noise
noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title('Citra Asli')

plt.subplot(1,2,2)
plt.imshow(noisy_img, cmap='gray')
plt.title('Citra Noise Periodik')
plt.show()

# ===============================
# 3. FFT + SPEKTRUM
# ===============================
start_fft = time.time()

f = np.fft.fft2(noisy_img)
fshift = np.fft.fftshift(f)

magnitude = 20 * np.log(np.abs(fshift) + 1)
phase = np.angle(fshift)

end_fft = time.time()
fft_time = end_fft - start_fft

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.imshow(magnitude, cmap='gray')
plt.title('Magnitude Spectrum')

plt.subplot(1,2,2)
plt.imshow(phase, cmap='gray')
plt.title('Phase Spectrum')
plt.show()

# ===============================
# 4. REKONSTRUKSI
# ===============================
mag_only = np.abs(fshift)
img_mag = np.fft.ifft2(np.fft.ifftshift(mag_only))

phase_only = np.exp(1j * phase)
img_phase = np.fft.ifft2(np.fft.ifftshift(phase_only))

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(np.abs(img_mag), cmap='gray')
plt.title('Rekonstruksi Magnitude')

plt.subplot(1,2,2)
plt.imshow(np.abs(img_phase), cmap='gray')
plt.title('Rekonstruksi Phase')
plt.show()

# ===============================
# 5. FILTER IDEAL
# ===============================
def ideal_lowpass(shape, cutoff):
    rows, cols = shape
    mask = np.zeros((rows, cols))
    center = (rows//2, cols//2)
    for i in range(rows):
        for j in range(cols):
            d = np.sqrt((i-center[0])**2 + (j-center[1])**2)
            if d < cutoff:
                mask[i,j] = 1
    return mask

def ideal_highpass(shape, cutoff):
    return 1 - ideal_lowpass(shape, cutoff)

mask_lp = ideal_lowpass(img.shape, 30)
mask_hp = ideal_highpass(img.shape, 30)

img_lp = np.fft.ifft2(np.fft.ifftshift(fshift * mask_lp))
img_hp = np.fft.ifft2(np.fft.ifftshift(fshift * mask_hp))

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(np.abs(img_lp), cmap='gray')
plt.title('Ideal Lowpass')

plt.subplot(1,2,2)
plt.imshow(np.abs(img_hp), cmap='gray')
plt.title('Ideal Highpass')
plt.show()

# ===============================
# 6. FILTER GAUSSIAN
# ===============================
def gaussian_lowpass(shape, cutoff):
    rows, cols = shape
    mask = np.zeros((rows, cols))
    center = (rows//2, cols//2)
    for i in range(rows):
        for j in range(cols):
            d = (i-center[0])**2 + (j-center[1])**2
            mask[i,j] = np.exp(-d/(2*(cutoff**2)))
    return mask

def gaussian_highpass(shape, cutoff):
    return 1 - gaussian_lowpass(shape, cutoff)

mask_glp = gaussian_lowpass(img.shape, 30)
mask_ghp = gaussian_highpass(img.shape, 30)

img_glp = np.fft.ifft2(np.fft.ifftshift(fshift * mask_glp))
img_ghp = np.fft.ifft2(np.fft.ifftshift(fshift * mask_ghp))

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(np.abs(img_glp), cmap='gray')
plt.title('Gaussian Lowpass')

plt.subplot(1,2,2)
plt.imshow(np.abs(img_ghp), cmap='gray')
plt.title('Gaussian Highpass')
plt.show()

# ===============================
# 7. NOTCH FILTER (NOISE PERIODIK)
# ===============================
mask = np.ones_like(fshift)

# titik noise (sesuaikan dari spektrum)
mask[120:140, 120:140] = 0
mask[120:140, 180:200] = 0

img_notch = np.fft.ifft2(np.fft.ifftshift(fshift * mask))

plt.imshow(np.abs(img_notch), cmap='gray')
plt.title('Notch Filter')
plt.show()

# ===============================
# 8. WAVELET (2 LEVEL)
# ===============================
start_wavelet = time.time()

coeffs = pywt.wavedec2(img, 'db4', level=2)

end_wavelet = time.time()
wavelet_time = end_wavelet - start_wavelet

cA, (cH, cV, cD), (cH2, cV2, cD2) = coeffs

plt.figure(figsize=(10,5))
plt.subplot(2,2,1)
plt.imshow(cA, cmap='gray')
plt.title('Approximation')

plt.subplot(2,2,2)
plt.imshow(cH, cmap='gray')
plt.title('Horizontal')

plt.subplot(2,2,3)
plt.imshow(cV, cmap='gray')
plt.title('Vertical')

plt.subplot(2,2,4)
plt.imshow(cD, cmap='gray')
plt.title('Diagonal')
plt.show()

# ===============================
# 9. REKONSTRUKSI WAVELET
# ===============================
coeffs_filtered = [cA, (None, None, None), (None, None, None)]
reconstructed = pywt.waverec2(coeffs_filtered, 'db4')

plt.imshow(reconstructed, cmap='gray')
plt.title('Rekonstruksi Wavelet')
plt.show()

# ===============================
# 10. PSNR & WAKTU KOMPUTASI
# ===============================
def psnr(original, processed):
    mse = np.mean((original - processed) ** 2)
    return 20 * np.log10(255.0 / np.sqrt(mse))

psnr_fft = psnr(img, np.abs(img_notch))
psnr_wavelet = psnr(img, reconstructed)

print("===== HASIL ANALISIS =====")
print("PSNR FFT (Notch):", psnr_fft)
print("PSNR Wavelet:", psnr_wavelet)
print("Waktu FFT:", fft_time)
print("Waktu Wavelet:", wavelet_time)