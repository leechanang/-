import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 불러오기
image_path = '../data/Lena.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# DFT를 위한 함수 정의
def dft(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    return magnitude_spectrum

# 주파수 도메인으로 변환
dft_img = dft(image)

# 입력 받은 반지름
r1 = int(input("첫 번째 원의 반지름 입력: "))
r2 = int(input("두 번째 원의 반지름 입력: "))

# 중심 좌표 및 크기 계산
rows, cols = image.shape
center_row, center_col = rows // 2, cols // 2
x, y = np.ogrid[:rows, :cols]
mask1 = np.logical_and(np.sqrt((x - center_row)**2 + (y - center_col)**2) > r1,
                        np.sqrt((x - center_row)**2 + (y - center_col)**2) < r2)

# 필터링
dft_img_filtered = dft_img * mask1

# 역 DFT
f_ishift = np.fft.ifftshift(dft_img_filtered)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

# 결과 출력
plt.subplot(121),plt.imshow(image, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
plt.title('Band Pass Filtered Image'), plt.xticks([]), plt.yticks([])
plt.show()
