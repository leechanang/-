import cv2
import numpy as np

# Lena 이미지를 컬러 영상으로 읽기
lena_color = cv2.imread('../Data/Lena.png', cv2.IMREAD_COLOR)

# 영상을 흑백으로 변환
lena_gray = cv2.cvtColor(lena_color, cv2.COLOR_BGR2GRAY)

# 흑백으로 변환된 영상에 Histogram Equalization 적용
lena_equalized = cv2.equalizeHist(lena_gray)

# 흑백으로 변환된 영상에 Gamma Correction 적용
gamma = 0.5
lena_gray = lena_gray.astype(np.float32) / 255
lena_gamma_corrected = np.power(lena_gray, gamma)

# 영상을 HSV 컬러 스페이스로 변환
lena_hsv = cv2.cvtColor(lena_color, cv2.COLOR_BGR2HSV)

# HSV 값을 0에서 255로 정규화
h, s, v = cv2.split(lena_hsv)
h_norm = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX)
s_norm = cv2.normalize(s, None, 0, 255, cv2.NORM_MINMAX)
v_norm = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)

# H 채널에 Median Filter 적용
h_median_filtered = cv2.medianBlur(h_norm, 7)

# S 채널에 Gaussian Filter 적용
s_gaussian_filtered = cv2.GaussianBlur(s_norm, (7, 7), 0)

# V 채널에 Bilateral Filter 적용
v_bilateral_filtered = cv2.bilateralFilter(v_norm, -1, 0.3, 10)

# 결과 이미지를 화면에 출력
cv2.imshow('Lena Color', lena_color)
cv2.imshow('Lena Gray', lena_gray)
cv2.imshow('Lena Equalized', lena_equalized)
cv2.imshow('Lena Gamma Corrected', lena_gamma_corrected)
cv2.imshow('H Channel Median Filtered', h_median_filtered)
cv2.imshow('S Channel Gaussian Filtered', s_gaussian_filtered)
cv2.imshow('V Channel Bilateral Filtered', v_bilateral_filtered)

cv2.waitKey(0)
cv2.destroyAllWindows()
