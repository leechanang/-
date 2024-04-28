import cv2
import numpy as np

# 이미지 불러오기
image_path = '../data/Lena.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 사용자 입력 받기
binary_method = input("이진화 방법을 선택하세요 (otsu 또는 adaptive median): ")
morphology_operation = input("적용할 모폴로지 연산을 선택하세요 (erosion, dilation, opening, closing): ")
iterations = int(input("모폴로지 연산을 적용할 횟수를 입력하세요: "))

# 이진화
if binary_method == 'otsu':
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
elif binary_method == 'adaptive median':
    binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# 모폴로지 연산 수행
kernel = np.ones((3,3),np.uint8)
if morphology_operation == 'erosion':
    result = cv2.erode(binary_image, kernel, iterations=iterations)
elif morphology_operation == 'dilation':
    result = cv2.dilate(binary_image, kernel, iterations=iterations)
elif morphology_operation == 'opening':
    result = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=iterations)
elif morphology_operation == 'closing':
    result = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=iterations)

# 결과 출력
cv2.imshow('Original Image', image)
cv2.imshow('Binary Image', binary_image)
cv2.imshow('Morphology Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
