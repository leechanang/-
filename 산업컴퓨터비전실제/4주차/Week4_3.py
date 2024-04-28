import cv2
import matplotlib.pyplot as plt

image = cv2.imread('../data/Lena.png', 0)
_, binary = cv2.threshold(image, -1, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


eroded = cv2.morphologyEx(binary, cv2.MORPH_ERODE, (3, 3), iterations=10)
dilated = cv2.morphologyEx(binary, cv2.MORPH_DILATE, (3, 3), iterations=10)

opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                          iterations=5)
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                          iterations=5)

grad = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

# 가우시안 블러 필터 적용 (노이즈 제거를 위해)
blur = cv2.GaussianBlur(image, (5, 5), 0)

canny = cv2.Canny(blur, 30, 150)  # 최소 임계값: 50, 최대 임계값: 150


plt.figure(figsize=(10, 10))
plt.subplot(331)
plt.axis('off')
plt.title('binary')
plt.imshow(binary, cmap='gray')
plt.subplot(332)
plt.axis('off')
plt.title('erode 10 times')
plt.imshow(eroded, cmap='gray')
plt.subplot(333)
plt.axis('off')
plt.title('dilate 10 times')
plt.imshow(dilated, cmap='gray')
plt.subplot(334)
plt.axis('off')
plt.title('open 5 times')
plt.imshow(opened, cmap='gray')
plt.subplot(335)
plt.axis('off')
plt.title('close 5 times')
plt.imshow(closed, cmap='gray')
plt.subplot(336)
plt.axis('off')
plt.title('gradient')
plt.imshow(grad, cmap='gray')
plt.subplot(337)
plt.axis('off')
plt.title('original')
plt.imshow(image, cmap='gray')
plt.subplot(338)
plt.axis('off')
plt.title('canny')
plt.imshow(canny, cmap='gray')
plt.tight_layout()
plt.show()