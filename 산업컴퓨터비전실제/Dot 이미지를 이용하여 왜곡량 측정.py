import cv2
import numpy as np
import math

# 이미지 파일 경로
image_path = "../data/dot.bmp"

# DOT의 실제 크기 (예: mm 단위)
dot_size_mm = 0.25

# 이미지 sensor의 pixel size (예: um 단위)
pixel_size_um = 5.5

# 이미지 불러오기
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 이미지 반전
inverted_gray = cv2.bitwise_not(gray)

# 이미지 전처리: 가우시안 블러 및 이진화
blur = cv2.GaussianBlur(inverted_gray, (5, 5), 0)
_, binary = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)

# 원 검출을 위한 컨투어 검출
_, contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 컨투어로부터 원 추출
circles = []
for contour in contours:
    area = cv2.contourArea(contour)
    if 100 < area < 5000:  # 원의 크기 범위에 따라 조정 가능
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * math.pi * (area / (perimeter * perimeter))
        if 0.7 < circularity < 1.3:  # 원 형태의 컨투어만 추출
            (x, y), radius = cv2.minEnclosingCircle(contour)
            circles.append((int(x), int(y), int(radius)))

# 이미지 중심 좌표
image_center = np.array([image.shape[1] // 2, image.shape[0] // 2])

# 중심 원 찾기
min_distance = float('inf')
center_circle = None
for circle in circles:
    circle_center = np.array([circle[0], circle[1]])
    distance_to_center = np.linalg.norm(circle_center - image_center)
    if distance_to_center < min_distance:
        min_distance = distance_to_center
        center_circle = circle

# 코너 원 찾기
corner_circles = []
corner_regions = [(0, 0), (image.shape[1], 0), (0, image.shape[0]), (image.shape[1], image.shape[0])]
for corner_x, corner_y in corner_regions:
    min_distance_to_corner = float('inf')
    corner_circle = None
    for circle in circles:
        distance_to_corner = np.linalg.norm(np.array([circle[0], circle[1]]) - np.array([corner_x, corner_y]))
        if distance_to_corner < min_distance_to_corner:
            min_distance_to_corner = distance_to_corner
            corner_circle = circle
    if corner_circle is not None:
        corner_circles.append(corner_circle)

# 결과 출력
if center_circle is not None:
    # 거리 측정 (중심 원과 가장 가까운 코너 원 사이의 거리)
    min_distance_to_center = float('inf')
    for circle in corner_circles:
        distance_to_center = np.linalg.norm(np.array([circle[0], circle[1]]) - np.array(center_circle[:2]))
        if distance_to_center < min_distance_to_center:
            min_distance_to_center = distance_to_center

    # 오른쪽에 있는 원 찾기
    right_circle = None
    min_distance_to_right = float('inf')
    for circle in circles:
        if circle[0] > center_circle[0] and abs(circle[1] - center_circle[1]) < center_circle[2]:
            distance_to_right = np.linalg.norm(np.array([circle[0], circle[1]]) - np.array(center_circle[:2]))
            if distance_to_right < min_distance_to_right:
                min_distance_to_right = distance_to_right
                right_circle = circle

    # 결과 출력
    if right_circle is not None:
        # 중심 원과 오른쪽 원 사이의 거리 계산 (픽셀)
        distance_to_right_px = np.linalg.norm(np.array(right_circle[:2]) - np.array(center_circle[:2]))

        # 픽셀을 mm로 변환한 거리 계산
        distance_to_right_mm = distance_to_right_px * pixel_size_um / 1000  # 픽셀을 mm로 변환

        # print("중심 원과 오른쪽 원 사이의 거리 (픽셀):", distance_to_right_px)
        # print("중심 원과 오른쪽 원 사이의 거리 (mm):", distance_to_right_mm)
    else:
        print("오른쪽에 있는 원을 찾을 수 없습니다.")

    # 거리 측정 (픽셀)
    distance_px = min_distance_to_center

    # 픽셀을 mm로 변환한 거리 계산
    distance_mm = distance_px * pixel_size_um / 1000  # 픽셀을 mm로 변환

    # 중심 원과 코너 원 사이의 x축 방향 원의 수 계산
    num_circles_x = int(abs((center_circle[0] - corner_circles[0][0]) / (2 * center_circle[2]))/2)

    # 중심 원과 코너 원 사이의 y축 방향 원의 수 계산
    num_circles_y = int(abs((center_circle[1] - corner_circles[0][1]) / (2 * center_circle[2]))/2)

    # 배율 계산
    Mag = distance_to_right_mm / (dot_size_mm * 2)

    # 왜곡 치수
    real_mm = math.sqrt(((dot_size_mm * num_circles_x * 2)**2) + ((dot_size_mm * num_circles_y * 2)**2)) / Mag
    distortion_ratio = ((real_mm - distance_mm) / real_mm) * 100


    # 결과 출력
    print("Mag:", Mag, "X")
    print("측정 거리 (mm):", distance_mm)
    print("실제 거리 (mm)", real_mm)
    print("왜곡 계수 (%):", distortion_ratio)
    # print("x축 원 갯수:", num_circles_x)
    # print("y축 원 갯수:", num_circles_y)


    # 원 시각화
    cv2.circle(image, (center_circle[0], center_circle[1]), center_circle[2], (0, 0, 255), 4)  # 중심 원
    for circle in corner_circles:
        cv2.circle(image, (circle[0], circle[1]), circle[2], (255, 0, 0), 4)  # 코너 원

    # 이미지의 절반 크기로 축소
    resized_image = cv2.resize(image, (0, 0), fx=0.3, fy=0.3)

    # 이미지 표시
    cv2.imshow("Detected circles", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("중심 원을 찾을 수 없습니다.")
