import cv2
import numpy as np

def extract_object(img):
    """
    주어진 이미지에서 물체를 추출하여 반환합니다.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    return img[y:y+h, x:x+w]

def match_brightness(images):
    # 이미지의 평균 밝기값 계산
    avg_brightness = np.mean([np.mean(img) for img in images])

    # 각 이미지의 밝기 조절
    adjusted_images = []
    for img in images:
        # 현재 이미지의 평균 밝기값 계산
        img_brightness = np.mean(img)
        # 평균 밝기값에 따라 목표 밝기값에 대한 보정 계수 계산
        adjustment_factor = avg_brightness / img_brightness
        # 이미지에 보정 계수를 적용하여 밝기 보정
        adjusted_image = cv2.convertScaleAbs(img, alpha=adjustment_factor, beta=0)
        adjusted_images.append(adjusted_image)

    return adjusted_images

def combine_images(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # 좋은 매칭점 선별
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 매칭점을 이용하여 변환 행렬 계산 (LMedS)
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)

    # 변환 행렬을 이용하여 이미지 이동
    h, w = img2.shape[:2]
    result = cv2.warpAffine(img1, M, (w * 2, h * 2))
    result[0:h, 0:w] = img2

    return result

# 이미지 로드 및 자르기
image = cv2.imread('../data/battery.bmp')
height, width, _ = image.shape
cut_height = height // 6
cut_image = image[:-cut_height]

# 이미지 분할
segments = [cut_image[:, i*(width//4):(i+1)*(width//4)] for i in range(4)]
segments[1] = cv2.flip(segments[1], 1)
segments[2] = cv2.flip(segments[2], 1)

# 이미지 저장 경로
segment_paths = [f'../data/b/segment_{i+1}.jpg' for i in range(4)]
object_paths = [f'../data/b/object_{i+1}.jpg' for i in range(4)]

# 분할된 이미지 저장
for i, segment in enumerate(segments):
    cv2.imwrite(segment_paths[i], segment)

# 이미지 로드 및 물체 추출하여 저장
for segment_path, object_path in zip(segment_paths, object_paths):
    img = cv2.imread(segment_path)
    object_image = extract_object(img)
    cv2.imwrite(object_path, object_image)

# 이미지 밝기 조정
images = [cv2.imread(path) for path in object_paths]
bright_adjusted_images = match_brightness(images)

# 이미지 합치기
result_14 = combine_images(bright_adjusted_images[3], bright_adjusted_images[0])
result_23 = combine_images(bright_adjusted_images[1], bright_adjusted_images[2])

# 이미지 합치고 물체 추출하기
result_14_object = extract_object(result_14)
result_23_object = extract_object(result_23)

resize_result_14_object = cv2.resize(result_14_object, None, fx=1/3, fy=1/3)
resize_result_23_object = cv2.resize(result_23_object, None, fx=1/3, fy=1/3)

cv2.imshow("result_14_object", resize_result_14_object)
cv2.imshow("result_23_object", resize_result_23_object)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 결과 이미지 저장
cv2.imwrite("../data/b/result_14_object.jpg", result_14_object)
cv2.imwrite("../data/b/result_23_object.jpg", result_23_object)

# 이미지 합치기
result_1423 = combine_images(result_23_object, result_14_object)
result_1423_object = extract_object(result_1423)
# cv2.imshow("3", result_1423)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 이미지 크기 조절
resize_result_1423_object = cv2.resize(result_1423_object, None, fx=1/3, fy=1/3)

# 결과 이미지 저장
cv2.imwrite("../data/b/panorama_resized_LMedS_custom_order_with_objects_contour.jpg", result_1423_object)

# 결과 이미지 출력
cv2.imshow("Panorama (Resized) with LMedS (Custom Order with Objects)", resize_result_1423_object)
cv2.waitKey(0)
cv2.destroyAllWindows()
