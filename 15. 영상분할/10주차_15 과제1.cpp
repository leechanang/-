#include <opencv2/opencv.hpp>

using namespace cv;

int main() {
    // 이미지를 읽기
    Mat image = imread("C:/Users/Chan's Victus/Documents/class/Project/image/keyboard.bmp", IMREAD_GRAYSCALE);

    if (image.empty()) {
        std::cerr << "이미지를 읽을 수 없습니다." << std::endl;
        return -1;
    }

    // 이미지 이진화
    Mat binaryImage;
    threshold(image, binaryImage, 128, 255, THRESH_BINARY);

    // 레이블링
    Mat labeledImage, stats, centroids;
    int nLabels = connectedComponentsWithStats(binaryImage, labeledImage, stats, centroids);

    // 원본 이미지에 컨투어 그리기
    Mat contourImage = image.clone();
    std::vector<std::vector<Point>> contours;
    findContours(binaryImage, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (const auto& contour : contours) {
        Rect rect = boundingRect(contour);
        rectangle(contourImage, rect, Scalar(255, 0, 0), 2);
    }

    // 결과 표시
    
    imshow("Origin", image);

    namedWindow("Contours", WINDOW_NORMAL);
    imshow("Contours", contourImage);

    waitKey(0);

    return 0;
}
