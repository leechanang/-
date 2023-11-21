#include <opencv2/opencv.hpp>

using namespace cv;

int main() {
    // �̹����� �б�
    Mat image = imread("C:/Users/Chan's Victus/Documents/class/Project/image/keyboard.bmp", IMREAD_GRAYSCALE);

    if (image.empty()) {
        std::cerr << "�̹����� ���� �� �����ϴ�." << std::endl;
        return -1;
    }

    // �̹��� ����ȭ
    Mat binaryImage;
    threshold(image, binaryImage, 128, 255, THRESH_BINARY);

    // ���̺�
    Mat labeledImage, stats, centroids;
    int nLabels = connectedComponentsWithStats(binaryImage, labeledImage, stats, centroids);

    // ���� �̹����� ������ �׸���
    Mat contourImage = image.clone();
    std::vector<std::vector<Point>> contours;
    findContours(binaryImage, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (const auto& contour : contours) {
        Rect rect = boundingRect(contour);
        rectangle(contourImage, rect, Scalar(255, 0, 0), 2);
    }

    // ��� ǥ��
    
    imshow("Origin", image);

    namedWindow("Contours", WINDOW_NORMAL);
    imshow("Contours", contourImage);

    waitKey(0);

    return 0;
}
