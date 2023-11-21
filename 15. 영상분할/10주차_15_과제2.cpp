#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main() {
    // �̹����� �б�
    Mat image = imread("C:/Users/Chan's Victus/Documents/class/Project/image/shape.bmp");

    if (image.empty()) {
        cerr << "�̹����� ���� �� �����ϴ�." << endl;
        return -1;
    }

    // �׷��̽����Ϸ� ��ȯ
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    // ���� ����
    Mat invertedImage = 255 - gray;


    // ����ȭ
    Mat binary;
    threshold(invertedImage, binary, 10, 255, THRESH_BINARY );

    //imshow("11", binary);

    // �������� ������ ���� ������ ����
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(binary, binary, MORPH_CLOSE, kernel, Point(-1, -1), 1);
  
    // ������ ����
    vector<vector<Point>> contours;
    findContours(binary, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

    // �׸� �׸���
    Mat resultImage = image.clone();
    for (size_t i = 0; i < contours.size(); i++) {
        // �ܰ����� ���� �ٻ� �ٰ��� ���ϱ�
        vector<Point> approxCurve;
        approxPolyDP(contours[i], approxCurve, arcLength(contours[i], true) * 0.02, true);

        // ���� ũ�� �̻��� �ٰ����� �׸���
        if (approxCurve.size() >= 3) {
            drawContours(resultImage, vector<vector<Point>>{contours[i]}, 0, Scalar(0, 0, 0), 2);
        }
    }

    // ��� ǥ��
 
    imshow("Original Image", image);

    imshow("Result Image", resultImage);

    waitKey(0);

    return 0;
}