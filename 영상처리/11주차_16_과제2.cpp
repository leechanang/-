#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main() {
    // ������ ����� ũ��
    int width = 5;
    int height = 5;

    // ������
    int radius = 2;

    // ������ ��� �ʱ�ȭ
    Mat accumulator = Mat::zeros(width, height, CV_32F);

    // �־��� ����
    vector<Point> points = { Point(0, 2), Point(2, 0), Point(4, 2) };

    // ���� ��ȯ ����
    for (const Point& point : points) {
        for (int a = 0; a < width; ++a) {
            for (int b = 0; b < height; ++b) {
                int x = point.x;
                int y = point.y;
                if ((x - a) * (x - a) + (y - b) * (y - b) == radius * radius) {
                    accumulator.at<float>(a, b) += 1;
                }
            }
        }
    }

    // ��� ���
    cout << "Hough Transform Result (10x10 matrix):" << endl;
    for (int i = 0; i < 11; ++i) {
        for (int j = 0; j < 11; ++j) {
            // 10x10 ��ķ� ũ�� ����
            int value = static_cast<int>(accumulator.at<float>(i * width / 11, j * height / 11));
            cout << value << "\t";
        }
        cout << endl;
    }

    return 0;
}
