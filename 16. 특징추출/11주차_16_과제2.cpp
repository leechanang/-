#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main() {
    // 누적기 행렬의 크기
    int width = 5;
    int height = 5;

    // 반지름
    int radius = 2;

    // 누적기 행렬 초기화
    Mat accumulator = Mat::zeros(width, height, CV_32F);

    // 주어진 점들
    vector<Point> points = { Point(0, 2), Point(2, 0), Point(4, 2) };

    // 허프 변환 수행
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

    // 결과 출력
    cout << "Hough Transform Result (10x10 matrix):" << endl;
    for (int i = 0; i < 11; ++i) {
        for (int j = 0; j < 11; ++j) {
            // 10x10 행렬로 크기 조절
            int value = static_cast<int>(accumulator.at<float>(i * width / 11, j * height / 11));
            cout << value << "\t";
        }
        cout << endl;
    }

    return 0;
}
