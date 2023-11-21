#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    Mat src = imread("C:/Users/Chan's Victus/Documents/class/Project/image/hough_test3.jpg", 1);  // 원본 이미지를 컬러로 읽음
    if (src.empty()) {
        cout << "can not open " << endl;
        return -1;
    }

    Mat dst, cdst;
    Canny(src, dst, 100, 200);


    cvtColor(dst, cdst, COLOR_GRAY2BGR);

    vector<Vec4i> lines;
    HoughLinesP(dst, lines, 1, CV_PI / 90, 100, 10, 1);

    for (size_t i = 0; i < lines.size(); i++) {
        Vec4i l = lines[i];
        line(src, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 255, 0), 2, LINE_AA);  // 원본 이미지에 직선 그리기
    }

    imshow("source", src);  // 원본 이미지에 그린 직선이 표시된 이미지 출력
    waitKey();
    return 0;
}
