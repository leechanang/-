#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main() {
    Mat img = imread("C:/Users/Chan's Victus/Documents/class/Project/image/lenna.jpg", IMREAD_GRAYSCALE);
    if (img.empty()) {
        cerr << "Error loading the image" << endl;
        return -1;
    }

    // 수직 및 수평방향 투영
    Mat vert_proj, horiz_proj;
    reduce(img, vert_proj, 0, REDUCE_SUM, CV_32S);
    reduce(img, horiz_proj, 1, REDUCE_SUM, CV_32S);

    // 결과를 출력하기 위한 이미지
    Mat vert_display = Mat::ones(256, img.cols, CV_8U) * 255;
    Mat horiz_display = Mat::ones(img.rows, 256, CV_8U) * 255;

    // 투영 결과를 그래프로 그림
    for (int x = 0; x < img.cols; x++) {
        line(vert_display, Point(x, 255), Point(x, 255 - vert_proj.at<int>(x) / 1000), Scalar(0), 1);
    }

    for (int y = 0; y < img.rows; y++) {
        line(horiz_display, Point(0, y), Point(horiz_proj.at<int>(y) / 1000, y), Scalar(0), 1);
    }

    imshow("Original Image", img);
    imshow("Vertical Projection", vert_display);
    imshow("Horizontal Projection", horiz_display);
    waitKey(0);

    return 0;
}
