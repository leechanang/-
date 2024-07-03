#include <opencv2/opencv.hpp>


using namespace cv;
using namespace std;

vector<Point2f> src_points;
vector<Point2f> dst_points;

void click_event(int event, int x, int y, int flags, void* userdata) {
    if (event == EVENT_LBUTTONDOWN) {
        if (src_points.size() < 4) {
            src_points.push_back(Point2f(x, y));
            circle(*((Mat*)userdata), Point(x, y), 5, Scalar(0, 0, 255), -1);
        }
        else if (dst_points.size() < 4) {
            dst_points.push_back(Point2f(x, y));
            circle(*((Mat*)userdata), Point(x, y), 5, Scalar(255, 0, 0), -1);
        }
        imshow("Image", *((Mat*)userdata));
    }
}

int main() {
    Mat img = imread("C:/Users/Chan's Victus/Documents/class/Project/image/book.jpg");
    Mat img_clone = img.clone();

    namedWindow("Image");
    setMouseCallback("Image", click_event, &img_clone);

    imshow("Image", img_clone);
    waitKey(0);

    if (src_points.size() == 4 && dst_points.size() == 4) {
        Mat warp_mat = getPerspectiveTransform(src_points, dst_points);
        Mat result;
        warpPerspective(img, result, warp_mat, img.size());
        imshow("Transformed", result);
        waitKey(0);
    }
    else {
        cout << "정확히 4개의 출발점과 4개의 목적지를 선택해야 합니다" << endl;
    }

    return 0;
}
