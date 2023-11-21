#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

bool dragging = false;
Point startPoint, endPoint;

void onMouse(int event, int x, int y, int flags, void* param) {
    Mat* image = reinterpret_cast<Mat*>(param);
    switch (event) {
    case EVENT_LBUTTONDOWN:
        dragging = true;
        startPoint = Point(x, y);
        break;

    case EVENT_LBUTTONUP:
        dragging = false;
        endPoint = Point(x, y);
        Rect roi(startPoint, endPoint);
        (*image)(roi) = 255 - (*image)(roi);
        imshow("Image", *image);
        break;
    }
}

int main() {
    Mat image = imread("C:/Users/Chan's Victus/Documents/class/Project/image/lenna.jpg", 1);
    if (image.empty()) {
        cerr << "Error loading the image" << endl;
        return -1;
    }

    namedWindow("Image");
    setMouseCallback("Image", onMouse, &image);

    imshow("Image", image);
    waitKey(0);

    return 0;
}
