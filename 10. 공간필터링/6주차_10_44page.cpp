#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;


Mat src, detected_edges,dst;

int lowThreshold = 0;
int max_lowThreshold = 255;
int kernel_size = 3;

static void CannyThreshold(int, void*)
{

    blur(src, detected_edges, Size(3, 3));
    Canny(detected_edges, detected_edges, lowThreshold,  lowThreshold*2, kernel_size);
    dst = Scalar::all(0);
    src.copyTo(dst, detected_edges);
    imshow("Image", src);
    imshow("Canny", dst);
}

int main()
{
    src = imread("C:/Users/Chan's Victus/Documents/class/Project/image/lenna.jpg", IMREAD_GRAYSCALE);
    if (src.empty()) { return -1; }
    dst.create(src.size(), src.type());
    namedWindow("Canny", WINDOW_AUTOSIZE);
    createTrackbar("Min Threshold:", "Canny", &lowThreshold, max_lowThreshold, CannyThreshold);
    CannyThreshold(0, 0);
    waitKey(0);
    return 0;
}
