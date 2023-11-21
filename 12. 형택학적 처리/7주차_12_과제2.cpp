#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main()
{
    Mat input_image = (Mat_<uchar>(6, 8) <<
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 255, 255, 0, 0, 0,
        0, 255, 255, 255, 255, 255, 255, 0,
        0, 0, 255, 255, 255, 0, 255, 0,
        0, 255, 255, 255, 255, 255, 255, 0,
        0, 0, 0, 0, 0, 0, 0, 0);

    Mat kernel = (Mat_<uchar>(3, 3) <<
        0, 1, 0,
        1, 1, 1,
        0, 1, 0);

    Mat eroded_image;
    morphologyEx(input_image, eroded_image, MORPH_ERODE, kernel);

    // Display the result
    const int rate = 30;
    resize(input_image, input_image, Size(), rate, rate, INTER_NEAREST);
    imshow("Original", input_image);
    resize(eroded_image, eroded_image, Size(), rate, rate, INTER_NEAREST);
    imshow("dilate", eroded_image);
    waitKey(0);
    return 0;
}