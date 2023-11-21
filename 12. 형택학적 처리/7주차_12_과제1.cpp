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

    Mat dilated_image;
    morphologyEx(input_image, dilated_image, MORPH_DILATE,kernel);

    // Display the result
    const int rate = 30;
    resize(input_image, input_image, Size(), rate, rate, INTER_NEAREST);
    imshow("Original", input_image);
    resize(dilated_image, dilated_image, Size(), rate, rate, INTER_NEAREST);
    imshow("dilate", dilated_image);
    waitKey(0);
    return 0;

    return 0;
}