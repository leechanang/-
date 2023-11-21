#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main()
{
    Mat input_image = (Mat_<uchar>(8, 8) <<
        0,0,0,0,0,255,0,0,
        0,255,255,255,0,0,0,0,
        0,255,255,255,0,0,255,255,
        0,255,255,255,0,255,255,255,
        255,255,255,0,0,255,0,255,
        0,0,0,0,255,255,255,255,
        0,0,0,0,255,255,255,0,
        0,0,0,0,0,0,0,0);

    Mat kernel = (Mat_<uchar>(3, 3) <<
        0,1,0,
        1,1,1,
        0,1,0);

    Mat open_image, close_image ;
    morphologyEx(input_image, open_image, MORPH_OPEN, kernel);
    morphologyEx(input_image, close_image, MORPH_CLOSE, kernel);

    // Display the result
    const int rate = 30;
    resize(input_image, input_image, Size(), rate, rate, INTER_NEAREST);
    imshow("Original", input_image);
    resize(open_image, open_image, Size(), rate, rate, INTER_NEAREST);
    imshow("open", open_image);
    resize(close_image, close_image, Size(), rate, rate, INTER_NEAREST);
    imshow("close", close_image);
    waitKey(0);
    return 0;
}