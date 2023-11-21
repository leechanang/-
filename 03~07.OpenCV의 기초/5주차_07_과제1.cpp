#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main()
{
    Mat image = imread("C:/Users/Chan's Victus/Documents/class/Project/image/logo.jpg", 1);

    Mat bgr[3], blue_img, red_img, green_img, zero(image.size(), CV_8U, Scalar(0));
    split(image, bgr);

   
    merge(vector<Mat>{bgr[0], zero, zero}, blue_img); // Blue channel
    merge(vector<Mat>{zero, bgr[1], zero}, green_img); // Green channel
    merge(vector<Mat>{zero, zero, bgr[2]}, red_img);   // Red channel

    imshow("image", image);
    imshow("blue_img", blue_img);
    imshow("green_img", green_img);
    imshow("red_img", red_img);
    waitKey(0);

    return 0;
}
