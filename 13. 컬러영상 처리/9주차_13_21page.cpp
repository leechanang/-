#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main()
{
	Mat img = imread("C:/Users/Chan's Victus/Documents/class/Project/image/image1.jpg", IMREAD_COLOR);
	if (img.empty()) { return -1; }
	Mat imgHSV;
	cvtColor(img, imgHSV, COLOR_BGR2HSV);
	Mat imgThresholded;
	inRange(imgHSV, Scalar(100, 0, 0), Scalar(120, 255, 255), imgThresholded);
	imshow("Thresholded Image", imgThresholded);
	imshow("Original", img);
	waitKey(0);
	return 0;
}