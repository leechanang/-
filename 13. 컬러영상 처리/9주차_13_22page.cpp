#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main()
{
	VideoCapture cap("C:/Users/Chan's Victus/Documents/class/Project/image/tennis_ball.mp4");
	if (!cap.isOpened())
		return -1;
	for (;;)
	{
		Mat imgHSV;
		Mat frame;
		cap >> frame;
		cvtColor(frame, imgHSV, COLOR_BGR2HSV);
		Mat imgThresholded;
		inRange(imgHSV, Scalar(30, 10, 10), Scalar(38, 255, 255), imgThresholded);
		imshow("frame", frame);
		imshow("dst", imgThresholded);
		if (waitKey(30) >= 0) break;
	}
	waitKey(0);
	return 0;
}