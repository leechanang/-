#include <opencv2/opencv.hpp>
#include<iostream>
using namespace std;
using namespace cv;

Mat img;
int drawing = false;

void drawCircle(int event, int x, int y, int, void* param) {
	if (event == EVENT_LBUTTONDOWN)
		drawing = true;
	else if (event == EVENT_MOUSEMOVE) {
		if (drawing == true)
			circle(img, Point(x, y), 3, Scalar(0, 0, 255), 10);
	}
	else if (event == EVENT_LBUTTONUP)
		drawing = false;
	imshow("Image", img);
}
int main()
{
	img = imread("C:/Users/Chan's Victus/Documents/class/photo/bug.jpg");
	if (img.empty()) { cout << "¿µ»óÀ» ÀÐÀ» ¼ö ¾øÀ½" << endl; return -1; }
	imshow("Image", img);
	setMouseCallback("Image", drawCircle);
	waitKey(0);
	imwrite("d:/bug1.jpg", img);
	return 0;
}
