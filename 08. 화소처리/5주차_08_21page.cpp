#include "opencv2/opencv.hpp"
#include <stdlib.h>
#include <stdio.h>

using namespace cv;

Mat src, src_gray, dst;
int threshold_value = 0;
int threshold_type = 0;

void Threshold_Demo(int, void*)
{
	threshold(src_gray, dst, threshold_value, 255, threshold_type);
	imshow("결과 영상" , dst);
}

int main()
{
	src = imread("C:/Users/Chan's Victus/Documents/class/Project/image/lenna.jpg");
	cvtColor(src, src_gray, COLOR_BGR2GRAY);
	namedWindow("결과 영상", WINDOW_AUTOSIZE);

	createTrackbar("임계값" , "결과 영상",
		&threshold_value,
		255, Threshold_Demo);

	Threshold_Demo(0, 0);

	while (true)
	{

		int c;
		c = waitKey(20);
		if ((char)c == 27) { // ESC키가 입력되면 무한 루프 종료}
			break;
		}
	}
	return 0;

}