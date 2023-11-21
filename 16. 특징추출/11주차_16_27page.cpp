#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
using namespace cv;
using namespace std;

int main()
{
	Mat src, src_gray;
	src = imread("C:/Users/Chan's Victus/Documents/class/Project/image/plates.jpg", 1);
	imshow("src", src);
	cvtColor(src, src_gray, COLOR_BGR2GRAY); // 그레이스케일로 변환한다.
	GaussianBlur(src_gray, src_gray, Size(9, 9), 2, 2); // 가우시안 블러링 적용
	vector<Vec3f> circles;
	// 원을 검출하는 허프 변환
	HoughCircles(src_gray, circles, HOUGH_GRADIENT, 1, src_gray.rows / 8, 200, 50, 0, 0);
	// 원을 영상 위에 그린다.
	for (size_t i = 0; i < circles.size(); i++) {
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		circle(src, center, 3, Scalar(0, 255, 0), -1, 8, 0); // 원의 중심을 그린다.
		circle(src, center, radius, Scalar(0, 0, 255), 3, 8, 0); // 원을 그린다.
	}
	imshow("Hough Circle Transform", src);
	waitKey(0);
	return 0;
}