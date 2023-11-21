#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
using namespace cv;
using namespace std;
int main()
{
	Mat src = imread("C:/Users/Chan's Victus/Documents/class/Project/image/building.jpg", 0);
	if (src.empty()) { cout << "can not open " << endl; return -1; }
	Mat dst, cdst;
	Canny(src, dst, 100, 200);
	imshow("edge", dst);
	cvtColor(dst, cdst, COLOR_GRAY2BGR);
	vector<Vec4i> lines;
	HoughLinesP(dst, lines, 1, CV_PI / 180, 50, 100, 20);
	for (size_t i = 0; i < lines.size(); i++) {
		Vec4i l = lines[i];
		line(cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
	}
	imshow("source", src);
	imshow("detected lines", cdst);
	waitKey();
	return 0;
}