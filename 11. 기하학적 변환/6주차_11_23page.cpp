#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main()
{
	Mat src = imread("C:/Users/Chan's Victus/Documents/class/Project/image/book.jpg");
	Point2f inputp[4];
	inputp[0] = Point2f(30, 81);
	inputp[1] = Point2f(274, 247);
	inputp[2] = Point2f(298, 40);
	inputp[3] = Point2f(598, 138);
	Point2f outputp[4];
	outputp[0] = Point2f(0, 0);
	outputp[1] = Point2f(0, src.rows);
	outputp[2] = Point2f(src.cols, 0);
	outputp[3] = Point2f(src.cols, src.rows);
	Mat h = getPerspectiveTransform(inputp, outputp);
	Mat out;
	warpPerspective(src, out, h, src.size());
	imshow("Source Image", src);
	imshow("Warped Source Image", out);
	waitKey(0);
}