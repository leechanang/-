#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main()
{

	Mat src = imread("C:/Users/Chan's Victus/Documents/class/Project/image/lenna.jpg", IMREAD_COLOR); 
	Point2f srcTri[3];
	Point2f dstTri[3];
	Mat warp_mat(2, 3, CV_32FC1);
	Mat warp_dst;
	warp_dst = Mat::zeros(src.rows, src.cols, src.type()); srcTri[0] = Point2f(0, 0);
	srcTri[1] = Point2f(src.cols - 1.0f, 0);
	srcTri[2] = Point2f(0, src.rows - 1.0f);
	dstTri[0] = Point2f(src.cols * 0.0f, src.rows * 0.33f); dstTri[1] = Point2f(src.cols * 0.85f, src.rows * 0.25f); dstTri[2] = Point2f(src.cols * 0.15f, src.rows * 0.7f); warp_mat = getAffineTransform(srcTri, dstTri); warpAffine(src, warp_dst, warp_mat, warp_dst.size());
	imshow("src", src);
	imshow("dst", warp_dst);
	waitKey(0);
	return 1;
}