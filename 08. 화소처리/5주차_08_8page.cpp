#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

void brighten(Mat& img, int value)
{
	for (int r = 0; r < img.rows; r++)
		for (int c = 0; c < img.cols; ++c)
			img.at<uchar>(r, c) = saturate_cast<uchar> (img.at<uchar> (r, c) + value);
}

int main()
{
	Mat img = imread("C:/Users/Chan's Victus/Documents/class/Project/image/lenna.jpg", IMREAD_GRAYSCALE);
	imshow("Original Image", img);

	brighten(img, 30);
	imshow("New image", img);
	waitKey(0);

	return 0;
}