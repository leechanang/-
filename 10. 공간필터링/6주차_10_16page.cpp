#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

void filter(Mat img, Mat& dst, Mat mask) { filter2D(img, dst, -1, mask); }

int main()
{
	Mat image = imread("C:/Users/Chan's Victus/Documents/class/Project/image/filter_sharpen.jpg", IMREAD_GRAYSCALE);
	CV_Assert(image.data);

	float data1[] = {
		0, -1, 0,
		-1, 5, -1,
		0, -1, 0,
	};

	float data2[] = {
		-1, -1, -1,
		-1, 9, -1,
		-1, -1, -1,
	};

	Mat mask1(3, 3, CV_32F, data1);
	Mat mask2(3, 3, CV_32F, data2);
	Mat sharpen1, sharpen2;
	filter(image, sharpen1, mask1);
	filter(image, sharpen2, mask2);
	sharpen1.convertTo(sharpen1, CV_8U);
	sharpen2.convertTo(sharpen2, CV_8U);

	imshow("image", image);
	imshow("sharpen1", sharpen1), imshow("sharpen2", sharpen2);
	waitKey();
	return 0;
}