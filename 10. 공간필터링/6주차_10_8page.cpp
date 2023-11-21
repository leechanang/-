#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

void filter(Mat img, Mat& dst, Mat mask)
{
	dst = Mat(img.size(), CV_32F, Scalar(0));
	Point h_m = mask.size() / 2;

	for (int i = h_m.y; i < img.rows - h_m.y; i++) {
		for (int j = h_m.x; j < img.cols - h_m.x; j++)
		{
			float sum = 0;
			for (int u = 0; u < mask.rows; u++) {
				for (int v = 0; v < mask.cols; v++)
				{
					int y = i + u - h_m.y;
					int x = j + v - h_m.x;
					sum += mask.at<float>(u, v) * img.at<uchar>(y, x);
				}
			}
			dst.at<float>(i, j) = sum;
		}
	}
}
int main()
{
	Mat image = imread("C:/Users/Chan's Victus/Documents/class/Project/image/filter_blur.jpg",
		IMREAD_GRAYSCALE);
	CV_Assert(image.data);

	float data[] = {
		1 / 9.f, 1 / 9.f, 1 / 9.f,
		1 / 9.f, 1 / 9.f, 1 / 9.f,
		1 / 9.f, 1 / 9.f, 1 / 9.f
	};
	Mat mask(3, 3, CV_32F, data);
	Mat blur;
	filter(image, blur, mask);
	blur.convertTo(blur, CV_8U);

	imshow("image", image), imshow("blur", blur);
	waitKey();
	return 0;
}
