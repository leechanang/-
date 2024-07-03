#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

void bgr2hsi(Mat img, Mat& hsi)
{
	hsi = Mat(img.size(), CV_32FC3);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++)
		{
			float B = img.at<Vec3b>(i, j)[0];
			float G = img.at<Vec3b>(i, j)[1];
			float R = img.at<Vec3b>(i, j)[2];

			float s = 1 - 3 * min(R, min(G, B)) / (R + B + G);
			float v = (R + G + B) / 3.0f;

			float tmpl = ((R - G) + (R - B)) * 0.5f;
			float tmp2 = sqrt((R - G) * (R - B) + (G - B) * (G - B));
			float angle = acos(tmpl / tmp2) * (180.f / CV_PI);
			float h = (B <= G) ? angle : 360 - angle;

			hsi.at<Vec3f>(i, j) = Vec3f(h / 2, s * 255, v);
		}
	}
}
int main()
{
	Mat BGR_img = imread("C:/Users/Chan's Victus/Documents/class/Project/image/color_space.jpg", IMREAD_COLOR);
	CV_Assert(BGR_img.data);
	Mat HSI_img, HSV_img, hsi[3], hsv[3];

	bgr2hsi(BGR_img, HSI_img);
	cvtColor(BGR_img, HSV_img, COLOR_BGR2HSV);
	split(HSI_img, hsi);
	split(HSV_img, hsv);

	imshow("BGR_img", BGR_img);
	imshow("Hue", hsi[0]);
	imshow("Saturation", hsi[1]);
	imshow("Intensity", hsi[2]);
	imshow("OpenCV_Hue", hsv[0]);
	imshow("OpenCV_Saturation", hsv[1]);
	imshow("OpenCV_Value", hsv[2]);
	waitKey();
	return 0;
}
