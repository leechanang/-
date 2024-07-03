#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;


int main() {
	Mat img, img_edge, labels, centroids, img_color, stats;
	img = cv::imread("C:/Users/Chan's Victus/Documents/class/Project/image/coins.png", IMREAD_GRAYSCALE);
	threshold(img, img_edge, 128, 255, THRESH_BINARY_INV);
	imshow("Image after threshold", img_edge);
	int n = connectedComponentsWithStats(img_edge, labels, stats, centroids);
	vector<Vec3b> colors(n + 1);
	colors[0] = Vec3b(0, 0, 0);
	for (int i = 1; i <= n; i++) {
		colors[i] = Vec3b(rand() % 256, rand() % 256, rand() % 256);
	}
	img_color = cv::Mat::zeros(img.size(), CV_8UC3);
	for (int y = 0; y < img_color.rows; y++)
		for (int x = 0; x < img_color.cols; x++)
		{
			int label = labels.at<int>(y, x);
			img_color.at<cv::Vec3b>(y, x) = colors[label];
		}
	cv::imshow("Labeled map", img_color);
	cv::waitKey();
	return 0;
}