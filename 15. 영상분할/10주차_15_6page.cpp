#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int threshold_value = 128;
int threshold_type = 0;
const int max_value = 255;
const int max_binary_value = 255;
Mat src, src_gray, dst;
static void MyThreshold(int, void*)
{
	threshold(src, dst, threshold_value, max_binary_value, threshold_type);
	imshow("result", dst);
}
int main()
{
	src = imread("C:/Users/Chan's Victus/Documents/class/Project/image/lenna.jpg", IMREAD_GRAYSCALE);
	namedWindow("result", WINDOW_AUTOSIZE);
	createTrackbar("�Ӱ谪", "result", &threshold_value, max_value, MyThreshold);
	MyThreshold(0, 0); // �ʱ�ȭ�� ���Ͽ� ȣ���Ѵ�.
	waitKey();
	return 0;
}