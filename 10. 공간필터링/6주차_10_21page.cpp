#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main()
{
	Mat src = imread("C:/Users/Chan's Victus/Documents/class/Project/image/city1.jpg", IMREAD_GRAYSCALE);
	if (src.empty()) { return -1; }
	Mat dst;
	Mat noise_img = Mat::zeros(src.rows, src.cols, CV_8U);
	randu(noise_img, 0, 255); // noise_img �� ��� ȭ�Ҹ� 0 ���� 255 ������ ������ ä��
	Mat black_img = noise_img < 10; // noise_img �� ȭ�Ұ��� 10 ���� ������ 1�̵Ǵ� black_img ����
	Mat white_img = noise_img > 245; // noise_img �� ȭ�Ұ��� 245 ���� ũ�� 1�̵Ǵ� white_img ����
	Mat src1 = src.clone();
	src1.setTo(255, white_img); // white_img �� ȭ�Ұ��� 1 �̸� src1 ȭ�Ұ��� 255 �� �Ѵ�=> salt noise
	src1.setTo(0, black_img); // black_img �� ȭ�Ұ��� 1 �̸� src1 ȭ�Ұ��� 0 ���� �Ѵ�=> pepper noise
	medianBlur(src1, dst, 5);
	imshow("source", src1);
	imshow("result", dst);
	waitKey(0);
	return 0;
}
