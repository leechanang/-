#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

void displayDFT(Mat& src)
{
	Mat image_array[2] = { Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(),CV_32F) };
	// �� DFT ��� ������ 2���� �������� �и��Ѵ�.
	split(src, image_array);
	Mat mag_image;
	// �� Ǫ���� ��ȯ ������� ���밪�� ����Ѵ�.
	magnitude(image_array[0], image_array[1], mag_image);
	// �� Ǫ���� ��ȯ ������� ����� ũ�� ������ �α� �����Ϸ� ��ȯ�Ѵ�.
	// 0���� ������ �ʵ��� 1�� �����ش�.
	mag_image += Scalar::all(1);
	log(mag_image, mag_image);
	// �� 0���� 255�� ������ ����ȭ�Ѵ�.
	normalize(mag_image, mag_image, 0, 1, NORM_MINMAX);
	imshow("DFT", mag_image);
	waitKey(0);
}

void shuffleDFT(Mat& src)
{
	int cX = src.cols / 2;
	int cY = src.rows / 2;
	Mat q1(src, Rect(0, 0, cX, cY));
	Mat q2(src, Rect(cX, 0, cX, cY));
	Mat q3(src, Rect(0, cY, cX, cY));
	Mat q4(src, Rect(cX, cY, cX, cY));
	Mat tmp;
	q1.copyTo(tmp);
	q4.copyTo(q1);
	tmp.copyTo(q4);
	q2.copyTo(tmp);
	q3.copyTo(q2);
	tmp.copyTo(q3);
}

int main()
{
	Mat img = imread("C:/Users/Chan's Victus/Documents/class/Project/image/lenna.jpg", IMREAD_GRAYSCALE);
	Mat img_float, dft1, inversedft, inversedft1;
	img.convertTo(img_float, CV_32F);
	dft(img_float, dft1, DFT_COMPLEX_OUTPUT);
	// ����ȯ�� �����Ѵ�.
	idft(dft1, inversedft, DFT_SCALE | DFT_REAL_OUTPUT);
	inversedft.convertTo(inversedft1, CV_8U);
	imshow("invertedfft", inversedft1);
	imshow("original", img);
	waitKey(0);
	return 0;
}