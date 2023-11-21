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

// ���� ���͸� �����.
Mat getFilter(Size size)
{
	Mat filter(size, CV_32FC2, Vec2f(0, 0));
	circle(filter, size / 2, 50, Vec2f(1, 1), -1);
	return filter;
}
int main()
{
	Mat src = imread("C:/Users/Chan's Victus/Documents/class/Project/image/lenna.jpg", IMREAD_GRAYSCALE);
	Mat src_float;
	imshow("original", src);
	// �׷��̽����� ������ �Ǽ� �������� ��ȯ�Ѵ�.
	src.convertTo(src_float, CV_32FC1, 1.0 / 255.0);
	Mat dft_image;
	dft(src_float, dft_image, DFT_COMPLEX_OUTPUT);
	shuffleDFT(dft_image);
	Mat lowpass = getFilter(dft_image.size());
	Mat result;
	// ���� ���Ϳ� DFT ������ ���� ���Ѵ�.
	multiply(dft_image, lowpass, result);
	displayDFT(result);
	Mat inverted_image;
	shuffleDFT(result);
	idft(result, inverted_image, DFT_SCALE | DFT_REAL_OUTPUT);
	imshow("inverted", inverted_image);
	waitKey(0);
	return 1;
}