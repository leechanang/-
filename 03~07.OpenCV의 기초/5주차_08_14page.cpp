#include "opencv2/opencv.hpp"
#include <iostream>
using namespace std;
using namespace cv;

int main()
{
	double alpha = 1.0;
	int beta = 0;
	Mat image =	imread("C:/Users/Chan's Victus/Documents/class/Project/image/contrast.jpg");
	Mat	oimage;
	cout << "���İ��� �Է��Ͻÿ� : [1.0 3.0] : "; cin >> alpha;
	cout << "��Ÿ���� �Է��Ͻÿ� : [0 100] : "; cin >> beta;
			
	image.convertTo(oimage, 1, alpha, beta);
	imshow("Original Image", image);
	imshow("New Image", oimage);
	waitKey();
	return 0;
}