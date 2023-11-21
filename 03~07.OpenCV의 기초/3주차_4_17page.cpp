#include "opencv2/opencv.hpp"
#include <iostream>
using namespace cv;
using namespace std;

int main()
{
	Mat	img = imread("../image/lenna.jpg");
	if (img.empty()) {cout << "������ ���� �� ����" << endl ; return 1;}
	imshow ("img", img);

	cout << "���� �� = " << img.rows << endl;
	cout<< "���� �� = " << img.cols << endl;
	cout<< "����� ũ�� = " << img.size () << endl;
	cout<< "��ü ȭ�� ���� = " << img.total () << endl;
	cout<< "�� ȭ�� ũ�� = " << img.elemSize () << endl;
	cout<< "Ÿ�� = " << img.type ()<< endl;
	cout<< "ä�� = " << img.channels()<< endl;

	waitKey(0);
	return 0;
}