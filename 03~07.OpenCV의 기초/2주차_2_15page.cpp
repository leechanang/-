#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;
int main()
{
	Mat src = imread("C:/Users/Chan's Victus/Documents/class/photo/photo1.jpg", IMREAD_COLOR);
	if (src.empty()) { cout << "������ ���� �� ����" << endl; }
	imshow("src", src);
	while (1) {
		int key = waitKeyEx(); // ����ڷκ��� Ű�� ��ٸ���.
		cout << key << " ";
		if (key == 'q') break; // ����ڰ� ��q'�� ������ �����Ѵ�.
		else if (key == 2424832) { // ����ȭ��ǥ Ű
			src -= 50;// ������ ��ο�����.
		}
		else if (key == 2555904) { // ������ȭ��ǥ Ű
			src += 50; // ������ �������.
		}
		imshow("src", src); // ������ ����Ǿ����Ƿ� �ٽ� ǥ���Ѵ�.
	}
	return 0;
}