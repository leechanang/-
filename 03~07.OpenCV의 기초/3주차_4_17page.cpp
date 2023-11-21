#include "opencv2/opencv.hpp"
#include <iostream>
using namespace cv;
using namespace std;

int main()
{
	Mat	img = imread("../image/lenna.jpg");
	if (img.empty()) {cout << "영상을 읽을 수 없음" << endl ; return 1;}
	imshow ("img", img);

	cout << "행의 수 = " << img.rows << endl;
	cout<< "열의 수 = " << img.cols << endl;
	cout<< "행렬의 크기 = " << img.size () << endl;
	cout<< "전체 화소 개수 = " << img.total () << endl;
	cout<< "한 화소 크기 = " << img.elemSize () << endl;
	cout<< "타입 = " << img.type ()<< endl;
	cout<< "채널 = " << img.channels()<< endl;

	waitKey(0);
	return 0;
}