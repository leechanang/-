#include "opencv2/opencv.hpp"
using namespace cv;
using namespace std;
int main() {
	Mat m1(10, 15, CV_32S, Scalar(100));
	Rect r1(3, 1, 5, 4); 	Mat D1 = m1(r1); D1 = Scalar(200);
	Rect r2(8, 5, 6, 4);    Mat D3 = m1(r2); D3 = Scalar(300);
	Rect r3(5, 3, 5, 4);    Mat D2 = m1(r3); D2 = Scalar(555);
	cout << m1 << endl; return 0;
}