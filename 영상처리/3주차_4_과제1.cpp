#include "opencv2/opencv.hpp"
#include <iostream>
using namespace cv;
using namespace std;

int main()
{
	Range r1(2, 3), r2(3, 5);


	int data[] = {
		10,11,12,13,14,15,16,
		20,21,22,23,24,25,26,
		30,31,32,33,34,35,36,
		40,41,42,43,44,45,46,
	};
	Mat m1(5, 7, CV_32S, data);
	
	cout << m1(r1, r2)<< endl;

}