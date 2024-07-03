#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;
int main()
{
	Mat src = imread("C:/Users/Chan's Victus/Documents/class/photo/photo1.jpg", IMREAD_COLOR);
	if (src.empty()) { cout << "영상을 읽을 수 없음" << endl; }
	imshow("src", src);
	while (1) {
		int key = waitKeyEx(); // 사용자로부터 키를 기다린다.
		cout << key << " ";
		if (key == 'q') break; // 사용자가 ‘q'를 누르면 종료한다.
		else if (key == 2424832) { // 왼쪽화살표 키
			src -= 50;// 영상이 어두워진다.
		}
		else if (key == 2555904) { // 오른쪽화살표 키
			src += 50; // 영상이 밝아진다.
		}
		imshow("src", src); // 영상이 변경되었으므로 다시 표시한다.
	}
	return 0;
}
