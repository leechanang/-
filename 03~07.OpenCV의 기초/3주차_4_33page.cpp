#include "opencv2/opencv.hpp"
#include <iostream>
using namespace cv;
using namespace std;

Mat img, roi;
int mx1, my1, mx2, my2; // 마우스로 지정한 사각형의 좌표
bool cropping = false; // 사각형 선택 중임을 나타내는 플래그 변수

//마우스 이벤트가 발생하면 호출되는 콜백 함수이다.
void onMouse (int event, int x, int y, int flags, void* param)
{
	if (event == EVENT_LBUTTONDOWN) {// 마우스의 왼쪽 버튼을 누르면
			mx1 = x; // 사각형의 좌측 상단 좌표 저장
			my1 = y;
			cropping = true;
		}
	else if (event == EVENT_LBUTTONUP) 	{// 마우스의 왼쪽 버튼에서 손을 떼면
			mx2 = x; // 사각형의 우측 하단 좌표 저장
			my2 = y;
			cropping = false;
			rectangle(img, Rect(mx1, my1, mx2-mx1, my2-my1), Scalar(0, 255, 0), 2);
			imshow("image", img);
		}
	}

int main() {
	img = imread("../image/lenna.jpg");
	imshow("image", img);
	Mat clone =	img.clone(); // 복사본을 만들어둔다

	setMouseCallback("image", onMouse);

	while (1) {
		int	key = waitKey(100);
		if (key == 'q') break; // 사용자가 ‘q’ 를 누르면 종료
		else if (key == 'c') {   // 사용자가 ‘c’ 를 누르면 관심영역을 파일로 저장
				roi = clone(Rect(mx1, my1, mx2-mx1, my2-my1));
				imwrite("../image/result.jpg", roi);
		}
	}
	return 0;
}