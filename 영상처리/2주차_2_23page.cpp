#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;
// 마우스 이벤트가 발생하면 호출되는 콜백 함수이다.
void onMouse(int event, int x, int y, int flags, void* param)
{
	if (event == EVENT_LBUTTONDOWN) {
		Mat& img = *(Mat*)(param);
		circle(img, Point(x, y), 200, Scalar(0, 255, 0), 10);
		putText(img, "I found a dog!", Point(x, y + 200),
			FONT_HERSHEY_PLAIN, 2.0, 255, 2);
		imshow("src", img); // 영상이 변경되면 다시 표시한다.
	}
	else if (event == EVENT_RBUTTONDOWN) {}
	else if (event == EVENT_MBUTTONDOWN) {}
	else if (event == EVENT_MOUSEMOVE) {
	}
}
int main()
{
	Mat src = imread("C:/Users/Chan's Victus/Documents/class/photo/dog.jpg", IMREAD_COLOR);
	if (src.empty()) { cout << "영상을 읽을 수 없음" << endl; }
	imshow("src", src);
	setMouseCallback("src", onMouse, &src);
	waitKey(0);
	return 0;
}
