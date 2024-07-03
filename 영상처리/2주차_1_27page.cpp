#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
int main()
{
	Scalar blue(255, 0, 0), red(0, 0, 255), green(0, 255, 0);//색상선언
	Scalar white = Scalar(255, 255, 255);//흰색 색상
	Scalar yellow(0, 255, 255);

	Mat image(400, 600, CV_8UC3, white);
	Point pt1(50, 130), pt2(200, 300), pt3(300, 150), pt4(400, 50); // 좌표선언
	Rect rect(pt3, Size(200, 150));

	line(image, pt1, pt2, red); //직선그리기
	line(image, pt3, pt4, green, 2, LINE_AA); // 안티에일리싱 선
	line(image, pt3, pt4, green, 3, LINE_8, 1); // 8방향 연결선, 1비트 시프트

	rectangle(image, rect, blue, 2); // 사각형 그리기
	rectangle(image, rect, blue, FILLED, LINE_4, 1); // 4방향 연결선, 1비트 시프트 
	rectangle(image, pt1, pt2, red, 3);

	imshow("직선 & 사각형", image);
	waitKeyEx(0);
	return 0;
}
