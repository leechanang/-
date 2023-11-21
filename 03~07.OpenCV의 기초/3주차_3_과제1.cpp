#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
// 문자열 출력 함수 - 그림자 효과
void put_string(Mat& frame, string text, Point pt, int value)
{
	text += to_string(value);
	Point shade = pt + Point(2, 2);
	int font = FONT_HERSHEY_SIMPLEX;
	putText(frame, text, shade, font, 0.7, Scalar(0, 0, 0), 2); // 그림자 효과
	putText(frame, text, pt, font, 0.7, Scalar(120, 200, 90), 2); // 작성 문자
}

int main()
{
	VideoCapture capture(0); //0번 카메라 연결, 비디오 캡쳐 객체 선언 및 연결
	if (!capture.isOpened()) //비디오 파일 예외 처리
	{
		cout << "카메라가 연결되지 않았습니다." << endl;
		exit(1);
	} //카메라 속성 획득 및 출력



	

	for (;;) {  //무한 반복
		Mat frame;
		capture.read(frame); //카메라 영상 받기

		Scalar red(0, 0, 255);

		Point2f pt1(200, 100);
		Rect rect(pt1, Size(100, 200));
		rectangle(frame, rect, red, 3);

		frame(rect) += Scalar(0, 50, 0);

		imshow("카메라 영상보기", frame);
		if (waitKey(30) >= 0) break; // 30ms 지연


	}

}