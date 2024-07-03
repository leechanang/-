#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

void put_string(Mat& frame, string text, Point pt, int value)
{
	text += to_string(value);
	Point shade = pt + Point(2, 2);
	int font = FONT_HERSHEY_SIMPLEX;
	putText(frame, text, shade, font, 0.7, Scalar(0, 0, 0), 2);
	putText(frame, text, pt, font, 0.7, Scalar(120, 200, 90), 2);
}
int main()
{
	VideoCapture capture; 
	capture.open("../image/video_file.avi"); // 동영상 파일 개방
	CV_Assert(capture.isOpened());

	double frame_rate = capture.get(CAP_PROP_FPS); // 초당 프레임 수
	int delay = 1000 / frame_rate; // 지연시간
	int frame_cnt = 0; //현재 프레임의 번호
	Mat frame;

	while (capture.read(frame)) //프레임 반복 재생
	{
		if (waitKey(delay) >= 0) break; // 프레임간 지연시간 지정

		if (frame_cnt < 100);
		else if (frame_cnt < 200) frame -= Scalar(0, 0, 100); //199프레임까지 빨간색 증가
		else if (frame_cnt < 300) frame += Scalar(100, 0, 0); //299프레임까지 파란색 증가
		else if (frame_cnt < 400) frame = frame * 1.5; //399프레임까지 대비 증가
		else if (frame_cnt < 500) frame = frame * 0.5; //499프레임까지 대비 감소

		put_string(frame, "frame_cnt", Point(20, 50), frame_cnt);
		imshow("동영상 파일읽기", frame);

		frame_cnt++;
	}
	return 0;
}
