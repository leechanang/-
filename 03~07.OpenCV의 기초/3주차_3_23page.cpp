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
	capture.open("../image/video_file.avi"); // ������ ���� ����
	CV_Assert(capture.isOpened());

	double frame_rate = capture.get(CAP_PROP_FPS); // �ʴ� ������ ��
	int delay = 1000 / frame_rate; // �����ð�
	int frame_cnt = 0; //���� �������� ��ȣ
	Mat frame;

	while (capture.read(frame)) //������ �ݺ� ���
	{
		if (waitKey(delay) >= 0) break; // �����Ӱ� �����ð� ����

		if (frame_cnt < 100);
		else if (frame_cnt < 200) frame -= Scalar(0, 0, 100); //199�����ӱ��� ������ ����
		else if (frame_cnt < 300) frame += Scalar(100, 0, 0); //299�����ӱ��� �Ķ��� ����
		else if (frame_cnt < 400) frame = frame * 1.5; //399�����ӱ��� ��� ����
		else if (frame_cnt < 500) frame = frame * 0.5; //499�����ӱ��� ��� ����

		put_string(frame, "frame_cnt", Point(20, 50), frame_cnt);
		imshow("������ �����б�", frame);

		frame_cnt++;
	}
	return 0;
}
