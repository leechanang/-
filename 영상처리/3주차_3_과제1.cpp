#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
// ���ڿ� ��� �Լ� - �׸��� ȿ��
void put_string(Mat& frame, string text, Point pt, int value)
{
	text += to_string(value);
	Point shade = pt + Point(2, 2);
	int font = FONT_HERSHEY_SIMPLEX;
	putText(frame, text, shade, font, 0.7, Scalar(0, 0, 0), 2); // �׸��� ȿ��
	putText(frame, text, pt, font, 0.7, Scalar(120, 200, 90), 2); // �ۼ� ����
}

int main()
{
	VideoCapture capture(0); //0�� ī�޶� ����, ���� ĸ�� ��ü ���� �� ����
	if (!capture.isOpened()) //���� ���� ���� ó��
	{
		cout << "ī�޶� ������� �ʾҽ��ϴ�." << endl;
		exit(1);
	} //ī�޶� �Ӽ� ȹ�� �� ���



	

	for (;;) {  //���� �ݺ�
		Mat frame;
		capture.read(frame); //ī�޶� ���� �ޱ�

		Scalar red(0, 0, 255);

		Point2f pt1(200, 100);
		Rect rect(pt1, Size(100, 200));
		rectangle(frame, rect, red, 3);

		frame(rect) += Scalar(0, 50, 0);

		imshow("ī�޶� ���󺸱�", frame);
		if (waitKey(30) >= 0) break; // 30ms ����


	}

}