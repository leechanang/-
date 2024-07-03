#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main()
{
	VideoCapture capture(0); // 0�� ī�޶� ����
	CV_Assert(capture.isOpened());

	double fps = 29.97; //�ʴ� ������ �� 
	int delay = cvRound(1000.0 / fps); // �����Ӱ� �����ð�
	Size size(640, 360); //������ ���� �ػ�
	int fourcc = VideoWriter::fourcc('D', 'X', '5', '0'); // ���� �ڵ� ����

	capture.set(CAP_PROP_FRAME_WIDTH, size.width); //�ػ� ����
	capture.set(CAP_PROP_FRAME_HEIGHT, size.height);

	cout << "width x height : " << size << endl;
	cout << "VideoWriter::fourcc : " << fourcc << endl;
	cout << "delay : " << delay << endl;
	cout << "fps : " << fps << endl;

	VideoWriter writer; //������ ���� ���� ��ü
	writer.open("../image/video_file.avi", fourcc, fps, size); //���� ���� �� ����
	CV_Assert(writer.isOpened());

	for (;;) {
		Mat frame;
		capture >> frame;  //ī�޶� ���� �ޱ�
		writer << frame; // �������� ���������� ���� , writer.writer(frame);

		imshow("ī�޶� ���󺸱�", frame);
		if (waitKey(delay) >= 0)
			break;
	}
	return 0;
}
