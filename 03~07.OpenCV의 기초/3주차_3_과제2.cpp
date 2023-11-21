#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main()
{
	VideoCapture capture(0); //0�� ī�޶� ����, ���� ĸ�� ��ü ���� �� ����
	if (!capture.isOpened()) //���� ���� ���� ó��
	{
		cout << "ī�޶� ������� �ʾҽ��ϴ�." << endl;
		exit(1);
	} //ī�޶� �Ӽ� ȹ�� �� ���

	double fps = 15; //�ʴ� ������ �� 
	int delay = cvRound(1000.0 / fps); // �����Ӱ� �����ð�
	Size size(640, 480); //������ ���� �ػ�
	int fourcc = VideoWriter::fourcc('D', 'I', 'V', 'X'); // ���� �ڵ� ����
	

	VideoWriter writer; //������ ���� ���� ��ü
	writer.open("../image/flip_test.avi", fourcc, fps, size); //���� ���� �� ����
	CV_Assert(writer.isOpened());

	

	for (;;) {
		Mat frame;
		capture >> frame;  //ī�޶� ���� �ޱ�
		
		
		Mat result;
		flip(frame, result, 1);
		writer << result;

		imshow("ī�޶� ���󺸱�", frame);
		imshow("ī�޶� ���󺸱� filp", result);
		if (waitKey(delay) >= 0)
			break;
	}
}