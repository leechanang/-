#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main()
{
	VideoCapture capture(0); //0번 카메라 연결, 비디오 캡쳐 객체 선언 및 연결
	if (!capture.isOpened()) //비디오 파일 예외 처리
	{
		cout << "카메라가 연결되지 않았습니다." << endl;
		exit(1);
	} //카메라 속성 획득 및 출력

	double fps = 15; //초당 프레임 수 
	int delay = cvRound(1000.0 / fps); // 프레임간 지연시간
	Size size(640, 480); //동영상 파일 해상도
	int fourcc = VideoWriter::fourcc('D', 'I', 'V', 'X'); // 압축 코덱 설정
	

	VideoWriter writer; //동영상 파일 저장 객체
	writer.open("../image/flip_test.avi", fourcc, fps, size); //파일 개방 및 설정
	CV_Assert(writer.isOpened());

	

	for (;;) {
		Mat frame;
		capture >> frame;  //카메라 영상 받기
		
		
		Mat result;
		flip(frame, result, 1);
		writer << result;

		imshow("카메라 영상보기", frame);
		imshow("카메라 영상보기 filp", result);
		if (waitKey(delay) >= 0)
			break;
	}
}