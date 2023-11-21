#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main()
{
	//VideoCapture cap(0); // 웹캠인 경우
	VideoCapture cap("../image/trailer.mp4");// 동영상 파일인 경우
	if (!cap.isOpened()) 
	{ 
		cout << "동영상을 읽을 수 없음" << endl ;
	}

	namedWindow("frame", 1);// 윈도우 생성
	for (;;)
	{
			Mat frame;
			cap >> frame; // 동영상에서 하나의 프레임을 추출한다.
			imshow("frame", frame);
			if (waitKey(30) >= 0) break;
	}
	return 0;
}
