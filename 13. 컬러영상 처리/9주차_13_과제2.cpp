#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>

using namespace cv;
using namespace std;

int main() {
    // 카메라 열기
    VideoCapture cap(0); // 0은 기본 카메라 장치를 의미합니다. 다른 카메라 장치를 사용하려면 숫자를 바꿀 수 있습니다.

    // 카메라 열기에 실패하면 오류 메시지 출력하고 종료
    if (!cap.isOpened()) {
        cerr << "카메라를 열 수 없습니다." << endl;
        return -1;
    }

    // 얼굴 검출기 초기화
    CascadeClassifier face_cascade;
    if (!face_cascade.load("haarcascade_frontalface_alt.xml")) { // 다운로드한 파일의 경로를 제공해야 합니다.
        cerr << "얼굴 검출기를 불러올 수 없습니다." << endl;
        return -1;
    }

    // 영상 처리 루프
    while (true) {
        Mat frame;
        cap >> frame; // 프레임 읽기

        if (frame.empty()) {
            cerr << "프레임을 읽을 수 없습니다." << endl;
            break;
        }

        // 프레임을 그레이스케일로 변환
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // 얼굴 검출
        vector<Rect> faces;
        face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, Size(30, 30));

        // 검출된 얼굴에 사각형 그리기
        for (const Rect& face : faces) {
            rectangle(frame, face, Scalar(0, 0, 255), 2);
        }

        // 화면에 프레임 표시
        imshow("Face Detection", frame);

        // 'q' 키를 누르면 루프 종료
        if (waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
    destroyAllWindows();

    return 0;
}
