#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>

using namespace cv;
using namespace std;

int main() {
    // ī�޶� ����
    VideoCapture cap(0); // 0�� �⺻ ī�޶� ��ġ�� �ǹ��մϴ�. �ٸ� ī�޶� ��ġ�� ����Ϸ��� ���ڸ� �ٲ� �� �ֽ��ϴ�.

    // ī�޶� ���⿡ �����ϸ� ���� �޽��� ����ϰ� ����
    if (!cap.isOpened()) {
        cerr << "ī�޶� �� �� �����ϴ�." << endl;
        return -1;
    }

    // �� ����� �ʱ�ȭ
    CascadeClassifier face_cascade;
    if (!face_cascade.load("haarcascade_frontalface_alt.xml")) { // �ٿ�ε��� ������ ��θ� �����ؾ� �մϴ�.
        cerr << "�� ����⸦ �ҷ��� �� �����ϴ�." << endl;
        return -1;
    }

    // ���� ó�� ����
    while (true) {
        Mat frame;
        cap >> frame; // ������ �б�

        if (frame.empty()) {
            cerr << "�������� ���� �� �����ϴ�." << endl;
            break;
        }

        // �������� �׷��̽����Ϸ� ��ȯ
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // �� ����
        vector<Rect> faces;
        face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, Size(30, 30));

        // ����� �󱼿� �簢�� �׸���
        for (const Rect& face : faces) {
            rectangle(frame, face, Scalar(0, 0, 255), 2);
        }

        // ȭ�鿡 ������ ǥ��
        imshow("Face Detection", frame);

        // 'q' Ű�� ������ ���� ����
        if (waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
    destroyAllWindows();

    return 0;
}
