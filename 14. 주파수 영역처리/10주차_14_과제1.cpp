#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

    int main() {
        // �̹����� �б�
        Mat image = imread("C:/Users/Chan's Victus/Documents/class/Project/image/image.jpg", IMREAD_GRAYSCALE);

        if (image.empty()) {
            cerr << "�̹����� ���� �� �����ϴ�." << endl;
            return -1;
        }

        // �̹����� ũ�⸦ 2�� �ŵ��������� ����� (Ǫ���� ��ȯ�� ����)
        Mat padded;
        int m = getOptimalDFTSize(image.rows);
        int n = getOptimalDFTSize(image.cols);
        copyMakeBorder(image, padded, 0, m - image.rows, 0, n - image.cols, BORDER_CONSTANT, Scalar::all(0));

        // 2D Ǫ���� ��ȯ ����
        Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
        Mat complexImage;
        merge(planes, 2, complexImage);
        dft(complexImage, complexImage);

        // ����� ũ�⸦ �α� �����Ϸ� ǥ��
        split(complexImage, planes);
        magnitude(planes[0], planes[1], planes[0]);
        Mat magImage = planes[0];

        magImage += Scalar::all(1);
        log(magImage, magImage);

        // �߽��� �������� ����
        int cx = magImage.cols / 2;
        int cy = magImage.rows / 2;

        Mat q0(magImage, Rect(0, 0, cx, cy));
        Mat q1(magImage, Rect(cx, 0, cx, cy));
        Mat q2(magImage, Rect(0, cy, cx, cy));
        Mat q3(magImage, Rect(cx, cy, cx, cy));

        Mat tmp;
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);

        q1.copyTo(tmp);
        q2.copyTo(q1);
        tmp.copyTo(q2);

        // �̹����� ǥ��
        normalize(magImage, magImage, 0, 1, NORM_MINMAX);
        imshow("Fourier Transform", magImage);
        waitKey(0);

        return 0;
    }
