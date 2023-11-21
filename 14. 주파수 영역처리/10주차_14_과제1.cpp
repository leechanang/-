#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

    int main() {
        // 이미지를 읽기
        Mat image = imread("C:/Users/Chan's Victus/Documents/class/Project/image/image.jpg", IMREAD_GRAYSCALE);

        if (image.empty()) {
            cerr << "이미지를 읽을 수 없습니다." << endl;
            return -1;
        }

        // 이미지의 크기를 2의 거듭제곱으로 만들기 (푸리에 변환을 위해)
        Mat padded;
        int m = getOptimalDFTSize(image.rows);
        int n = getOptimalDFTSize(image.cols);
        copyMakeBorder(image, padded, 0, m - image.rows, 0, n - image.cols, BORDER_CONSTANT, Scalar::all(0));

        // 2D 푸리에 변환 수행
        Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
        Mat complexImage;
        merge(planes, 2, complexImage);
        dft(complexImage, complexImage);

        // 결과의 크기를 로그 스케일로 표시
        split(complexImage, planes);
        magnitude(planes[0], planes[1], planes[0]);
        Mat magImage = planes[0];

        magImage += Scalar::all(1);
        log(magImage, magImage);

        // 중심을 기준으로 스왑
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

        // 이미지를 표시
        normalize(magImage, magImage, 0, 1, NORM_MINMAX);
        imshow("Fourier Transform", magImage);
        waitKey(0);

        return 0;
    }
