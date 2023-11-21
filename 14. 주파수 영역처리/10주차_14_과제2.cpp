#include <opencv2/opencv.hpp>


using namespace cv;

int main() {
    // �̹����� �б�
    Mat image = imread("C:/Users/Chan's Victus/Documents/class/Project/image/image.jpg", IMREAD_GRAYSCALE);

    if (image.empty()) {
        std::cerr << "�̹����� ���� �� �����ϴ�." << std::endl;
        return -1;
    }

    // ���� ���������� smoothing (Gaussian Blur)
    Mat spatialSmoothed;
    GaussianBlur(image, spatialSmoothed, Size(5, 5), 0);

    // ���ļ� ���������� smoothing (DFT)
    Mat complexImage, magnitudeImage;
    Mat planes[] = { Mat_<float>(image), Mat::zeros(image.size(), CV_32F) };
    merge(planes, 2, complexImage);
    dft(complexImage, complexImage);
    split(complexImage, planes);
    magnitude(planes[0], planes[1], magnitudeImage);
    log(magnitudeImage + Scalar::all(1), magnitudeImage);
    normalize(magnitudeImage, magnitudeImage, 0, 1, NORM_MINMAX);

    // ����� ǥ��
    namedWindow("Spatial Smoothing", WINDOW_NORMAL);
    imshow("Spatial Smoothing", spatialSmoothed);

    namedWindow("Frequency Smoothing", WINDOW_NORMAL);
    imshow("Frequency Smoothing", magnitudeImage);

    waitKey(0);
    return 0;
}
