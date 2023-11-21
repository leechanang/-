#include <opencv2/opencv.hpp>


using namespace cv;

int main() {
    // 이미지를 읽기
    Mat image = imread("C:/Users/Chan's Victus/Documents/class/Project/image/image.jpg", IMREAD_GRAYSCALE);

    if (image.empty()) {
        std::cerr << "이미지를 읽을 수 없습니다." << std::endl;
        return -1;
    }

    // 공간 영역에서의 smoothing (Gaussian Blur)
    Mat spatialSmoothed;
    GaussianBlur(image, spatialSmoothed, Size(5, 5), 0);

    // 주파수 영역에서의 smoothing (DFT)
    Mat complexImage, magnitudeImage;
    Mat planes[] = { Mat_<float>(image), Mat::zeros(image.size(), CV_32F) };
    merge(planes, 2, complexImage);
    dft(complexImage, complexImage);
    split(complexImage, planes);
    magnitude(planes[0], planes[1], magnitudeImage);
    log(magnitudeImage + Scalar::all(1), magnitudeImage);
    normalize(magnitudeImage, magnitudeImage, 0, 1, NORM_MINMAX);

    // 결과를 표시
    namedWindow("Spatial Smoothing", WINDOW_NORMAL);
    imshow("Spatial Smoothing", spatialSmoothed);

    namedWindow("Frequency Smoothing", WINDOW_NORMAL);
    imshow("Frequency Smoothing", magnitudeImage);

    waitKey(0);
    return 0;
}
