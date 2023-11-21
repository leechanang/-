#include <opencv2/opencv.hpp>
using namespace cv;

int main()
{
    Mat image = imread("C:/Users/Chan's Victus/Documents/class/Project/image/coins.jpg");

    if (image.empty())
    {
        std::cerr << "Image not found!" << std::endl;
        return -1;
    }
    //1) 명암도 영상 변환(color -> grayscale)
    Mat gray_image;
    cvtColor(image, gray_image, COLOR_BGR2GRAY);
    
    //2) 블러링
    Mat blurred_image;
    GaussianBlur(gray_image, blurred_image, Size(9, 9), 0);

    //3) 이진화
    Mat binary_image;
    threshold(blurred_image, binary_image, 0, 255, THRESH_OTSU);

    //4) 모폴로지 연산
    Mat morph_image;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(binary_image, morph_image, MORPH_OPEN , kernel);
    morphologyEx(morph_image, morph_image, MORPH_CLOSE , kernel);
    

    imshow("Origin Image", image);
    imshow("After Image", morph_image);
    waitKey(0);

    return 0;
}
