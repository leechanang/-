#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main()
{
	Mat src = imread("C:/Users/Chan's Victus/Documents/class/Project/image/city1.jpg", IMREAD_GRAYSCALE);
	if (src.empty()) { return -1; }
	Mat dst;
	Mat noise_img = Mat::zeros(src.rows, src.cols, CV_8U);
	randu(noise_img, 0, 255); // noise_img 의 모든 화소를 0 부터 255 까지의 난수로 채움
	Mat black_img = noise_img < 10; // noise_img 의 화소값이 10 보다 작으면 1이되는 black_img 생성
	Mat white_img = noise_img > 245; // noise_img 의 화소값이 245 보다 크면 1이되는 white_img 생성
	Mat src1 = src.clone();
	src1.setTo(255, white_img); // white_img 의 화소값이 1 이면 src1 화소값을 255 로 한다=> salt noise
	src1.setTo(0, black_img); // black_img 의 화소값이 1 이면 src1 화소값을 0 으로 한다=> pepper noise
	medianBlur(src1, dst, 5);
	imshow("source", src1);
	imshow("result", dst);
	waitKey(0);
	return 0;
}
