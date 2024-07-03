#include <opencv2/opencv.hpp>


using namespace cv;
using namespace std;

void medianFilter(const Mat& src, Mat& dst, int ksize) {
    dst = src.clone();
    int offset = ksize / 2;
    for (int y = offset; y < src.rows - offset; y++) {
        for (int x = offset; x < src.cols - offset; x++) {
            vector<uchar> neighbors;

            for (int i = -offset; i <= offset; i++) {
                for (int j = -offset; j <= offset; j++) {
                    neighbors.push_back(src.at<uchar>(y + i, x + j));
                }
            }

            sort(neighbors.begin(), neighbors.end());
            dst.at<uchar>(y, x) = neighbors[neighbors.size() / 2];
        }
    }
}

int main() {
    Mat src = imread("C:/Users/Chan's Victus/Documents/class/Project/image/city1.jpg", IMREAD_GRAYSCALE);
    if (src.empty()) { return -1; }

    Mat dst;
    Mat noise_img = Mat::zeros(src.rows, src.cols, CV_8U);
    randu(noise_img, 0, 255);
    Mat black_img = noise_img < 10;
    Mat white_img = noise_img > 245;
    Mat src1 = src.clone();
    src1.setTo(255, white_img);
    src1.setTo(0, black_img);

    // Use our custom medianFilter function instead of OpenCV's medianBlur
    medianFilter(src1, dst, 5);

    imshow("source", src1);
    imshow("result", dst);
    waitKey(0);
    return 0;
}
