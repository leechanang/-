#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void on_trackbar(int pos, void* param) {
    vector<Mat>* images = reinterpret_cast<vector<Mat>*>(param);
    Mat blended;
    double alpha = pos / 100.0;
    double beta = 1 - alpha;
    addWeighted(images->at(0), alpha, images->at(1), beta, 0, blended);
    imshow("Blended Image", blended);
}

int main() {
    Mat image1 = imread("C:/Users/Chan's Victus/Documents/class/Project/image/lenna.jpg");
    Mat image2 = imread("C:/Users/Chan's Victus/Documents/class/Project/image/logo.jpg");

    if (image1.empty() || image2.empty()) {
        cerr << "Error loading the images" << endl;
        return -1;
    }

    resize(image2, image2, image1.size());

    namedWindow("Blended Image");

    int alpha_slider = 50;

    // Create the images vector outside the trackbar creation
    vector<Mat> images = { image1, image2 };
    createTrackbar("Alpha", "Blended Image", &alpha_slider, 100, on_trackbar, &images);

    on_trackbar(alpha_slider, &images);

    waitKey(0);

    return 0;
}
