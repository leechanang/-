#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

Mat img;
Rect roi;
Point startPoint, endPoint;
bool drag = false;

void showHistogram(Mat& image) {
    int bins = 256;
    int histSize[] = { bins };
    float range[] = { 0, 256 };
    const float* ranges[] = { range };

    Mat histR, histG, histB;

    int channels[] = { 0 };
    calcHist(&image, 1, channels, Mat(), histR, 1, histSize, ranges, true, false);
    channels[0] = 1;
    calcHist(&image, 1, channels, Mat(), histG, 1, histSize, ranges, true, false);
    channels[0] = 2;
    calcHist(&image, 1, channels, Mat(), histB, 1, histSize, ranges, true, false);

    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double)hist_w / bins);
    Mat histImageR(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
    Mat histImageG(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
    Mat histImageB(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

    normalize(histR, histR, 0, histImageR.rows, NORM_MINMAX, -1, Mat());
    normalize(histG, histG, 0, histImageG.rows, NORM_MINMAX, -1, Mat());
    normalize(histB, histB, 0, histImageB.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < bins; i++) {
        line(histImageR, Point(bin_w * (i - 1), hist_h - cvRound(histR.at<float>(i - 1))),
            Point(bin_w * i, hist_h - cvRound(histR.at<float>(i))),
            Scalar(0, 0, 255), 2, 8, 0);
        line(histImageG, Point(bin_w * (i - 1), hist_h - cvRound(histG.at<float>(i - 1))),
            Point(bin_w * i, hist_h - cvRound(histG.at<float>(i))),
            Scalar(0, 255, 0), 2, 8, 0);
        line(histImageB, Point(bin_w * (i - 1), hist_h - cvRound(histB.at<float>(i - 1))),
            Point(bin_w * i, hist_h - cvRound(histB.at<float>(i))),
            Scalar(255, 0, 0), 2, 8, 0);
    }
    imshow("Red Histogram", histImageR);
    imshow("Green Histogram", histImageG);
    imshow("Blue Histogram", histImageB);
}
void mouseCallback(int event, int x, int y, int flags, void* userdata) {
    if (event == EVENT_LBUTTONDOWN) {
        startPoint = Point(x, y);
        drag = true;
    }
    else if (event == EVENT_LBUTTONUP) {
        endPoint = Point(x, y);
        roi = Rect(startPoint, endPoint);
        Mat selectedRegion = img(roi);
        showHistogram(selectedRegion);
        drag = false;
    }
    else if (drag && event == EVENT_MOUSEMOVE) {
        Mat tempImg = img.clone();
        rectangle(tempImg, startPoint, Point(x, y), Scalar(0, 255, 0), 2);
        imshow("Image", tempImg);
    }
}
int main() {
    img = imread("C:/Users/Chan's Victus/Documents/class/Project/image/lenna.jpg");
    if (img.empty()) {
        cerr << "Error loading the image" << endl;
        return -1;
    }
    namedWindow("Image", WINDOW_AUTOSIZE);
    imshow("Image", img);
    setMouseCallback("Image", mouseCallback);

    waitKey(0);
    return 0;
}
