#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;

Mat frame;
Point roi_start, roi_end;
bool selecting = false;
bool roi_selected = false;
chrono::high_resolution_clock::time_point start_time;
Mat selected_roi;  // 드래그한 영역 저장
bool show_histogram = false;  // 히스토그램 그래프를 표시할지 여부
Mat roi_window;  // ROI를 표시할 새로운 창
bool space_pressed = false;  // 스페이스바를 누르면 영상 일시 중지

void showROIHistogram(Mat& roi);

void selectROI(int event, int x, int y, int flags, void* param) {
    if (space_pressed) {
        return;
    }

    if (event == EVENT_LBUTTONDOWN) {
        roi_start = Point(x, y);
        roi_end = Point(x, y);
        selecting = true;
        roi_selected = false;
        show_histogram = false;
    }
    else if (event == EVENT_LBUTTONUP) {
        roi_end = Point(x, y);
        selecting = false;
        roi_selected = true;
        show_histogram = true;
    }
    else if (event == EVENT_MOUSEMOVE) {
        if (selecting) {
            roi_end = Point(x, y);
        }
    }
}

int main() {
    VideoCapture video("C:/Users/Chan's Victus/Documents/class/Project/image/trailer.mp4");

    if (!video.isOpened()) {
        cerr << "Error: Could not open the video file." << endl;
        return -1;
    }

    namedWindow("Video");
    setMouseCallback("Video", selectROI);

    namedWindow("Hue Histogram", WINDOW_NORMAL);  // 히스토그램 창

    while (true) {
        if (!space_pressed) {
            video >> frame;
        }

        Mat frame_copy = frame.clone();

        if (selecting) {
            rectangle(frame_copy, roi_start, roi_end, Scalar(0, 255, 0), 2);
        }
        else if (roi_selected) {
            Point top_left(min(roi_start.x, roi_end.x), min(roi_start.y, roi_end.y));
            Point bottom_right(max(roi_start.x, roi_end.x), max(roi_start.y, roi_end.y));
            Rect roi_rect(top_left, bottom_right);
            selected_roi = frame(roi_rect).clone();
            roi_selected = false;
        
            namedWindow("Selected ROI", WINDOW_NORMAL);
          
            resizeWindow("Selected ROI", selected_roi.cols, selected_roi.rows);
            imshow("Selected ROI", selected_roi);
        }

        if (show_histogram && !selected_roi.empty()) {
            showROIHistogram(selected_roi);
        }

        imshow("Video", frame_copy);

        char key = waitKey(16);

        if (key == 27) {  // Press ESC to exit
            break;
        }
        else if (key == 32) {  // Press Spacebar to pause/resume video
            space_pressed = !space_pressed;
        }
    }

    video.release();
    destroyAllWindows();
    return 0;
}

void showROIHistogram(Mat& roi) {
    Mat roi_hsv;
    cvtColor(roi, roi_hsv, COLOR_BGR2HSV);

    vector<Mat> hsv_channels;
    split(roi_hsv, hsv_channels);
    Mat hue = hsv_channels[0];
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    Mat hist;
    calcHist(&hue, 1, 0, Mat(), hist, 1, &histSize, &histRange);

    normalize(hist, hist, 0, 255, NORM_MINMAX);

    Mat histImage(256, 256, CV_8UC3, Scalar(0, 0, 0));
    for (int i = 0; i < 256; ++i) {
        line(histImage, Point(i, 256), Point(i, 256 - hist.at<float>(i)), Scalar(0, 0, 255), 1);
    }

    imshow("Hue Histogram", histImage);
}
