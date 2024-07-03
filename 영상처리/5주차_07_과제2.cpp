#include <opencv2/opencv.hpp>

using namespace cv;

int main() {
 
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the camera." << std::endl;
        return -1;
    }

    Mat frame;
    while (true) {

        cap >> frame;

        resize(frame, frame, Size(400, 300));

        Rect roi(30, 30, 320, 240);

        Mat blackImage = Mat::zeros(frame.size(), frame.type());

        frame(roi).copyTo(blackImage(roi));

        rectangle(blackImage, roi, Scalar(0, 0, 255), 2);

        imshow("Camera Window", blackImage);

        char c = (char)waitKey(25);
        if (c == 27)
            break;
    }

    cap.release();
    destroyAllWindows();

    return 0;
}