#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

// �� �Լ� ����
bool compareRectSize(const Rect& a, const Rect& b) {
    return (a.width * a.height) > (b.width * b.height);
}

int main() {
    // ���� ������ ���� Haar Cascade ���� ���
    String cascadePath = cv::samples::findFile("../Project/haarcascade_car.xml"); // ���⿡ haarcascade_cars.xml ������ ��θ� �־��ּ���.

    // �̹��� ����
    String imagePath = cv::samples::findFile("../image/cars/test6.png"); 

    // �̹��� �ҷ�����
    Mat image = imread(imagePath);

    if (image.empty()) {
        cerr << "�̹����� �ҷ��� �� �����ϴ�." << endl;
        return -1;
    }

    // Haar Cascade �з��� �ʱ�ȭ
    CascadeClassifier carCascade;
    if (!carCascade.load(cascadePath)) {
        cerr << "Haar Cascade ������ �ҷ��� �� �����ϴ�." << endl;
        return -1;
    }

    // �׷��̽����Ϸ� ��ȯ
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    // �̹��� ��Ȱȭ
    equalizeHist(gray, gray);

    // �������� ����
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    Mat kernel2 = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    Mat opening, closing;
    morphologyEx(gray, opening, MORPH_OPEN, kernel);
    morphologyEx(opening, closing, MORPH_CLOSE, kernel2);

    // ���� ����
    vector<Rect> cars;
    carCascade.detectMultiScale(closing, cars, 1.1, 1, 0, Size(100, 100));

    // �̹��� ���� �̸����� Ȯ���� ����
    size_t lastDotPos = imagePath.find_last_of(".");
    string imageNameWithoutExtension = imagePath.substr(0, lastDotPos);

    // ����� ����� �ؽ�Ʈ ���� ����
    string outputFilePath = imageNameWithoutExtension + ".txt";
    ofstream outputFile(outputFilePath);

    // ũ������� ����
    sort(cars.begin(), cars.end(), compareRectSize);

    // �ִ� 5�������� ���
    size_t numCarsToPrint = min(cars.size(), static_cast<size_t>(5));

    // ����� ���� ǥ�� �� ��ǥ ���
    for (size_t i = 0; i < numCarsToPrint; ++i) {
        Rect car = cars[i];
        rectangle(image, car, Scalar(0, 255, 0), 1);

        // ��ǥ �� ũ�⸦ �ؽ�Ʈ ���Ͽ� ������ �����Ͽ� ���
        outputFile << i + 1 << '\t' << car.x << '\t' << car.y << '\t' << car.width << '\t' << car.height << endl;
    }

    // ���� �ݱ�
    outputFile.close();

    // ��� ���
    imshow("Detected Cars", image);
    waitKey(0);

    return 0;
}
