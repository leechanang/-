#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

// 비교 함수 정의
bool compareRectSize(const Rect& a, const Rect& b) {
    return (a.width * a.height) > (b.width * b.height);
}

int main() {
    // 차량 검출을 위한 Haar Cascade 파일 경로
    String cascadePath = cv::samples::findFile("../Project/haarcascade_car.xml"); // 여기에 haarcascade_cars.xml 파일의 경로를 넣어주세요.

    // 이미지 열기
    String imagePath = cv::samples::findFile("../image/cars/test6.png"); 

    // 이미지 불러오기
    Mat image = imread(imagePath);

    if (image.empty()) {
        cerr << "이미지를 불러올 수 없습니다." << endl;
        return -1;
    }

    // Haar Cascade 분류기 초기화
    CascadeClassifier carCascade;
    if (!carCascade.load(cascadePath)) {
        cerr << "Haar Cascade 파일을 불러올 수 없습니다." << endl;
        return -1;
    }

    // 그레이스케일로 변환
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    // 이미지 평활화
    equalizeHist(gray, gray);

    // 모폴로지 연산
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    Mat kernel2 = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    Mat opening, closing;
    morphologyEx(gray, opening, MORPH_OPEN, kernel);
    morphologyEx(opening, closing, MORPH_CLOSE, kernel2);

    // 차량 검출
    vector<Rect> cars;
    carCascade.detectMultiScale(closing, cars, 1.1, 1, 0, Size(100, 100));

    // 이미지 파일 이름에서 확장자 제거
    size_t lastDotPos = imagePath.find_last_of(".");
    string imageNameWithoutExtension = imagePath.substr(0, lastDotPos);

    // 결과를 출력할 텍스트 파일 생성
    string outputFilePath = imageNameWithoutExtension + ".txt";
    ofstream outputFile(outputFilePath);

    // 크기순으로 정렬
    sort(cars.begin(), cars.end(), compareRectSize);

    // 최대 5개까지만 출력
    size_t numCarsToPrint = min(cars.size(), static_cast<size_t>(5));

    // 검출된 차량 표시 및 좌표 기록
    for (size_t i = 0; i < numCarsToPrint; ++i) {
        Rect car = cars[i];
        rectangle(image, car, Scalar(0, 255, 0), 1);

        // 좌표 및 크기를 텍스트 파일에 탭으로 구분하여 기록
        outputFile << i + 1 << '\t' << car.x << '\t' << car.y << '\t' << car.width << '\t' << car.height << endl;
    }

    // 파일 닫기
    outputFile.close();

    // 결과 출력
    imshow("Detected Cars", image);
    waitKey(0);

    return 0;
}
