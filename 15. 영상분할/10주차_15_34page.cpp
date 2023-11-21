#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

Mat preprocessing(Mat img)
{
	Mat gray, th_img;
	cvtColor(img, gray, COLOR_BGR2GRAY); // ��ӵ� ��ȯ
	GaussianBlur(gray, gray, Size (7, 7), 2, 2); // ����
	threshold(gray, th_img, 130, 255, THRESH_BINARY | THRESH_OTSU); // ����ȭ
	morphologyEx(th_img, th_img, MORPH_OPEN, Mat(), Point(-1, -1), 1); // ��������
	return th_img;
}

vector<RotatedRect> find_coins(Mat img)
{
	vector<vector<Point> > contours;
	findContours(img.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	vector<RotatedRect> circles;
	for (int i = 0; i< (int)contours.size(); i++)
	{
		RotatedRect mr = minAreaRect(contours[i]);
		mr.angle = (mr.size.width + mr.size.height) / 4.0f;

		if (mr.angle > 18) circles.push_back(mr);
	}
return circles;
}

int main()
{
	int coin_no = 20;
	String frame = format("C:/Users/Chan's Victus/Documents/class/Project/image/coin/%2d.png", coin_no);
	Mat image = imread(frame, 1);
	CV_Assert(image.data);
	Mat th_img = preprocessing(image);
	vector<RotatedRect > circles = find_coins(th_img);
	
	for (int i = 0; i < circles.size(); i++) {
		float radius = circles[i].angle; // ������ü ������
		circle(image, circles[i].center, radius, Scalar(0, 255, 0), 2); // ���� ǥ��
	}
	imshow("��ó������", th_img);
	imshow("��������", image);
	waitKey();
	return 0;
}

