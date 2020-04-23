#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// Declare all used constants
const int RES = 180;

const float CONTRAST = 1.5;

const Size BLUR_KERNEL = Size(3, 3);
const double BLUR_SIGMA = 2.5;

const int CANNY_LOWER = 50;
const int CANNY_UPPER = 200;

// NOTE: these values need to be improved to ensure to always find the corners of a door
const int CORNERS_MAX = 40;
const double CORNERS_QUALITY = 0.05;
const double CORNERS_MIN_DIST = 10.0;
const int CORNERS_MASK_OFFSET = 10;

// Declare all used functions
bool detect(Mat image);

int main(int argc, char** argv)
{
	String fileName = "door_1.jpg";
	if (argc > 1)
	{
		fileName = argv[1] + String(".jpg");
	}

	Mat image;
	image = imread(samples::findFile("data/" + fileName), IMREAD_COLOR);

	if (image.empty()) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	namedWindow("Display window", WINDOW_AUTOSIZE);

	bool success = detect(image);
	
	if (success)
	{
		waitKey(0);
	}

	return 0;
}

bool detect(Mat image)
{
	// Scale image down
	int width = image.cols;
	int height = image.rows;
	float ratio = height / width;
	resize(image, image, Size(RES * ratio, RES), 0.0, 0.0, INTER_AREA);
	// NOTE: different interpolation methods can be used

	// Convert to grayscale
	Mat gray;
	cvtColor(image, gray, COLOR_BGR2GRAY);

	// Increase contrast
	gray.convertTo(gray, -1, CONTRAST, 0);

	// Blur the image
	Mat blurred;
	GaussianBlur(gray, blurred, BLUR_KERNEL, BLUR_SIGMA);

	// Generate edges
	Mat edges;
	Canny(blurred, edges, CANNY_LOWER, CANNY_UPPER);

	// Generate mask and find corners
	vector<Point2f> corners;
	Mat mask;

	mask = Mat::zeros(image.size(), CV_8U);
	Rect rect = Rect(CORNERS_MASK_OFFSET, CORNERS_MASK_OFFSET, image.size().width - CORNERS_MASK_OFFSET, image.size().height - CORNERS_MASK_OFFSET);
	mask(rect) = 1;

	goodFeaturesToTrack(blurred, corners, CORNERS_MAX, CORNERS_QUALITY, CORNERS_MIN_DIST, mask);

	// Display result
	cout << corners.size();
	for (int i = 0; i < corners.size(); i++)
	{
		circle(image, corners[i], 3, Scalar(0, 255, 0), FILLED);
	}
	imshow("Display window", image);
	waitKey(0);

	return true;
}