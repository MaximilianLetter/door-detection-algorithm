#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// Declare all used constants
int const RES = 180;

float const CONTRAST = 1.5;

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
	resize(image, image, Size(RES * ratio, RES));

	// Convert to grayscale
	Mat gray;
	cvtColor(image, gray, COLOR_BGR2GRAY);

	// Increase contrast
	gray.convertTo(gray, -1, CONTRAST, 0);

	// Display result
	imshow("Display window", gray);
	waitKey(0);

	return true;
}