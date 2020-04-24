#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

#define _USE_MATH_DEFINES
#include <math.h>

using namespace cv;
using namespace std;

// Declare all used constants
const int RES = 180;

const float CONTRAST = 1.5;

// Blur constants
const Size BLUR_KERNEL = Size(3, 3);
const float BLUR_SIGMA = 2.5;

// Canny constants
const int CANNY_LOWER = 50;
const int CANNY_UPPER = 200;

// NOTE: these values need to be improved to ensure to always find the corners of a door
// Corner detection constants
const int CORNERS_MAX = 40;
const float CORNERS_QUALITY = 0.05;
const float CORNERS_MIN_DIST = 10.0;
const int CORNERS_MASK_OFFSET = 10;

// Vertical lines constants
const float LINE_MAX = 0.85;
const float LINE_MIN = 0.3;
const float LINE_ANGLE_MIN = 0.875; // RAD

// Rectangles constants
const float ANGLE_MAX = 0.175; // RAD
const float LENGTH_DIFF_MAX = 0.15;
const float ASPECT_RATIO_MIN = 0.35;
const float ASPECT_RATIO_MAX = 0.7;
const float LENGTH_HOR_DIFF_MAX = 1.1;
const float LENGTH_HOR_DIFF_MIN = 0.7;

// Declare all used functions
bool detect(Mat image);
vector<vector<Point2f>> cornersToVertLines(vector<Point2f> corners, int height);
vector<vector<Point2f>> vertLinesToRectangles(vector<vector<Point2f>> lines);

float getDistance(Point2f p1, Point2f p2);
float getOrientation(Point2f p1, Point2f p2);

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
	int width = image.size().width;
	int height = image.size().height;
	float ratio = float(height) / float(width);
	resize(image, image, Size(RES, int(RES * ratio)), 0.0, 0.0, INTER_AREA);
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

	// Connect corners to vertical lines
	vector<vector<Point2f>> lines = cornersToVertLines(corners, int(RES * ratio));

	// Group corners based on found lines to rectangles
	vector<vector<Point2f>> rectangles = vertLinesToRectangles(lines);

	// Display results

	cout << corners.size();
	for (int i = 0; i < corners.size(); i++)
	{
		circle(image, corners[i], 3, Scalar(0, 255, 0), FILLED);
	}

	//cout << lines.size();
	//for (int i = 0; i < lines.size(); i++)
	//{
	//	line(image, lines[i][0], lines[i][1], Scalar(0, 0, 255), 1);
	//}

	cout << rectangles.size();
	for (int i = 0; i < rectangles.size(); i++)
	{
		/*polylines(image, rectangles[i], true, Scalar(255, 255, 0), 1);*/
		line(image, rectangles[i][0], rectangles[i][1], Scalar(255, 255, 0), 1);
		line(image, rectangles[i][1], rectangles[i][2], Scalar(255, 255, 0), 1);
		line(image, rectangles[i][2], rectangles[i][3], Scalar(255, 255, 0), 1);
		line(image, rectangles[i][3], rectangles[i][0], Scalar(255, 255, 0), 1);
	}

	imshow("Display window", image);

	return true;
}

// Group corners to vertical lines that represent the door posts. 
vector<vector<Point2f>> cornersToVertLines(vector<Point2f> corners, int height)
{
	float lengthMax = LINE_MAX * height;
	float lengthMin = LINE_MIN * height;

	vector<vector<Point2f>> lines;
	vector<bool> done;

	for (int i = 0; i < corners.size(); i++)
	{
		for (int j = 0; j < corners.size(); j++)
		{
			if (j <= i) continue;

			float distance = getDistance(corners[i], corners[j]);
			if (distance < lengthMin || distance > lengthMax)
			{
				continue;
			}

			float orientation = getOrientation(corners[i], corners[j]);
			if (orientation < LINE_ANGLE_MIN)
			{
				continue;
			}

			// Sort by y-value, so that the high points are first
			vector<Point2f> line;
			if (corners[i].y < corners[j].y)
			{
				line = { corners[i], corners[j] };
			}
			else {
				line = { corners[j], corners[i] };
			}
			lines.push_back(line);
		}
	}

	return lines;
}

vector<vector<Point2f>> vertLinesToRectangles(vector<vector<Point2f>> lines)
{
	vector<vector<Point2f>> rects;

	for (int i = 0; i < lines.size(); i++)
	{
		for (int j = 0; j < lines.size(); j++)
		{
			if (j <= i) continue;

			// Only build rectangle if the two lines are completely distinct
			if ((lines[i][0] == lines[j][0]) || (lines[i][0] == lines[j][1]) || (lines[i][1] == lines[j][0]) || (lines[i][1] == lines[j][1]))
			{
				continue;
			}

			// NOTE: both of these values was calculated before,
			// maybe store them for reusage
			// Check if length difference of lines is close
			float length1 = getDistance(lines[i][0], lines[i][1]);
			float length2 = getDistance(lines[i][0], lines[i][1]);
			float lengthDiff = abs(length1 - length2);
			float lengthAvg = (length1 + length2) / 2;

			if ((lengthDiff > (lengthAvg * LENGTH_DIFF_MAX)))
			{
				continue;
			}

			// Check if top distance is in range of the given aspect ratio
			float lengthMin = lengthAvg * ASPECT_RATIO_MIN;
			float lengthMax = lengthAvg * ASPECT_RATIO_MAX;

			float distanceTop = getDistance(lines[i][0], lines[j][0]);
			if (distanceTop < lengthMin || distanceTop > lengthMax)
			{
				continue;
			}

			// Check if bottom distance is similar to top distance
			float distanceBot = getDistance(lines[i][1], lines[j][1]);
			if (distanceBot > (distanceTop * LENGTH_HOR_DIFF_MAX)
				|| distanceBot < (distanceTop * LENGTH_HOR_DIFF_MIN))
			{
				continue;
			}

			// Test orientation of top horizontal line
			float orientationTop = getOrientation(lines[i][0], lines[j][0]);
			if (orientationTop > ANGLE_MAX)
			{
				continue;
			}

			// Test orientation of bottom horizontal line
			float orientationBot = getOrientation(lines[i][0], lines[j][0]);
			if (orientationBot > ANGLE_MAX)
			{
				continue;
			}

			// Sort in order: leftBot > leftTop > rightTop > rightBot
			vector<Point2f> group = { lines[i][1], lines[i][0], lines[j][0], lines[j][1] };
			rects.push_back(group);
		}
	}

	return rects;
}

// Get the distance between two points
float getDistance(Point2f p1, Point2f p2)
{
	return sqrt(pow((p1.x - p2.x), 2) + pow((p1.y - p2.y), 2));
}

// Get the orientation of the line consisting of two points
float getOrientation(Point2f p1, Point2f p2)
{
	if (p1.x != p2.x)
	{
		return (2 / M_PI) * atan(abs(p1.y - p2.y) / abs(p1.x - p2.x));
	}
	else
	{
		return 180.0;
	}
}