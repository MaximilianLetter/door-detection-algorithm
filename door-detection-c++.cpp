#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <chrono>

#define _USE_MATH_DEFINES
#include <math.h>

using namespace cv;
using namespace std;

Mat globalImg;

// Declare all used constants
const int RES = 360;

const float ROI_WIDTH = 0.8;
const float ROI_HEIGHT = 0.125;

const float CONTRAST = 1.2;

// Blur constants
const Size BLUR_KERNEL = Size(3, 3);
const float BLUR_SIGMA = 2.5;

// Canny constants
const int CANNY_LOWER = 50;
const int CANNY_UPPER = 200;

// NOTE: these values need to be improved to ensure to always find the corners of a door
// Corner detection constants
const int CORNERS_MAX = 50;
//const float CORNERS_QUALITY = 0.01;
const float CORNERS_BOT_QUALITY = 0.05;
const float CORNERS_TOP_QUALITY = 0.01;
const float CORNERS_MIN_DIST = 15.0;
//const int CORNERS_MASK_OFFSET = 10;
const bool CORNERS_HARRIS = false;

// Vertical lines constants
//const float LINE_MAX = 0.9;
const float LINE_MIN = 0.3;
const float LINE_ANGLE_MIN = 0.875; // RAD

// Rectangles constants
const float ANGLE_MAX = 0.175; // RAD
const float LENGTH_DIFF_MAX = 0.12;
const float ASPECT_RATIO_MIN = 0.3;
const float ASPECT_RATIO_MAX = 0.8;
const float LENGTH_HOR_DIFF_MAX = 1.2;
const float LENGTH_HOR_DIFF_MIN = 0.7;
const float RECTANGLE_THRESH = 10.0;
const float RECTANGLE_OPPOSITE_THRESH = 10.0;

// Comparison of rectangles to edges constants
const float RECT_THRESH = 0.85;
const float LINE_THRESH = 0.5;
const int LINE_WIDTH = 4;
const float BOT_LINE_BONUS = 0.25;

// Selection of best candidate constants
const float UPVOTE_FACTOR = 1.2;
const float DOOR_IN_DOOR_DIFF_THRESH = 18.0; // Divider of image height
const float COLOR_DIFF_THRESH = 50.0;
const float ANGLE_DEVIATION_THRESH = 10.0;
const float CLOSE_TO_INPUT_THRESH = 20.0;

// Declare all used functions
bool detect(Mat& image, Point2f point, vector<Point2f>& result);
vector<vector<Point2f>> cornersToVertLines(vector<Point2f> cornersBot, vector<Point2f> cornersTop);
vector<vector<Point2f>> vertLinesToRectangles(vector<vector<Point2f>> lines);
float compareRectangleToEdges(vector<Point2f> rect, Mat edges);
vector<Point2f> selectBestCandidate(vector<vector<Point2f>> candidates, vector<float> scores, Point2f inputPoint, Mat gray);

float getDistance(Point2f p1, Point2f p2);
float getOrientation(Point2f p1, Point2f p2);
float getCornerAngle(Point2f p1, Point2f p2, Point2f p3);

void clickCallBack(int event, int x, int y, int flags, void* userdata);

int main(int argc, char** argv)
{
	String fileName = "door_1.jpg";
	bool video = false;
	if (argc > 2)
	{
		video = true;
		fileName = argv[2] + String(".mp4");
	}
	else if (argc > 1)
	{
		fileName = argv[1] + String(".jpg");
	}

	namedWindow("Display window", WINDOW_AUTOSIZE);
	namedWindow("Edges window", WINDOW_AUTOSIZE);
	namedWindow("Dev window", WINDOW_AUTOSIZE);

	setMouseCallback("Display window", clickCallBack, NULL);

	if (!video)
	{
		Mat image;
		image = imread(samples::findFile("data/" + fileName), IMREAD_COLOR);

		if (image.empty()) // Check for invalid input
		{
			cout << "Could not open or find the image" << std::endl;
			return -1;
		}

		vector<Point2f> result = {};
		bool success = detect(image, Point2f(), result);

		if (result.size() > 0)
		{
			cout << "DOOR FOUND" << endl;

			// Scale up to match input size (6x for FHD, 4x for HD)
			for (int i = 0; i < result.size(); i++)
			{
				result[i] = result[i] * 17;
			}

			line(image, result[0], result[1], Scalar(255, 255, 0), 5);
			line(image, result[1], result[2], Scalar(255, 255, 0), 5);
			line(image, result[2], result[3], Scalar(255, 255, 0), 5);
			line(image, result[3], result[0], Scalar(255, 255, 0), 5);
		}
		else {
			cout << "NO DOOR" << endl;
		}

		resize(image, image, image.size() / 6);
		imshow("Display window", image);
		waitKey(0);
	}
	else
	{
		cout << fileName;
		VideoCapture cap(samples::findFile("data/" + fileName));

		if (!cap.isOpened())
		{
			cout << "Error opening video stream or file" << endl;
			
			return -1;
		}

		// frames get loaded rotated -> flip width and height
		int frameWidth = cap.get(CAP_PROP_FRAME_HEIGHT);
		int frameHeight = cap.get(CAP_PROP_FRAME_WIDTH);
		cout << frameWidth << " " << frameHeight;
		int codec = VideoWriter::fourcc('M', 'J', 'P', 'G');
		VideoWriter video("./results/output.avi", codec, 25.0, Size(frameWidth, frameHeight));

		Mat frame;

		while (1)
		{
			cap >> frame;

			if (frame.empty()) break;

			rotate(frame, frame,  ROTATE_90_CLOCKWISE);
			resize(frame, frame, frame.size() / 4);


			frame.copyTo(globalImg);

			imshow("Display window", frame);
			video.write(frame);

			char c = (char)waitKey(25);
			if (c == 27) break;
		}

		cap.release();
		video.release();
	}
	
	cv::destroyAllWindows();

	return 0;
}

bool detect(Mat& input, Point2f inputPoint, vector<Point2f>& result)
{
	auto t1 = chrono::steady_clock::now();

	Mat image;
	input.copyTo(image);

	// Scale image down
	int width = image.size().width;
	int height = image.size().height;
	//float ratio = float(height) / float(width);
	//resize(image, image, Size(RES, int(RES * ratio)), 0.0, 0.0, INTER_AREA);
	// NOTE: It was already resized

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
	imshow("Edges window", edges);

	// Find ROI's based on user input
	Mat maskBot, maskTop;
	vector<Point2f> cornersBot, cornersTop;


	// Find ROI based on inputPoint
	int roiBotWidth = width * ROI_WIDTH;
	int roiBotHeight = height * ROI_HEIGHT;
	Point2f roiPoint = Point2f(inputPoint.x - roiBotWidth / 2, inputPoint.y - roiBotHeight / 2);
	Rect roiBot = Rect(roiPoint.x, roiPoint.y, roiBotWidth, roiBotHeight);

	// Cut overlapping parts off
	roiBot = roiBot & Rect(0, 0, width, height);

	maskBot = Mat::zeros(image.size(), CV_8U);
	maskBot(roiBot) = 1;

	
	// Extract top corners to join
	int lowLineBot = roiBot.y + roiBot.height;
	int roiTopHeight = lowLineBot - (LINE_MIN * height);
	//line(blurred, Point2f(5, roiTopHeight), Point2f(width-5, roiTopHeight), 255, 3);

	// NOTE: order matters
	Point polygonPoints[4] = {
		Point(0, 0),
		Point(width, 0),
		Point(roiBot.x + roiBot.width, roiTopHeight),
		Point(roiBot.x, roiTopHeight)
	};

	maskTop = Mat::zeros(image.size(), CV_8U);
    fillConvexPoly(maskTop, polygonPoints, 4, cv::Scalar(255));

	
	goodFeaturesToTrack(blurred, cornersBot, CORNERS_MAX, CORNERS_BOT_QUALITY, CORNERS_MIN_DIST, maskBot, 3, CORNERS_HARRIS);
	goodFeaturesToTrack(blurred, cornersTop, CORNERS_MAX, CORNERS_TOP_QUALITY, CORNERS_MIN_DIST, maskTop, 3, CORNERS_HARRIS);


	for (int i = 0; i < cornersBot.size(); i++)
	{
		circle(blurred, cornersBot[i], 3, 255, -1);
	}

	for (int i = 0; i < cornersTop.size(); i++)
	{
		circle(blurred, cornersTop[i], 3, 255, -1);
	}

	// Connect corners to vertical lines
	vector<vector<Point2f>> lines = cornersToVertLines(cornersBot, cornersTop);

	// Group corners based on found lines to rectangles
	vector<vector<Point2f>> rectangles = vertLinesToRectangles(lines);

	// NOTE: this could be done in vertLinesToRectangles aswell
	// Compare the found rectangles to the edge image
	vector<vector<Point2f>> candidates;
	vector<float> scores;
	for (int i = 0; i < rectangles.size(); i++)
	{
		float result = compareRectangleToEdges(rectangles[i], edges);

		if (result > RECT_THRESH)
		{
			candidates.push_back(rectangles[i]);
			scores.push_back(result);
		}
	}
	for (int i = 0; i < candidates.size(); i++)
	{
		/*polylines(image, rectangles[i], true, Scalar(255, 255, 0), 1);*/
		line(blurred, candidates[i][0], candidates[i][1], Scalar(255, 255, 0), 2);
		line(blurred, candidates[i][1], candidates[i][2], Scalar(255, 255, 0), 2);
		line(blurred, candidates[i][2], candidates[i][3], Scalar(255, 255, 0), 2);
		line(blurred, candidates[i][3], candidates[i][0], Scalar(255, 255, 0), 2);
	}
	imshow("Dev window", blurred);

	// Select the best candidate out of the given rectangles
	if (candidates.size())
	{
		vector<Point2f> door = selectBestCandidate(candidates, scores, inputPoint, gray);
		result = door;
	}
	//cout << rectangles.size() << "; " << candidates.size() << endl;

	auto t2 = chrono::steady_clock::now();
	auto duration = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
	cout << duration << endl;

	return false;
}

// Group corners to vertical lines that represent the door posts
vector<vector<Point2f>> cornersToVertLines(vector<Point2f> cornersBot, vector<Point2f> cornersTop)
{
	/*float lengthMax = LINE_MAX * height;
	float lengthMin = LINE_MIN * height;*/

	vector<vector<Point2f>> lines;

	for (int i = 0; i < cornersBot.size(); i++)
	{
		for (int j = 0; j < cornersTop.size(); j++)
		{
			/*if (j <= i) continue;

			float distance = getDistance(corners[i], corners[j]);
			if (distance < lengthMin || distance > lengthMax)
			{
				continue;
			}*/

			float orientation = getOrientation(cornersBot[i], cornersTop[j]);
			if (orientation < LINE_ANGLE_MIN)
			{
				continue;
			}

			// Sort by y-value, so that the high points are first
			/*vector<Point2f> line;
			if (corners[i].y < corners[j].y)
			{
				line = { corners[i], corners[j] };
			}
			else {
				line = { corners[j], corners[i] };
			}*/

			vector<Point2f> line = { cornersTop[j], cornersBot[i] };
			lines.push_back(line);
		}
	}

	return lines;
}

// Group rectangles that represent door candidates out of vertical lines
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

			// NOTE: both of these values were calculated before,
			// maybe store them for reusage
			// Check if length difference of lines is close
			float length1 = getDistance(lines[i][0], lines[i][1]);
			// TODO save the values, OR length1 in first for loop

			float length2 = getDistance(lines[j][0], lines[j][1]);
			float lengthDiff = abs(length1 - length2);
			float lengthAvg = (length1 + length2) / 2;

			if (lengthDiff > (lengthAvg * LENGTH_DIFF_MAX))
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

			// NOTE: these tests might not be necessary if corner angle test exists
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

			float angles[4];
			angles[0] = getCornerAngle(lines[i][1], lines[i][0], lines[j][0]);
			angles[1] = getCornerAngle(lines[i][0], lines[j][0], lines[j][1]);
			angles[2] = getCornerAngle(lines[j][0], lines[j][1], lines[i][1]);
			angles[3] = getCornerAngle(lines[j][1], lines[i][1], lines[i][0]);

			bool rectangular = true;

			for (int k = 0; k < 4; k++)
			{
				if (abs(90.0 - angles[k]) > RECTANGLE_THRESH)
				{
					int kOpp = (k + 2) % 4;

					if (abs(180.0 - (angles[k] + angles[kOpp]) > RECTANGLE_OPPOSITE_THRESH))
					{
						rectangular = false;
						break;
					}
				}
			}

			if (!rectangular) continue;

			// Sort in order: leftBot > leftTop > rightTop > rightBot
			vector<Point2f> group = { lines[i][1], lines[i][0], lines[j][0], lines[j][1] };
			rects.push_back(group);
		}
	}

	return rects;
}

// Compare a possible rectangle with the existing edges in the edge image
float compareRectangleToEdges(vector<Point2f> rect, Mat edges)
{
	float result = 0.0;
	float bottomBonus = 0.0;

	for (int i = 0; i < rect.size(); i++)
	{
		// Next point to connect
		int j = (i + 1) % 4;

		Mat mask = Mat::zeros(edges.size(), CV_8U);
		line(mask, rect[i], rect[j], 1, LINE_WIDTH);

		// While this works, there might be a better option without copy
		Mat roi;
		edges.copyTo(roi, mask);

		float lineLength = getDistance(rect[i], rect[j]);
		float fillRatio = min(float(1.0), countNonZero(roi) / lineLength);

		if (i < 3)
		{
			if (fillRatio < LINE_THRESH)
			{
				return 0.0;
			}

			result += fillRatio;
		}
		else
		{
			// Bottom line
			bottomBonus = fillRatio * BOT_LINE_BONUS;
		}
	}

	// Get average fillRatio for all lines but bottom line
	result = (result / 3) + bottomBonus;

	return result;
}

// Select the candidate by comparing their scores, score boni if special requirements are met
vector<Point2f> selectBestCandidate(vector<vector<Point2f>> candidates, vector<float> scores, Point2f inputPoint, Mat gray)
{
	cout << candidates.size() << "size" << endl;
	for (int i = 0; i < candidates.size(); i++)
	{
		// NOTE: this can be a trap and should be somewhat dynamic
		// Test if inner content has a different color average
		int left = (candidates[i][0].x + candidates[i][1].x) / 2;
		int top = (candidates[i][1].y + candidates[i][2].y) / 2;
		int right = (candidates[i][2].x + candidates[i][3].x) / 2;
		int bottom = (candidates[i][3].y + candidates[i][0].y) / 2;

		// This whole process of masking the image seems like a workaround
		Rect rect = Rect(Point2i(left, bottom), Point2i(right, top));
		Mat mask = Mat::zeros(gray.size(), CV_8U);
		rectangle(mask, rect, 1);

		double inner = mean(gray, mask)[0];
		mask = 1 - mask;
		double outer = mean(gray, mask)[0];

		if (abs(inner - outer) > COLOR_DIFF_THRESH)
		{
			scores[i] *= UPVOTE_FACTOR;
		}

		// Test for corner angles
		/*float angle0 = getCornerAngle(candidates[i][3], candidates[i][0], candidates[i][1]);
		float angle1 = getCornerAngle(candidates[i][0], candidates[i][1], candidates[i][2]);
		float angle2 = getCornerAngle(candidates[i][1], candidates[i][2], candidates[i][3]);
		float angle3 = getCornerAngle(candidates[i][2], candidates[i][3], candidates[i][0]);

		float botAngleDiff = abs(angle0 - angle3);
		float topAngleDiff = abs(angle1 - angle2);

		if (botAngleDiff < ANGLE_DEVIATION_THRESH && topAngleDiff < ANGLE_DEVIATION_THRESH)
		{
			scores[i] *= UPVOTE_FACTOR;
		}*/

		// Check if there is a door with the same top corners
		for (int j = 0; j < candidates.size(); j++)
		{
			if (j == i) continue;
			// Gets upvoted x times for x doors with same topPoints -> defeating all other candidates that do not have multiple doors on it

			if (candidates[i][1] == candidates[j][1] && candidates[i][2] == candidates[j][2])
			{
				scores[i] = scores[i] * UPVOTE_FACTOR;
			}
		}

		// Check how close the bottom line center is to the input point
		Point2f bottomCenterPoint = (candidates[i][3] + candidates[i][0]) / 2;

		if (getDistance(bottomCenterPoint, inputPoint) < CLOSE_TO_INPUT_THRESH)
		{
			scores[i] *= UPVOTE_FACTOR;
		}
		//cout << "POINT1: " << candidates[i][3].x << " " << candidates[i][3].y << endl;
		//cout << "POINT2: " << candidates[i][0].x << " " << candidates[i][0].y << endl;
		//cout << "POINTBetween: " << bottomCenterPoint.x << " " << bottomCenterPoint.y << endl;

		//cout << scores[i];
	}

	int index = max_element(scores.begin(), scores.end()) - scores.begin();
	//cout << " winner " << index;
	vector<Point2f> door = candidates[index];

	return door;
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
		return abs((2 / M_PI) * atan(abs(p1.y - p2.y) / abs(p1.x - p2.x)));
	}
	else
	{
		return 180.0;
	}
}

// Get the angle between three points forming two lines
float getCornerAngle(Point2f p1, Point2f p2, Point2f p3)
{
	Point2f p12 = p1 - p2;
	Point2f p32 = p3 - p2;

	float angle = p12.dot(p32) / (norm(p12) * norm(p32));
	angle = abs(acos(angle) * 180/M_PI);

	return angle;
}

void clickCallBack(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		cout << "Position (" << x << ", " << y << ")" << endl;
		Point2f point = Point2f(x, y);

		vector<Point2f> result = {};
		bool success = detect(globalImg, point, result);

		if (result.size() > 0)
		{
			line(globalImg, result[0], result[1], Scalar(255, 255, 0), 2);
			line(globalImg, result[1], result[2], Scalar(255, 255, 0), 2);
			line(globalImg, result[2], result[3], Scalar(255, 255, 0), 2);
			line(globalImg, result[3], result[0], Scalar(255, 255, 0), 2);
		}

		imshow("result", globalImg);
		waitKey(0);
	}
}