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
const int RES = 240;

const float ROI_WIDTH = 0.8;
const float ROI_HEIGHT = 0.125;

// Blur constants
const Size BLUR_KERNEL = Size(3, 3);
const float BLUR_SIGMA = 2.5;

// Canny constants
const double CANNY_LOWER = 0.33; // NOTE: The lower threshold is lower than most canny auto thresholds, but necessary to catch some door edges
const double CANNY_UPPER = 1.33;

// NOTE: these values need to be improved to ensure to always find the corners of a door
// Corner detection constants
const int CORNERS_MAX = 150;
const float CORNERS_BOT_QUALITY = 0.01;
const float CORNERS_TOP_QUALITY = 0.01;
const float CORNERS_MIN_DIST = 12.0;

// Hough line constants
const int HOUGH_LINE_WIDTH = 5;
const int HOUGH_LINE_ADDITIONAL_WIDTH = 2;
const int HOUGH_LINE_WIDTH_MAX = 20;
const float HOUGH_LINE_DIFF_THRESH_PIXEL = 15;
const float HOUGH_LINE_DIFF_THRESH_ANGLE = 0.25;
const int HOUGH_COUNT_LIMIT = 20;

// Vertical lines constants
const float LINE_MIN = 0.4;

// Rectangles constants
const float ANGLE_MAX = 0.175; // RAD
const float LENGTH_DIFF_MAX = 0.12;
const float ASPECT_RATIO_MIN = 0.3;
const float ASPECT_RATIO_MAX = 0.6; // from 0.6
const float LENGTH_HOR_DIFF_MAX = 1.2;
const float LENGTH_HOR_DIFF_MIN = 0.7;
const float RECTANGLE_THRESH = 10.0;
const float RECTANGLE_OPPOSITE_THRESH = 10.0;

// Comparison of rectangles to edges constants
const float RECT_THRESH = 0.8;
const float LINE_THRESH = 0.5;
const int LINE_WIDTH = 8;

// Selection of best candidate constants
const float GOAL_INPUT_RANGE = 0.5;
const float GOAL_RATIO = 0.45;
const float GOAL_RATIO_RANGE = 0.15;
const float GOAL_ANGLES = 90;
const float GOAL_ANGLES_DIFF_RANGE = 20;

// Declare all used functions
bool detect(Mat image, Point2f point, vector<Point2f>& result);
vector<vector<Point2f>> cornersToVertLines(vector<Point2f> cornersBot, vector<Point2f> cornersTop, vector<Vec2f> houghLines, vector<int> houghLinesWidth, Size size);
vector<vector<Point2f>> vertLinesToRectangles(vector<vector<float>>& rectInnerAngles, vector<vector<Point2f>> lines);
float compareRectangleToEdges(vector<Point2f> rect, Mat edges);
vector<Point2f> selectBestCandidate(vector<vector<Point2f>> candidates, vector<float> scores, Point2f inputPoint, vector<vector<float>> rectInnerAngles, Size size);

float getDistance(Point2f p1, Point2f p2);
float getOrientation(Point2f p1, Point2f p2);
float getCornerAngle(Point2f p1, Point2f p2, Point2f p3);
double getMedian(Mat channel);

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

			resize(frame, frame, Size(427, 240));
			rotate(frame, frame,  ROTATE_90_CLOCKWISE);

			frame.copyTo(globalImg);

			imshow("Display window", frame);
			//video.write(frame);

			char c = (char)waitKey(25);
			if (c == 27) break;
		}

		cap.release();
		video.release();
	}
	
	cv::destroyAllWindows();

	return 0;
}

bool detect(Mat input, Point2f inputPoint, vector<Point2f>& result)
{
	auto t1 = chrono::steady_clock::now();

	Mat image;
	input.copyTo(image);

	// Scale image down
	int width = image.size().width;
	int height = image.size().height;

	// Convert to grayscale
	Mat imgGray;
	cvtColor(image, imgGray, COLOR_BGR2GRAY);

	// Blur the image
	Mat blurred;
	GaussianBlur(imgGray, blurred, BLUR_KERNEL, BLUR_SIGMA);

	auto tPRE = chrono::steady_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>(tPRE - t1).count();
	cout << "PRE : " << duration << endl;

	// Generate edges
	Mat edges;
	double median = getMedian(blurred);
	
	// Very dark images can go to values like 9, resulting in extremely noisy images
	median = max((double)30, median);

	double lowerThresh = max((double)0, (CANNY_LOWER * median));
	double higherThresh = min((double)255, (CANNY_UPPER * median));

	Canny(blurred, edges, lowerThresh, higherThresh);
	//imshow("Edges window", edges);

	auto tEDGES = chrono::steady_clock::now();
	duration = chrono::duration_cast<chrono::microseconds>(tEDGES - tPRE).count();
	cout << "EDGES : " << duration << endl;

	// Generate hough lines
	vector<Vec2f> houghLines;
	int thresh = (int)(imgGray.size().height * 0.25);
	HoughLines(edges, houghLines, 1, CV_PI / 180, thresh, 0, 0);
	vector<Vec2f> filteredHoughLines;
	vector<int> filteredHoughLinesWidth;

	// Go through lines and merge them into bigger lines
	for (size_t h = 0; h < houghLines.size(); h++)
	{
		bool lineDone = false;

		for (int f = 0; f < filteredHoughLines.size(); f++)
		{
			Vec2f diff = houghLines[h] - filteredHoughLines[f];
			if (abs(diff[0]) < HOUGH_LINE_DIFF_THRESH_PIXEL && abs(diff[1]) < HOUGH_LINE_DIFF_THRESH_ANGLE)
			{
				filteredHoughLines[f] = (filteredHoughLines[f] + houghLines[h]) / 2;
				int width = filteredHoughLinesWidth[f] + HOUGH_LINE_ADDITIONAL_WIDTH;
				filteredHoughLinesWidth[f] = min(width, HOUGH_LINE_WIDTH_MAX);
				lineDone = true;
				break;
			}
		}

		if (lineDone) continue;

		filteredHoughLines.push_back(houghLines[h]);
		filteredHoughLinesWidth.push_back(HOUGH_LINE_WIDTH);
	}

	auto tHOUGH = chrono::steady_clock::now();
	duration = chrono::duration_cast<chrono::microseconds>(tHOUGH - tPRE).count();
	cout << "HOUGH : " << duration << endl;


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

	
	goodFeaturesToTrack(blurred, cornersBot, CORNERS_MAX, CORNERS_BOT_QUALITY, CORNERS_MIN_DIST, maskBot, 3);
	goodFeaturesToTrack(blurred, cornersTop, CORNERS_MAX, CORNERS_TOP_QUALITY, CORNERS_MIN_DIST, maskTop, 3);

	auto tCORNERS = chrono::steady_clock::now();
	duration = chrono::duration_cast<chrono::microseconds>(tCORNERS - tHOUGH).count();
	cout << "CORNERS : " << duration << endl;


	/*for (int i = 0; i < cornersBot.size(); i++)
	{
		circle(blurred, cornersBot[i], 3, 255, -1);
	}

	for (int i = 0; i < cornersTop.size(); i++)
	{
		circle(blurred, cornersTop[i], 3, 255, -1);
	}*/

	// Connect corners to vertical lines
	vector<vector<Point2f>> lines = cornersToVertLines(cornersBot, cornersTop, filteredHoughLines, filteredHoughLinesWidth, imgGray.size());

	auto tVERTLINES = chrono::steady_clock::now();
	duration = chrono::duration_cast<chrono::microseconds>(tVERTLINES - tCORNERS).count();
	cout << "VERTLINES : " << duration << endl;

	// Group corners based on found lines to rectangles

	vector<vector<float>> rectInnerAngles;
	vector<vector<Point2f>> rectangles = vertLinesToRectangles(rectInnerAngles, lines);

	auto tQUADRANGLES = chrono::steady_clock::now();
	duration = chrono::duration_cast<chrono::microseconds>(tQUADRANGLES - tVERTLINES).count();
	cout << "QUADRANGLES : " << duration << endl;

	// NOTE: this could be done in vertLinesToRectangles aswell
	// Compare the found rectangles to the edge image
	vector<vector<Point2f>> candidates;
	vector<vector<float>> updInnerAngles;
	vector<float> scores;
	for (int i = 0; i < rectangles.size(); i++)
	{
		float result = compareRectangleToEdges(rectangles[i], edges);

		if (result > RECT_THRESH)
		{
			candidates.push_back(rectangles[i]);
			updInnerAngles.push_back(rectInnerAngles[i]);
			scores.push_back(result);
		}
	}
	rectInnerAngles = updInnerAngles;


	auto tCANDIDATES = chrono::steady_clock::now();
	duration = chrono::duration_cast<chrono::microseconds>(tCANDIDATES - tQUADRANGLES).count();
	cout << "CANDIDATES : " << duration << endl;

	//for (int i = 0; i < candidates.size(); i++)
	//{
	//	/*polylines(image, rectangles[i], true, Scalar(255, 255, 0), 1);*/
	//	line(blurred, candidates[i][0], candidates[i][1], Scalar(255, 255, 0), 2);
	//	line(blurred, candidates[i][1], candidates[i][2], Scalar(255, 255, 0), 2);
	//	line(blurred, candidates[i][2], candidates[i][3], Scalar(255, 255, 0), 2);
	//	line(blurred, candidates[i][3], candidates[i][0], Scalar(255, 255, 0), 2);
	//}
	//imshow("Dev window", blurred);

	// Select the best candidate out of the given rectangles
	if (candidates.size())
	{
		vector<Point2f> door = selectBestCandidate(candidates, scores, inputPoint, rectInnerAngles, imgGray.size());
		result = door;
	}
	//cout << rectangles.size() << "; " << candidates.size() << endl;

	


	auto t2 = chrono::steady_clock::now();

	duration = chrono::duration_cast<chrono::microseconds>(t2 - tCANDIDATES).count();
	cout << "SELECTION : " << duration << endl;

	duration = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
	cout << "_____ overall: " << duration << endl;

	return false;
}

// Group corners to vertical lines that represent the door posts
vector<vector<Point2f>> cornersToVertLines(vector<Point2f> cornersBot, vector<Point2f> cornersTop, vector<Vec2f> houghLines, vector<int> houghLinesWidth, Size size)
{
	vector<vector<Point2f>> lines;

	Mat houghMat;
	Rect fullRect = Rect(cv::Point(), size);
	/*int linesComputed = 0;*/

	for (size_t h = 0; h < houghLines.size(); h++)
	{
		houghMat = Mat::zeros(size, CV_8U);

		float rho = houghLines[h][0], theta = houghLines[h][1];

		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));

		float angle = abs(atan2(pt2.y - pt1.y, pt2.x - pt1.x) * 180.0 / CV_PI);
		if (angle < 80 || angle > 100)
		{
			continue;
		}
		//linesComputed++;

		//// houghLines are ordered by votes, therefor weaker lines can be omitted
		//if (linesComputed > HOUGH_COUNT_LIMIT)
		//{
		//	continue;
		//}

		line(houghMat, pt1, pt2, 1, houghLinesWidth[h], LINE_AA);

		vector<Point2f> houghPoints = {};
		for (int i = 0; i < cornersTop.size(); i++)
		{
			if (fullRect.contains(cornersTop[i]) && houghMat.at<uchar>(cornersTop[i]))
			{
				for (int j = 0; j < cornersBot.size(); j++)
				{
					if (fullRect.contains(cornersBot[j]) && houghMat.at<uchar>(cornersBot[j]))
					{
						vector<Point2f> line = { cornersTop[i], cornersBot[j] };
						lines.push_back(line);
					}
				}
			}
		}
	}

	return lines;
}

// Group rectangles that represent door candidates out of vertical lines
vector<vector<Point2f>> vertLinesToRectangles(vector<vector<float>>& rectInnerAngles, vector<vector<Point2f>> lines)
{
	vector<vector<Point2f>> rects;

	for (int i = 0; i < lines.size(); i++)
	{
		float length1 = getDistance(lines[i][0], lines[i][1]);

		for (int j = 0; j < lines.size(); j++)
		{
			if (j <= i) continue;

			// Only build rectangle if the two lines are completely distinct
			if ((lines[i][0] == lines[j][0]) || (lines[i][0] == lines[j][1]) || (lines[i][1] == lines[j][0]) || (lines[i][1] == lines[j][1]))
			{
				continue;
			}

			// Check if length difference of lines is close
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


			// These angles could be reused for voting and should be saved
			vector<float> angles;
			angles.push_back(getCornerAngle(lines[i][1], lines[i][0], lines[j][0]));
			angles.push_back(getCornerAngle(lines[i][0], lines[j][0], lines[j][1]));
			angles.push_back(getCornerAngle(lines[j][0], lines[j][1], lines[i][1]));
			angles.push_back(getCornerAngle(lines[j][1], lines[i][1], lines[i][0]));

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
			rectInnerAngles.push_back(angles);
		}
	}

	return rects;
}

// Compare a possible rectangle with the existing edges in the edge image
float compareRectangleToEdges(vector<Point2f> rect, Mat edges)
{
	float result = 0.0;

	for (int i = 0; i < rect.size() - 1; i++)
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

		if (fillRatio < LINE_THRESH)
		{
			return 0.0;
		}

		result += fillRatio;
	}

	// Get average fillRatio for all lines but bottom line
	result = result / 3;

	return result;
}

// Select the candidate by comparing their scores, score boni if special requirements are met
vector<Point2f> selectBestCandidate(vector<vector<Point2f>> candidates, vector<float> scores, Point2f inputPoint, vector<vector<float>> rectInnerAngles, Size size)
{
	//cout << candidates.size() << "size" << endl;
	float goalInputRange = size.width * GOAL_INPUT_RANGE;

	for (int i = 0; i < candidates.size(); i++)
	{
		// INPUT SCORE
		Point2f bottomCenter = (candidates[i][3] + candidates[i][0]) * 0.5;
		float inputDistance = getDistance(bottomCenter, inputPoint);
		float inputScore = (goalInputRange - inputDistance) / goalInputRange;

		// ASPECT SCORE
		float lineLeft = getDistance(candidates[i][0], candidates[i][1]);
		float lineTop = getDistance(candidates[i][1], candidates[i][2]);
		float lineRight = getDistance(candidates[i][2], candidates[i][3]);
		float lineBot = getDistance(candidates[i][3], candidates[i][0]);

		float aspectRatio = ((lineTop + lineBot) * 0.5) / ((lineLeft + lineRight) * 0.5);

		float aspectScore = (GOAL_RATIO_RANGE - abs(GOAL_RATIO - aspectRatio)) / GOAL_RATIO_RANGE;

		// ANGLE SCORE
		// NOTE: maybe punish single angle breakouts more
		float angle0 = rectInnerAngles[i][0];
		float angle1 = rectInnerAngles[i][1];
		float angle2 = rectInnerAngles[i][2];
		float angle3 = rectInnerAngles[i][3];

		float angleDiff = abs(GOAL_ANGLES - angle0) + abs(GOAL_ANGLES - angle1) + abs(GOAL_ANGLES - angle2) + abs(GOAL_ANGLES - angle3);
		float angleScore = (GOAL_ANGLES_DIFF_RANGE - angleDiff) / GOAL_ANGLES_DIFF_RANGE;


		scores[i] *= (1 + ((inputScore * 0.45 + aspectScore * 0.35 + angleScore * 0.2) * 0.5));
		/*cout << "ENDSCORE: " << scores[i] << endl;
		cout << "______________________" << endl;*/
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

// Calculates the median value of a single channel
// based on https://github.com/arnaudgelas/OpenCVExamples/blob/master/cvMat/Statistics/Median/Median.cpp
double getMedian(cv::Mat channel)
{
	double m = (channel.rows * channel.cols) / 2;
	int bin = 0;
	double med = -1.0;

	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange = { range };
	bool uniform = true;
	bool accumulate = false;
	cv::Mat hist;
	cv::calcHist(&channel, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

	for (int i = 0; i < histSize && med < 0.0; ++i)
	{
		bin += cvRound(hist.at< float >(i));
		if (bin > m && med < 0.0)
			med = i;
	}

	return med;
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