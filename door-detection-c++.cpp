#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <stdlib.h>
#include <iostream>
#include <chrono>

#define _USE_MATH_DEFINES
#include <math.h>

using namespace cv;
using namespace std;

// Declare all used constants

// State of optical flow keypoints
// unstable: not enough points for detection
// WATCHING: not enough frames to be stable
// stable: enough points and enough frames lived by
enum State { UNSTABLE, WATCHING, STABLE };
const int MIN_POINTS_COUNT = 40;
const int MIN_FRAME_COUNT = 60;
const float MIN_DEPTH_DISTANCE = 50.0;
const int DETECTION_FAILED_RESET_COUNT = 7;

// Image conversion and processing constants
const float CONTRAST = 1.2;
const Size BLUR_KERNEL = Size(3, 3);
const float BLUR_SIGMA = 2.5;
const double CANNY_LOWER = 0.33; // NOTE: The lower threshold is lower than most canny auto thresholds, but necessary to catch some door edges
const double CANNY_UPPER = 1.33;

// NOTE: Corner qualit could be tuned down, or amount up to find possibly needed corners
// Corner detection constants
const int CORNERS_MAX = 200;
const float CORNERS_QUALITY = 0.01;
const float CORNERS_MIN_DIST = 20.0;

// Hough line constants
const int HOUGH_LINE_WIDTH = 15;
const int HOUGH_LINE_ADDITIONAL_WIDTH = 5;
const int HOUGH_LINE_WIDTH_MAX = 30;
const float HOUGH_LINE_DIFF_THRESH_PIXEL = 10;
const float HOUGH_LINE_DIFF_THRESH_ANGLE = 0.05;
const int HOUGH_COUNT_LIMIT = 20;

// Vertical lines constants
const float LINE_MAX = 0.9;
const float LINE_MIN = 0.4;
const float POINT_DEPTH_CLOSENESS = 0.4;

// Rectangles constants
const float ANGLE_MAX = 0.175; // RAD
const float LENGTH_DIFF_MAX = 0.12;
const float ASPECT_RATIO_MIN = 0.3;
const float ASPECT_RATIO_MAX = 0.8; // from 0.6
const float LENGTH_HOR_DIFF_MAX = 1.2;
const float LENGTH_HOR_DIFF_MIN = 0.7;
const float RECTANGLE_THRESH = 10.0;
const float RECTANGLE_OPPOSITE_THRESH = 10.0;
const float LINE_DEPTH_CLOSENESS = 0.4;

// Comparison of rectangles to edges constants
const float RECT_THRESH = 0.8; // from 0.85
const float LINE_THRESH = 0.65;
const int LINE_WIDTH = 8;

// Selection formula constants
const float GOAL_RATIO = 0.45;
const float GOAL_RATIO_RANGE = 0.15;
const float GOAL_ANGLES = 90;
const float GOAL_ANGLES_DIFF_RANGE = 20;

// Declare all used functions
bool detect(Mat grayImage, vector<Point2f>points, vector<float>pointDepths, vector<Point2f>& result);
vector<vector<Point2f>> cornersToVertLines(vector<float>& lineDepths, vector<float>& lineLengths, vector<Point2f> corners, vector<float> pointDepths, vector<Vec2f> houghLines, vector<int> houghLinesWidth, float depthRange, Size size);
vector<vector<Point2f>> vertLinesToRectangles(vector<float>& rectDepthDiffs, vector<vector<Point2f>> lines, vector<float> lineDepths, vector<float> lineLengths, float depthRange);
float compareRectangleToEdges(vector<Point2f> rect, Mat edges);
vector<Point2f> selectBestCandidate(vector<vector<Point2f>> candidates, vector<float> rectDepthDiffs, float depthRange, vector<float> scores);

float getDistance(Point2f p1, Point2f p2);
float getOrientation(Point2f p1, Point2f p2);
float getCornerAngle(Point2f p1, Point2f p2, Point2f p3);
double getMedian(Mat channel);

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

	if (!video)
	{
		cout << "Optical Flow needs a video as input" << endl;
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

		// Used for resetting feature points if none top is detected
		Rect topRect = Rect(0, 0, frameWidth, frameHeight * 0.4);

		// Start Optical Flow setup
		vector<Point2f> p0, p1;
		Mat frame, frameGray;
		Mat prevFrameGray;
		vector<float> longTimeDistances;

		// Results in 360p by 0.5 factor (if it is HD 720p)
		// Results in 360p by 0.75 factor (if it is 480p)
		//Size smallSize = Size(cap.get(CAP_PROP_FRAME_WIDTH) * 0.5, cap.get(CAP_PROP_FRAME_HEIGHT) * 0.5);
		Size smallSize = Size(cap.get(CAP_PROP_FRAME_WIDTH) * 0.75, cap.get(CAP_PROP_FRAME_HEIGHT) * 0.75);

		// For video capturing only
		int codec = VideoWriter::fourcc('M', 'J', 'P', 'G');
		VideoWriter video("./results/output.avi", codec, 25.0, Size(smallSize.height, smallSize.width));
		
		// State properties that are modified over time
		State state = State::UNSTABLE;
		int frameCount = 0;
		int failedAttemptCount = 0;

		while (1)
		{
			auto t1 = chrono::steady_clock::now();

			cap >> frame;

			if (frame.empty()) break;

			resize(frame, frame, smallSize);
			rotate(frame, frame, ROTATE_90_CLOCKWISE);
			cvtColor(frame, frameGray, COLOR_BGR2GRAY);

			if (state == State::UNSTABLE)
			{
				goodFeaturesToTrack(frameGray, p0, CORNERS_MAX, CORNERS_QUALITY, CORNERS_MIN_DIST, Mat(), 7, false, 0.04);
				frameCount = 0;
				longTimeDistances = {};

				int topPoints = 0;

				if (p0.size() > MIN_POINTS_COUNT)
				{
					for (int i = 0; i < p0.size(); i++)
					{
						if (topRect.contains(p0[i]))
						{
							topPoints++;

							if (topPoints > 1)
							{
								prevFrameGray = frameGray.clone();
								state = State::WATCHING;
								break;
							}
						};
					}
				}
			}
			else
			{
				// State is now STABLE or WATCHING
				frameCount++;
				cout << "Frames in row: " << frameCount << endl;

				// Calculate optical flow
				vector<uchar> status;
				vector<float> err;
				TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
				calcOpticalFlowPyrLK(prevFrameGray, frameGray, p0, p1, status, err, Size(15, 15), 2, criteria);
				vector<Point2f> goodMatches;

				vector<bool> pointControl;
				bool pointLost = false;

				vector<float> distances;
				for (uint i = 0; i < p0.size(); i++)
				{
					// Select good points
					if (status[i] == 1) {
						goodMatches.push_back(p1[i]);

						float dist = getDistance(p0[i], p1[i]);

						distances.push_back(dist);
						pointControl.push_back(true);
					}
					else
					{
						pointLost = true;
						pointControl.push_back(false);
					}
				}

				// If too many points have been lost, return to UNSTABLE state
				if (goodMatches.size() < MIN_POINTS_COUNT)
				{
					state = State::UNSTABLE;
					continue;
				}

				float avgDistance = 0.0;
				// Initialization if longTimeDistances starts or was cleared
				if (longTimeDistances.size() == 0)
				{
					longTimeDistances = distances;
				}
				else
				{
					// Since points were lost since the last frame, distances need to be updated
					if (pointLost)
					{
						vector<float> updDistances;
						for (uint i = 0; i < longTimeDistances.size(); i++)
						{
							// If point i was a good point
							if (pointControl[i])
							{
								updDistances.push_back(longTimeDistances[i]);
							}
						}

						longTimeDistances = updDistances;
					}

					// Add calculated distances of this frame to the longTimeDistances
					for (uint i = 0; i < distances.size(); i++)
					{
						longTimeDistances[i] += distances[i];
						avgDistance += longTimeDistances[i];
					}
				}
				avgDistance = avgDistance / longTimeDistances.size();
				cout << avgDistance << endl;

				prevFrameGray = frameGray.clone();
				p0 = goodMatches;

				// Check if door detection is now possible
				if (frameCount > MIN_FRAME_COUNT || avgDistance > MIN_DEPTH_DISTANCE)
				{
					state = State::STABLE;
				}

				// Finally a door can be detected
				if (state == State::STABLE)
				{
					vector<Point2f> result = {};

					bool success = detect(frameGray, goodMatches, longTimeDistances, result);

					if (success)
					{
						failedAttemptCount = 0;

						line(frame, result[0], result[1], Scalar(255, 255, 0), 2);
						line(frame, result[1], result[2], Scalar(255, 255, 0), 2);
						line(frame, result[2], result[3], Scalar(255, 255, 0), 2);
						line(frame, result[3], result[0], Scalar(255, 255, 0), 2);
					}
					else
					{
						failedAttemptCount++;
						
						// Reset to start if this detection isnt going anywhere
						if (failedAttemptCount > DETECTION_FAILED_RESET_COUNT)
						{
							failedAttemptCount = 0;
							state = State::UNSTABLE;
						}
					}
				}
			}
			imshow("Display window", frame);
			video.write(frame);

			if ((char)waitKey(1) == 27) break;
		}

		cap.release();
		video.release();
	}
	
	cv::destroyAllWindows();

	return 0;
}

bool detect(Mat inputGray, vector<Point2f>points, vector<float>pointDepths, vector<Point2f>& result)
{
	auto t1 = chrono::steady_clock::now();

	Mat imgGray;
	inputGray.copyTo(imgGray);

	// Increase contrast
	imgGray.convertTo(imgGray, -1, CONTRAST, 0);

	// Blur the image
	Mat blurred;
	GaussianBlur(imgGray, blurred, BLUR_KERNEL, BLUR_SIGMA);

	// Generate edges
	Mat edges;
	double median = getMedian(blurred);

	// Very dark images can go to values like 9, resulting in extremely noisy images
	median = max((double)30, median);

	double lowerThresh = max((double)0, (CANNY_LOWER * median));
	double higherThresh = min((double)255, (CANNY_UPPER * median));

	Canny(blurred, edges, lowerThresh, higherThresh);
	imshow("Edges window", edges);

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
				continue;
			}
		}

		filteredHoughLines.push_back(houghLines[h]);
		filteredHoughLinesWidth.push_back(HOUGH_LINE_WIDTH);
	}

	float min = *min_element(pointDepths.begin(), pointDepths.end());
	float max = *max_element(pointDepths.begin(), pointDepths.end());
	float depthRange = max - min;

	// this is all just for testing
	/*for (uint i = 0; i < points.size(); i++)
	{
		float depthToColor = 255 * ((pointDepths[i] - min) / depthRange);
		Scalar col = Scalar(depthToColor, depthToColor, depthToColor);

		circle(blurred, points[i], 5, col, -1);
	}
	imshow("Dev window", blurred);*/

	// Connect corners to vertical lines
	vector<float> lineDepths = {};
	vector<float> lineLengths = {};
	vector<vector<Point2f>> lines = cornersToVertLines(lineDepths, lineLengths, points, pointDepths, filteredHoughLines, filteredHoughLinesWidth, depthRange, imgGray.size());

	/*for (int i = 0; i < lines.size(); i++)
	{
		line(blurred, lines[i][0], lines[i][1], Scalar(0, 255, 0), 2);
	}
	imshow("Dev window", blurred);*/

	// Group corners based on found lines to rectangles
	vector<float> rectDepthDiffs = {};
	vector<vector<Point2f>> rectangles = vertLinesToRectangles(rectDepthDiffs, lines, lineDepths, lineLengths, depthRange);

	// NOTE: this could be done in vertLinesToRectangles aswell
	// Compare the found rectangles to the edge image
	vector<vector<Point2f>> candidates;
	vector<float> updDepthDiffs;
	vector<float> scores;
	for (int i = 0; i < rectangles.size(); i++)
	{
		float result = compareRectangleToEdges(rectangles[i], edges);

		if (result > RECT_THRESH)
		{
			candidates.push_back(rectangles[i]);
			updDepthDiffs.push_back(rectDepthDiffs[i]);
			scores.push_back(result);
		}
	}
	rectDepthDiffs = updDepthDiffs;

	//for (int i = 0; i < candidates.size(); i++)
	//{
	//	line(blurred, candidates[i][0], candidates[i][1], Scalar(255, 255, 0), 2);
	//	line(blurred, candidates[i][1], candidates[i][2], Scalar(255, 255, 0), 2);
	//	line(blurred, candidates[i][2], candidates[i][3], Scalar(255, 255, 0), 2);
	//	line(blurred, candidates[i][3], candidates[i][0], Scalar(255, 255, 0), 2);
	//}
	//imshow("Dev window", blurred);

	// Select the best candidate out of the given rectangles
	if (candidates.size())
	{
		vector<Point2f> door = selectBestCandidate(candidates, rectDepthDiffs, depthRange, scores);
		result = door;

		auto t2 = chrono::steady_clock::now();
		auto duration = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
		cout << "Detection DURATION: " << duration << endl;

		return true;
	}

	auto t2 = chrono::steady_clock::now();
	auto duration = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
	cout << "Detection DURATION: " << duration << endl;

	return false;
}

// Group corners to vertical lines that represent the door posts
vector<vector<Point2f>> cornersToVertLines(vector<float>& lineDepths, vector<float>& lineLengths, vector<Point2f> corners, vector<float> depths, vector<Vec2f> houghLines, vector<int> houghLinesWidth, float depthRange, Size size)
{
	auto t1 = chrono::steady_clock::now();

	float lengthMax = LINE_MAX * size.height;
	float lengthMin = LINE_MIN * size.height;

	vector<vector<Point2f>> lines;

	Mat houghMat;
	Rect fullRect = Rect(cv::Point(), size);
	int linesComputed = 0;

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
		linesComputed++;

		// houghLines are ordered by votes, therefor weaker lines can be omitted
		if (linesComputed > HOUGH_COUNT_LIMIT)
		{
			continue;
		}

		line(houghMat, pt1, pt2, 1, houghLinesWidth[h], LINE_AA);

		vector<Point2f> houghPoints = {};
		vector<float> houghDepths = {};
		for (size_t j = 0; j < corners.size(); j++)
		{
			if (fullRect.contains(corners[j]) && houghMat.at<uchar>(corners[j]))
			{
				houghPoints.push_back(corners[j]);
				houghDepths.push_back(depths[j]);
			}
		}

		for (size_t i = 0; i < houghPoints.size(); i++)
		{
			float iDepth = houghDepths[i];

			for (int j = 0; j < houghPoints.size(); j++)
			{
				if (j <= i) continue;

				float jDepth = houghDepths[j];
				if (abs(iDepth - jDepth) > (depthRange * POINT_DEPTH_CLOSENESS))
				{
					continue;
				}

				float distance = getDistance(houghPoints[i], houghPoints[j]);
				if (distance < lengthMin || distance > lengthMax)
				{
					continue;
				}

				// Sort by y-value, so that the high points are first
				vector<Point2f> line;
				if (houghPoints[i].y < houghPoints[j].y)
				{
					line = { houghPoints[i], houghPoints[j] };
				}
				else {
					line = { houghPoints[j], houghPoints[i] };
				}

				lines.push_back(line);
				lineDepths.push_back((iDepth + jDepth) / 2);
				lineLengths.push_back(distance);
			}
		}
	}

	auto t2 = chrono::steady_clock::now();
	auto duration = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
	//cout << "cornersToVertLines " << duration << "ms" << endl;
	//cout << "Number of Lines: " << lines.size() << endl;

	return lines;
}

// Group rectangles that represent door candidates out of vertical lines
vector<vector<Point2f>> vertLinesToRectangles(vector<float>& rectDepthDiffs, vector<vector<Point2f>> lines, vector<float> lineDepths, vector<float> lineLengths, float depthRange)
{
	auto t1 = chrono::steady_clock::now();

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

			float depthDiff = abs(lineDepths[i] - lineDepths[j]);
			if (depthDiff > (depthRange * LINE_DEPTH_CLOSENESS))
			{
				continue;
			}

			// Check if length difference of lines is close
			float lengthDiff = abs(lineLengths[i] - lineLengths[j]);
			float lengthAvg = (lineLengths[i] + lineLengths[j]) / 2;

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
			rectDepthDiffs.push_back(depthDiff);
		}
	}

	auto t2 = chrono::steady_clock::now();
	auto duration = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
	//cout << "vertLinesToRectangles " << duration << "ms" << endl;

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
vector<Point2f> selectBestCandidate(vector<vector<Point2f>> candidates, vector<float> rectDepthDiffs, float depthRange, vector<float> scores)
{
	//cout << candidates.size() << "size" << endl;

	// These values have meaning for all candidates and need to be calculated once
	// NOTE: this is the first version and is cinda crappy
	float candidatesDepthRange = *max_element(rectDepthDiffs.begin(), rectDepthDiffs.end());
	float depthUpvoteRange = max((double)candidatesDepthRange, min(7.5, depthRange * 0.1));


	for (int i = 0; i < candidates.size(); i++)
	{
		// NOTE: this can be a trap and should be somewhat dynamic
		// Test if inner content has a different color average
		//int left = (candidates[i][0].x + candidates[i][1].x) / 2;
		//int top = (candidates[i][1].y + candidates[i][2].y) / 2;
		//int right = (candidates[i][2].x + candidates[i][3].x) / 2;
		//int bottom = (candidates[i][3].y + candidates[i][0].y) / 2;

		//// This whole process of masking the image seems like a workaround
		//Rect rect = Rect(Point2i(left, bottom), Point2i(right, top));
		//Mat mask = Mat::zeros(gray.size(), CV_8U);
		//rectangle(mask, rect, 1);

		//double inner = mean(gray, mask)[0];
		//mask = 1 - mask;
		//double outer = mean(gray, mask)[0];

		//if (abs(inner - outer) > COLOR_DIFF_THRESH)
		//{
		//	scores[i] *= UPVOTE_FACTOR;
		//}

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
		//for (int j = 0; j < candidates.size(); j++)
		//{
		//	if (j == i) continue;
		//	// Gets upvoted x times for x doors with same topPoints -> defeating all other candidates that do not have multiple doors on it

		//	if (candidates[i][1] == candidates[j][1] && candidates[i][2] == candidates[j][2])
		//	{
		//		scores[i] = scores[i] * UPVOTE_FACTOR;
		//	}
		//}

		//cout << "BEGINNING SCORE: " << scores[i] << endl;

		// ASPECT SCORE
		float lineLeft = getDistance(candidates[i][0], candidates[i][1]);
		float lineTop = getDistance(candidates[i][1], candidates[i][2]);
		float lineRight = getDistance(candidates[i][2], candidates[i][3]);
		float lineBot = getDistance(candidates[i][3], candidates[i][0]);

		float aspectRatio = ((lineTop + lineBot) * 0.5) / ((lineLeft + lineRight) * 0.5);

		float aspectScore = (GOAL_RATIO_RANGE - abs(GOAL_RATIO - aspectRatio)) / GOAL_RATIO_RANGE;

		// ANGLE SCORE
		// NOTE: maybe punish single angle breakouts more
		float angle0 = getCornerAngle(candidates[i][3], candidates[i][0], candidates[i][1]);
		float angle1 = getCornerAngle(candidates[i][0], candidates[i][1], candidates[i][2]);
		float angle2 = getCornerAngle(candidates[i][1], candidates[i][2], candidates[i][3]);
		float angle3 = getCornerAngle(candidates[i][2], candidates[i][3], candidates[i][0]);

		float angleDiff = abs(GOAL_ANGLES - angle0) + abs(GOAL_ANGLES - angle1) + abs(GOAL_ANGLES - angle2) + abs(GOAL_ANGLES - angle3);

		float angleScore = (GOAL_ANGLES_DIFF_RANGE - angleDiff) / GOAL_ANGLES_DIFF_RANGE;

		// DEPTH SCORE
		// NOTE: depthUpvoteRange is calculated on top
		float depthScore = (depthUpvoteRange - rectDepthDiffs[i]) / depthUpvoteRange;
		

		// BEGIN OF FORMULA
		scores[i] *= (1 + ((aspectScore * 0.4 + angleScore * 0.35 + depthScore * 0.25) * 0.5));
		/*cout << "ENDSCORE: " << scores[i] << endl;
		cout << "______________________" << endl;*/

	}

	int index = max_element(scores.begin(), scores.end()) - scores.begin();
	//cout << " winner " << index;
	vector<Point2f> door = candidates[index];

	//waitKey(0);

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