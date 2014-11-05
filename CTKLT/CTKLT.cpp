// CTKLT.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "CompressiveKLTracker.h"
#include "MouseCapture.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <windows.h>


using namespace cv;
using namespace std;

extern const Scalar obColors[10] = { Scalar(0, 0, 255), Scalar(255, 0, 255), Scalar(0, 178, 236), Scalar(255, 0, 0), Scalar(255, 165, 0), Scalar(255, 255, 0), Scalar(105, 139, 34), Scalar(255, 20, 147), Scalar(0, 0, 0), Scalar(85, 26, 139) };


const string resultPath = "TrackingResult.txt";

//const string videoPath = "D:\\MyData\\sequences\\ProcessedVideo\\40_100.wmv";
//const string locationPath = "D:\\MyData\\sequences\\ProcessedVideo\\40_100_locations.txt";

const string videoPath = "D:\\MyData\\sequences\\ProcessedVideo\\222_251.wmv";
const string locationPath = "D:\\MyData\\sequences\\ProcessedVideo\\222_251_locations.txt";

//const string videoPath = "D:\\MyData\\sequences\\ProcessedVideo\\kobe.avi";
//const string locationPath = "D:\\MyData\\sequences\\ProcessedVideo\\location.txt";

//const string videoPath = "D:\\MyData\\sequences\\videos\\test_input_opencv_short.wmv"; //
//const string locationPath = "D:\\MyData\\sequences\\videos\\locations.txt";


bool readLocation(string locationPath, vector<vector<Rect> > &vR);
void drawLocation(const int fIdx, Mat& img, const vector<vector<Rect> > &vR);
int frameIndex = -1;
int stopIndex = 8000;




int _tmain(int argc, _TCHAR* argv[])
{
	ofstream outFile;
	outFile.open(resultPath.c_str(), ios::out);
	if (!outFile)
	{
		cout << "error: could not open results file: " << resultPath << endl;
		return EXIT_FAILURE;
	}


	bool useVideo = true;
	VideoCapture cap;

	if (useVideo)
	{
		cout << "Use The Camere" << endl;

		vector<vector<Rect>> vLocations;

		//cap.open(0);// open the camera
		cap.open(videoPath); //use the video
		if (!cap.isOpened())
		{
			cout << "error:could not open the camere" << endl;
			return EXIT_FAILURE;
		}

		if (!readLocation(locationPath, vLocations))
		{
			return EXIT_FAILURE;
		}


		Mat frame, grayImg, result;

		//CompressiveTracker ct;
		vector<CompressiveKLTracker> cts;


		//while (frameIndex<3160)
		//{
		//	cap >> frame;
		//	cout << frameIndex << endl;
		//	frameIndex++;
		//}


		cap >> frame;
		frameIndex++;
		if (frame.empty())
		{
			cout << "error:failed to read the camera buffer " << endl;
			return EXIT_FAILURE;
		}


		VideoWriter video_writer("Output.avi", CV_FOURCC('M', 'J', 'P', 'G'), 25.0, Size(frame.cols, frame.rows));// write video
		char strFrame[256];

		vector<Rect> vbox;
		bool havedrawn = false;
		bool stop = false;

		while (!frame.empty() && frameIndex<stopIndex)
		{

			if (stop)
			{
				cts.clear();
				vbox.clear();

				if (frame.empty())
				{
					cout << "error:failed to read the camera buffer " << endl;
					break;
				}
				cvtColor(frame, grayImg, CV_RGB2GRAY);

				MouseCapture MC("CT", frame);
				havedrawn = MC.drawRect();

				for (int i = 0; i < MC.vrect.size(); i++)
				{

					cts.push_back(CompressiveKLTracker());
				}

				cout << "we have created " << cts.size() << " treckers" << endl;

				for (int i = 0; i < MC.vrect.size(); i++)
				{
					cts.at(i).init(grayImg, MC.vrect.at(i));
					vbox.push_back(Rect(MC.vrect.at(i)));
				}

				stop = false;
			}

			cap >> frame;
			frameIndex++;
			if (frame.empty())
			{
				cout << "error:failed to read the camera buffer " << endl;
				break;
			}


			if (havedrawn)
			{
				if (frame.empty())
				{
					cout << "error:failed to read the camera buffer " << endl;
					break;
				}

				cvtColor(frame, grayImg, CV_RGB2GRAY);

				for (int i = 0; i < cts.size(); i++)
				{
					double t = (double)cvGetTickCount();

					cts.at(i).processFrame(grayImg);

					t = (double)cvGetTickCount() - t;
					
					//printf("%.5f second, %.5f fps\n", t / (cvGetTickFrequency()*1000000.), 1 / (t / (cvGetTickFrequency()*1000000.)));

					rectangle(frame, cts.at(i).box0, obColors[2], 2);// Draw rectangle
					rectangle(frame, cts.at(i).box2, obColors[3], 2);// Draw rectangle
					//rectangle(frame, cts.at(i).box, obColors[4], 2);// Draw rectangle
					outFile << i << ": " << (int)cts.at(i).box2.x << " " << (int)cts.at(i).box2.y << " " << (int)cts.at(i).box2.width << " " << (int)cts.at(i).box2.height << endl;
				}


			}


			sprintf(strFrame, "#%d", frameIndex);
			cv::putText(frame, strFrame, cv::Point(0, 20), 2, 1, CV_RGB(25, 200, 25));

			drawLocation(frameIndex, frame, vLocations); //draw the detection results
			imshow("CT", frame);// Display
			video_writer << frame;


			if (waitKey(1 * int(!havedrawn)) == 's' || frameIndex == 18)//3177
			//if (waitKey(1) == 's' || frameIndex == 18)
			{
				stop = true;
			}

		}

	}



	return 0;
}






bool readLocation(string locationPath, vector<vector<Rect> > &vR)
{
	ifstream locationInFile(locationPath.c_str(), ios::in);

	if (!locationInFile)
	{
		cout << "error: could not open location file: " << locationPath << endl;
		return false;
	}


	string line, name, rect;
	vector<Rect> temp;
	while (getline(locationInFile, line))
	{
		istringstream iss(line);
		rect = "";
		iss >> name >> rect;
		//cout << name << " "<<rect << endl;

		if (rect == "")
		{
			vR.push_back(temp);
			temp.clear();
		}
		else
		{
			float xmin = -1.f;
			float ymin = -1.f;
			float xmax = -1.f;
			float ymax = -1.f;

			sscanf(rect.c_str(), "%f,%f,%f,%f", &xmin, &ymin, &xmax, &ymax);
			temp.push_back(Rect(xmin, ymin, xmax - xmin, ymax - ymin));
		}


	}


	vR.erase(vR.begin());//the first of element is invalid


}



void drawLocation(const int fIdx, Mat& img, const vector<vector<Rect> > &vR)
{
	for (int i = 0; i < vR.at(fIdx).size(); i++)
	{
		rectangle(img, vR.at(fIdx).at(i), Scalar(0, 255, 0), 2);
	}

	char strFrame[20];
	sprintf(strFrame, "#%d ", fIdx);
	putText(img, strFrame, cvPoint(0, 20), 2, 1, CV_RGB(25, 200, 25));
}


