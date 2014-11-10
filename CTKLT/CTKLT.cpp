// CTKLT.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "CompressiveKLTracker.h"
#include "FaceTracker.h"
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
const int nMaxTrackerNum = 20;
const int nReInitFreq = 25;

const string lDescription = "Detection";
const string rDescription = "Detection + Tracking";


const string resultPath = "TrackingResult.txt";

//const string videoPath = "D:\\MyData\\sequences\\ProcessedVideo\\40_100.wmv";
//const string locationPath = "D:\\MyData\\sequences\\ProcessedVideo\\40_100_locations.txt";

//const string videoPath = "D:\\MyData\\sequences\\ProcessedVideo\\222_251.wmv";                  
//const string locationPath = "D:\\MyData\\sequences\\ProcessedVideo\\222_251_locations.txt";

//const string videoPath = "D:\\MyData\\sequences\\ProcessedVideo\\kobe.avi";
//const string locationPath = "D:\\MyData\\sequences\\ProcessedVideo\\location.txt";

const string videoPath = "D:\\MyData\\sequences\\videos\\test_input_opencv_short.wmv"; //
const string locationPath = "D:\\MyData\\sequences\\videos\\locations.txt";


bool readLocation(string locationPath, vector<vector<Rect> > &vR);
void drawLocation(const int fIdx, Mat& img, const vector<vector<Rect> > &vR);
void checkLocation(vector<vector<Rect>>& vR, int nW, int nH);

int frameIdx = -1;
int stopIndex = 1030;




#define MULTIPLE 1




#ifndef MULTIPLE
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
		frameIdx++;
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

		while (!frame.empty() && frameIdx<stopIndex)
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
			frameIdx++;
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


			sprintf(strFrame, "#%d", frameIdx);
			cv::putText(frame, strFrame, cv::Point(0, 20), 2, 1, CV_RGB(25, 200, 25));

			drawLocation(frameIdx, frame, vLocations); //draw the detection results
			imshow("CT", frame);// Display
			video_writer << frame;


			if (waitKey(1 * int(!havedrawn)) == 's' || frameIdx == 18)//3177
			//if (waitKey(1) == 's' || frameIndex == 18)
			{
				stop = true;
			}

		}

	}



	return 0;
}


#else

int _tmain(int argc, _TCHAR* argv[])
{
	// read the video
	VideoCapture cap;
	cap.open(videoPath);
	if (!cap.isOpened())
	{
		cout << "Error:could not open the camere" << endl;
		return -1;
	}


	// read the detection results
	vector<vector<Rect>> locationRect;
	if (!readLocation(locationPath, locationRect))
	{
		cout << "Error:could not open the detection results\n" << endl;
		return -1;
	}




	Mat framePrevious, grayPrevious;
	Mat frameCurrent, grayCurrent;
	vector<CompressiveKLTracker> ctlts;
	ctlts.reserve(nMaxTrackerNum); // reserve enough capacity


	cap >> frameCurrent;
	frameIdx++;
	if (frameCurrent.empty())
	{
		cout << "error: could not read the frame " << (frameIdx) << endl;
		return -1;
	}
	cvtColor(frameCurrent, grayCurrent, CV_RGB2GRAY);

	int nW = frameCurrent.cols;
	int nH = frameCurrent.rows;


	checkLocation(locationRect, nW, nH);

	// write and show the side-by-side video
	VideoWriter video_writer("Output.avi", CV_FOURCC('M', 'J', 'P', 'G'), 25.0, Size(nW * 2, nH));// write video
	Mat imgResult(nH, nW * 2, CV_8UC3, Scalar(0));
	Rect r1(0, 0, nW, nH), r2(0 + nW, 0, nW, nH);
	Mat roi1 = imgResult(r1);
	Mat roi2 = imgResult(r2);



	bool needInit = true;
	bool needUpdate = false;
	bool needTracked = false;
	
	static int requestCount = 0;

	while (!frameCurrent.empty() && frameIdx < stopIndex)
	{
		while (needInit&& frameIdx < stopIndex)
		{
			requestCount++;
			cout << "-----------------------------initialing frame " << frameIdx << "------------------------------" << endl;

			frameCurrent.copyTo(roi1);
			frameCurrent.copyTo(roi2);

			if (locationRect.at(frameIdx).size()>0)
			{
				ctlts.clear();
				for (int i = 0; i < locationRect.at(frameIdx).size(); i++)
				{
					ctlts.push_back(CompressiveKLTracker(i, locationRect.at(frameIdx).at(i)));
				}

				initCTKLTS(grayCurrent, ctlts);

				rectangleDET(roi1, locationRect.at(frameIdx));
				rectangleCTKLTS(roi2, ctlts);

				needInit = false;
				needTracked = true;
			}


			addDescription(roi1, frameIdx, lDescription);
			addDescription(roi2, frameIdx, rDescription);

			imshow("Face Detection and Tracking", imgResult);
			video_writer << imgResult;

			if (needInit)
			{
				waitKey(1);
			}
			else
			{
				waitKey(0);
			}

			if (needInit)
			{
				cap >> frameCurrent;
				frameIdx++;
				if (frameCurrent.empty())
				{
					cout << "error: could not read the frame " << (frameIdx) << endl;
					break;
				}
				cvtColor(frameCurrent, grayCurrent, CV_RGB2GRAY);
			}

		}// while needInit


		while (needUpdate&& frameIdx < stopIndex)
		{
			requestCount++;
			cout << "-----------------------------updating frame " << frameIdx << "------------------------------" << endl;

			cout << ctlts.size() << endl;

			vector<Rect> & lR = locationRect.at(frameIdx);
			bool *matched = new bool[lR.size()];
			memset(matched, 0, sizeof(bool)*lR.size());


			for (int j = 0; j < lR.size(); j++)
			{
				float maxOverlapRatio = 0.0;
				int idx = -1;
				for (int i = 0; i < ctlts.size(); i++)
				{
					Rect intersection = ctlts.at(i).box&lR.at(j);
					if ((float(intersection.area()) / float(lR.at(j).area())) >= maxOverlapRatio)
					{
						maxOverlapRatio = float(intersection.area()) / float(lR.at(j).area());
						idx = i;
					}
				}


				if (maxOverlapRatio > 0.0&&idx != -1)
				{
					ctlts.at(idx).updateTracker(grayCurrent, lR.at(j));
					matched[j] = true;
				}
			}




			for (int i = 0; i < lR.size(); i++)
			{
				if (!matched[i])
				{
					//cout << "----------------------------------" << endl;
					//for (int j = 0; j < ctlts.size(); j++)
					//{
					//	float ratio=ctlts.at(j).classifyRect(lR.at(i));
					//	cout << "ratio: " << ratio << endl;
					//}

					ctlts.push_back(CompressiveKLTracker(i, lR.at(i)));
					ctlts.back().init(grayCurrent, lR.at(i));
				}

			}




			rectangleDET(roi1, locationRect.at(frameIdx));
			rectangleCTKLTS(roi2, ctlts);

			addDescription(roi1, frameIdx, lDescription);
			addDescription(roi2, frameIdx, rDescription);

			imshow("Face Detection and Tracking", imgResult);
			video_writer << imgResult;
			waitKey(0);

			needUpdate = false;
			needTracked = true;

			delete[] matched;


			cout << ctlts.size() << endl;
		}



		while (needTracked&& frameIdx < stopIndex)
		{
			cout << "-----------------------------tracking frame "<<frameIdx<<"-----------------------------" << endl;
			static int trackedCount = 0;

			cap >> frameCurrent;

			frameIdx++;
			if (frameCurrent.empty())
			{
				cout << "error: could not read frame " << (frameIdx) << endl;
				break;
			}
			cvtColor(frameCurrent, grayCurrent, CV_RGB2GRAY);

			frameCurrent.copyTo(roi1);
			frameCurrent.copyTo(roi2);

			double t = (double)cvGetTickCount();

			int failCount = trackAndCheck(grayCurrent, ctlts);

			t = (double)cvGetTickCount() - t;
			printf("%.5f second, %.5f fps\n", t / (cvGetTickFrequency()*1000000.), 1 / (t / (cvGetTickFrequency()*1000000.)));


			if (ctlts.empty())//当没有目标时，初始化
			{
				trackedCount = 0;
				needInit = true;
				needTracked = false;
				break;
			}

			if (failCount > 0) // 只要有一个跟踪失败，就尝试请求更新
			{
				if (locationRect.at(frameIdx).size() > 0)//请求检测结果，当有检测结果的时候才会去更新
				{
					trackedCount = 0;
					needUpdate = true;
					needTracked = false;
					break;
				}
				else //没有检测结果的时候，继续跟踪，跟踪次数+1，直到没有可跟踪的目标后初始化
				{
					trackedCount++;
				}

			}
			else
			{
				trackedCount++;
			}



			if (trackedCount > nReInitFreq)
			{
				if (locationRect.at(frameIdx).size() > 0)//请求检测结果，当有结果的时候才会去更新，否则继续跟踪
				{
					trackedCount = 0;
					needUpdate = true;
					needTracked = false;
					break;
				}
			}

			rectangleDET(roi1, locationRect.at(frameIdx));
			rectangleCTKLTS(roi2, ctlts);

			addDescription(roi1, frameIdx, lDescription);
			addDescription(roi2, frameIdx, rDescription);

			imshow("Face Detection and Tracking", imgResult);
			video_writer << imgResult;
			waitKey(0);

		}

        
		cout << requestCount << endl;

	}


	system("pause");
	return 0;
}


#endif








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



void checkLocation(vector<vector<Rect>>& vR, int nW, int nH)
{
	int badCount = 0;
	for (int i = 0; i < vR.size(); i++)
	{
		vector<Rect>::iterator ite;
		for (ite = vR.at(i).begin(); ite != vR.at(i).end();)
		{
			if ((*ite).x<0 || (*ite).y<0 || ((*ite).x + (*ite).width)>nW || ((*ite).y + (*ite).height)>nH)
			{
				ite = vR.at(i).erase(ite);
				badCount++;
				cout << i << endl;
			}
			else
			{
				ite++;
			}
		}
	}
	cout << "the number of bad location :" << badCount << endl;
}