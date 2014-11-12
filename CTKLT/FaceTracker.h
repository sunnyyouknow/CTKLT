#pragma once

#include "CompressiveKLTracker.h"
#include <vector>
#include <cv.h>
#include <opencv2\opencv.hpp>
#include <string>


void initCTKLTS(cv::Mat &img, std::vector<CompressiveKLTracker> & _ctklts);
void rectangleCTKLTS(cv::Mat &img, const std::vector<CompressiveKLTracker> & _ctklts);
void rectangleDET(cv::Mat &img, const std::vector<cv::Rect> & rec);
void addDescription(cv::Mat &img, const int _frameIdx, const std::string _description);

int trackAndCheck(cv::Mat & grayImg, std::vector<CompressiveKLTracker> & _cts);



class LocationIdentification
{
public:
	LocationIdentification(){ rec = cv::Rect(-1.f, -1.f, -1.f, -1.f); id = -1; confidence = -1.f; }
	LocationIdentification(cv::Rect _rec, int _id, float _confidence) :rec(_rec), id(_id), confidence(_confidence){}
	cv::Rect rec;
	int id;
	float confidence;
};


void rectangleDecIden(cv::Mat &img, const std::vector<LocationIdentification> & lid);