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



