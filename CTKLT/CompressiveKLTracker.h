/************************************************************************
* File:	CompressiveTracker.h
* Brief: CompressiveTrakcing+KLT
* Version: 1.0
* Author: xialan
************************************************************************/

#pragma once
#include "LKTracker.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

using std::vector;
using namespace cv;

const int nminPoints = 8;
const int nMaxPoints = 200;
const float fminFloat = 1;

extern const Scalar obColors[];

class CompressiveKLTracker
{
// constructor and destructor functions 
public:
	CompressiveKLTracker();
	CompressiveKLTracker(int _id);
	CompressiveKLTracker(int _id, Rect _box);
	~CompressiveKLTracker();


// variables for CT
private:
	int featureMinNumRect;
	int featureMaxNumRect;
	int featureNum;
	vector<vector<Rect>> features;
	vector<vector<float>> featuresWeight;
	int rOuterPositive;
	vector<Rect> samplePositiveBox;
	vector<Rect> sampleNegativeBox;
	int rSearchWindow;
	Mat imageIntegral;
	Mat samplePositiveFeatureValue;
	Mat sampleNegativeFeatureValue;
	vector<float> muPositive;
	vector<float> sigmaPositive;
	vector<float> muNegative;
	vector<float> sigmaNegative;
	float learnRate;
	vector<Rect> detectBox;
	Mat detectFeatureValue;
	RNG rng;

// functions for CT
private:
	void HaarFeature(Rect& _objectBox, int _numFeature);
	void sampleRect(Mat& _image, Rect& _objectBox, float _rInner, float _rOuter, int _maxSampleNum, vector<Rect>& _sampleBox);
	void sampleRect(Mat& _image, Rect& _objectBox, float _srw, vector<Rect>& _sampleBox);
	void getFeatureValue(Mat& _imageIntegral, vector<Rect>& _sampleBox, Mat& _sampleFeatureValue);
	void classifierUpdate(Mat& _sampleFeatureValue, vector<float>& _mu, vector<float>& _sigma, float _learnRate);
	void radioClassifier(vector<float>& _muPos, vector<float>& _sigmaPos, vector<float>& _muNeg, vector<float>& _sigmaNeg,
		Mat& _sampleFeatureValue, float& _radioMax, int& _radioMaxIndex);
public:
	void processFrame(Mat& _frame);
	void init(Mat& _frame, Rect _objectBox);


// variables for KLT
public:
	int id;
	int status;//the status of the tracker 0-fail 1-success 2...
	Rect box0;
	Rect box1;
	Rect box2;
	vector<Point2f> vp1;
	vector<Point2f> vp2;
	LKTracker lkt;

// functions for KLT
public:
	void bbPoints(std::vector<cv::Point2f>& points, const cv::Rect& bb);
	float bbPredict(const std::vector<cv::Point2f>& points1, const std::vector<cv::Point2f>& points2,
		const cv::Rect& bb1, cv::Rect& bb2);
	void bbPointsharris(cv::Mat& img, std::vector<cv::Point2f>& points, const cv::Rect& bb);


// variables for CTKLT
private:
	Mat preGrayFrame;
	float scaleRatio;
};