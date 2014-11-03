﻿#include "CompressiveKLTracker.h"
#include <math.h>
#include <iostream>
using namespace cv;
using namespace std;

#define PRINT(x) cout<<(#x)<<": "<<(x)<<endl;;

CompressiveKLTracker::CompressiveKLTracker()
{
	featureMinNumRect = 2;
	featureMaxNumRect = 4;	// number of rectangle from 2 to 4
	featureNum = 50;	// number of all weaker classifiers, i.e,feature pool
	rOuterPositive = 4;	// radical scope of positive samples
	rSearchWindow = 25; // size of search window
	muPositive = vector<float>(featureNum, 0.0f);
	muNegative = vector<float>(featureNum, 0.0f);
	sigmaPositive = vector<float>(featureNum, 1.0f);
	sigmaNegative = vector<float>(featureNum, 1.0f);
	learnRate = 0.85f;	// Learning rate parameter
	
	status = 0;

	scaleRatio = 1.0;
}


CompressiveKLTracker::CompressiveKLTracker(int _id)
{
	id = _id;

	featureMinNumRect = 2;
	featureMaxNumRect = 4;	// number of rectangle from 2 to 4
	featureNum = 50;	// number of all weaker classifiers, i.e,feature pool
	rOuterPositive = 4;	// radical scope of positive samples
	rSearchWindow = 25; // size of search window
	muPositive = vector<float>(featureNum, 0.0f);
	muNegative = vector<float>(featureNum, 0.0f);
	sigmaPositive = vector<float>(featureNum, 1.0f);
	sigmaNegative = vector<float>(featureNum, 1.0f);
	learnRate = 0.85f;	// Learning rate parameter

	status = 0;

	scaleRatio = 1.0;
}


CompressiveKLTracker::CompressiveKLTracker(int _id, Rect _box)
{
	id = _id;
	box1 = _box;
	box2 = _box;

	id = _id;

	featureMinNumRect = 2;
	featureMaxNumRect = 4;	// number of rectangle from 2 to 4
	featureNum = 50;	// number of all weaker classifiers, i.e,feature pool
	rOuterPositive = 4;	// radical scope of positive samples
	rSearchWindow = 25; // size of search window
	muPositive = vector<float>(featureNum, 0.0f);
	muNegative = vector<float>(featureNum, 0.0f);
	sigmaPositive = vector<float>(featureNum, 1.0f);
	sigmaNegative = vector<float>(featureNum, 1.0f);
	learnRate = 0.85f;	// Learning rate parameter

	status = 0;

	scaleRatio = 1.0;
}

CompressiveKLTracker::~CompressiveKLTracker()
{
}

/*
通过积分图来计算采集到的每一个样本的harr特征，这个特征通过与featuresWeight来相乘  
就相当于投影到随机测量矩阵中了，也就是进行稀疏表达了。
每一个样本有默认为50个harr特征，每一个harr特征是由2到3个随机选择的矩形框来构成的，  
对这些矩形框的灰度加权求和作为这一个harr特征的特征值。 
*/
void CompressiveKLTracker::HaarFeature(Rect& _objectBox, int _numFeature)
/*Description: compute Haar features
Arguments:
-_objectBox: [x y width height] object rectangle
-_numFeature: total number of features.The default is 50.
*/
{
	features = vector<vector<Rect>>(_numFeature, vector<Rect>());
	featuresWeight = vector<vector<float>>(_numFeature, vector<float>());

	int numRect;
	Rect rectTemp;
	float weightTemp;

	for (int i = 0; i<_numFeature; i++)
	{
		numRect = cvFloor(rng.uniform((double)featureMinNumRect, (double)featureMaxNumRect));

		for (int j = 0; j<numRect; j++)
		{
			//width范围为1～Box.width-2
			rectTemp.x = cvFloor(rng.uniform(0.0, (double)(_objectBox.width - 3)));
			rectTemp.y = cvFloor(rng.uniform(0.0, (double)(_objectBox.height - 3)));
			rectTemp.width = cvCeil(rng.uniform(0.0, (double)(_objectBox.width - rectTemp.x - 2)));
			rectTemp.height = cvCeil(rng.uniform(0.0, (double)(_objectBox.height - rectTemp.y - 2)));
			features[i].push_back(rectTemp);

			weightTemp = (float)pow(-1.0, cvFloor(rng.uniform(0.0, 2.0))) / sqrt(float(numRect));
			featuresWeight[i].push_back(weightTemp);

		}
	}
}



/*在上一帧跟踪的目标box的周围采集若干正样本和负样本，来初始化或者更新分类器的*/
void CompressiveKLTracker::sampleRect(Mat& _image, Rect& _objectBox, float _rInner, float _rOuter, int _maxSampleNum, vector<Rect>& _sampleBox)
/* Description: compute the coordinate of positive and negative sample image templates
Arguments:
-_image:        processing frame
-_objectBox:    recent object position
-_rInner:       inner sampling radius
-_rOuter:       Outer sampling radius
-_maxSampleNum: maximal number of sampled images
-_sampleBox:    Storing the rectangle coordinates of the sampled images.
*/
{
	int rowsz = _image.rows - _objectBox.height - 1;
	int colsz = _image.cols - _objectBox.width - 1;
	float inradsq = _rInner*_rInner;
	float outradsq = _rOuter*_rOuter;


	int dist;

	int minrow = max(0, (int)_objectBox.y - (int)_rInner);
	int maxrow = min((int)rowsz - 1, (int)_objectBox.y + (int)_rInner);
	int mincol = max(0, (int)_objectBox.x - (int)_rInner);
	int maxcol = min((int)colsz - 1, (int)_objectBox.x + (int)_rInner);



	int i = 0;

	float prob = ((float)(_maxSampleNum)) / (maxrow - minrow + 1) / (maxcol - mincol + 1);

	int r;
	int c;

	_sampleBox.clear();//important
	Rect rec(0, 0, 0, 0);

	for (r = minrow; r <= (int)maxrow; r++)
	{

		for (c = mincol; c <= (int)maxcol; c++)
		{
			//计算生成的box到目标box的距离 
			dist = (_objectBox.y - r)*(_objectBox.y - r) + (_objectBox.x - c)*(_objectBox.x - c);

			//后两个条件是保证距离需要在_rInner和_rOuter的范围内  
			//那么rng.uniform(0.,1.) < prob 这个是干嘛的呢？  
			//连着上面看，如果_maxSampleNum大于那个最大个数，prob就大于1，这样，  
			//rng.uniform(0.,1.) < prob这个条件就总能满足，表示在这个范围产生的  
			//所以box我都要了（因为我本身想要更多的，但是你给不了我那么多，那么你能给的，我肯定全要了）。  
			//那如果你给的太多了，我不要那么多，也就是prob<1，那我就随机地跳几个走好了  
			if (rng.uniform(0., 1.) < prob && dist < inradsq && dist >= outradsq)
			{

				rec.x = c;
				rec.y = r;
				rec.width = _objectBox.width;
				rec.height = _objectBox.height;

				_sampleBox.push_back(rec);

				i++;
			}
		}
	}

	_sampleBox.resize(i);

}


/*
这个sampleRect的重载函数是用来在上一帧跟踪的目标box的周围（距离小于_srw）采集若干box来待检测。  
与上面的那个不一样，上面那个是在这一帧已经检测出目标的基础上，采集正负样本来更新分类器的。  
上面那个属于论文中提到的算法的第四个步骤，这个是第一个步骤。
*/
void CompressiveKLTracker::sampleRect(Mat& _image, Rect& _objectBox, float _srw, vector<Rect>& _sampleBox)
/* Description: Compute the coordinate of samples when detecting the object.*/
{
	int rowsz = _image.rows - _objectBox.height - 1;
	int colsz = _image.cols - _objectBox.width - 1;
	float inradsq = _srw*_srw;


	int dist;

	int minrow = max(0, (int)_objectBox.y - (int)_srw);
	int maxrow = min((int)rowsz - 1, (int)_objectBox.y + (int)_srw);
	int mincol = max(0, (int)_objectBox.x - (int)_srw);
	int maxcol = min((int)colsz - 1, (int)_objectBox.x + (int)_srw);

	int i = 0;

	int r;
	int c;

	Rect rec(0, 0, 0, 0);
	_sampleBox.clear();//important

	for (r = minrow; r <= (int)maxrow; r++)
	for (c = mincol; c <= (int)maxcol; c++){
		dist = (_objectBox.y - r)*(_objectBox.y - r) + (_objectBox.x - c)*(_objectBox.x - c);

		if (dist < inradsq){

			rec.x = c;
			rec.y = r;
			rec.width = _objectBox.width;
			rec.height = _objectBox.height;

			_sampleBox.push_back(rec);

			i++;
		}
	}

	_sampleBox.resize(i);

}




// Compute the features of samples
void CompressiveKLTracker::getFeatureValue(Mat& _imageIntegral, vector<Rect>& _sampleBox, Mat& _sampleFeatureValue)
{
	int sampleBoxSize = _sampleBox.size();
	_sampleFeatureValue.create(featureNum, sampleBoxSize, CV_32F);
	float tempValue;
	int xMin;
	int xMax;
	int yMin;
	int yMax;

	for (int i = 0; i<featureNum; i++)//随机的50个haar特征 每个haar特征有features.size()个随机矩形
	{
		for (int j = 0; j<sampleBoxSize; j++)//sampleBoxSize是采样的个数，sampleBox是objecBox附近采样的Box（有x，y。width height大小和objec的一样）
		{
			tempValue = 0.0f;
			for (size_t k = 0; k<features[i].size(); k++)
			{
				xMin = _sampleBox[j].x + features[i][k].x;
				xMax = _sampleBox[j].x + features[i][k].x + features[i][k].width;
				yMin = _sampleBox[j].y + features[i][k].y;
				yMax = _sampleBox[j].y + features[i][k].y + features[i][k].height;
				tempValue += featuresWeight[i][k] *
					(_imageIntegral.at<float>(yMin, xMin) +
					_imageIntegral.at<float>(yMax, xMax) -
					_imageIntegral.at<float>(yMin, xMax) -
					_imageIntegral.at<float>(yMax, xMin));
			}
			_sampleFeatureValue.at<float>(i, j) = tempValue;
			//50个haar特征，存下每个特征周围的sampleBoxSize个采样框积分图像值
		}
	}
}


// Update the mean and variance of the gaussian classifier
void CompressiveKLTracker::classifierUpdate(Mat& _sampleFeatureValue, vector<float>& _mu, vector<float>& _sigma, float _learnRate)
{
	Scalar muTemp;
	Scalar sigmaTemp;

	for (int i = 0; i<featureNum; i++)
	{
		//计算所有正样本或者负样本的某个harr特征的期望和标准差 
		meanStdDev(_sampleFeatureValue.row(i), muTemp, sigmaTemp);

		_sigma[i] = (float)sqrt(_learnRate*_sigma[i] * _sigma[i] + (1.0f - _learnRate)*sigmaTemp.val[0] * sigmaTemp.val[0]
			+ _learnRate*(1.0f - _learnRate)*(_mu[i] - muTemp.val[0])*(_mu[i] - muTemp.val[0]));	// equation 6 in paper

		_mu[i] = _mu[i] * _learnRate + (1.0f - _learnRate)*muTemp.val[0];	// equation 6 in paper
	}
}



// Compute the ratio classifier 
void CompressiveKLTracker::radioClassifier(vector<float>& _muPos, vector<float>& _sigmaPos, vector<float>& _muNeg, vector<float>& _sigmaNeg,
	Mat& _sampleFeatureValue, float& _radioMax, int& _radioMaxIndex)
{
	float sumRadio;
	_radioMax = -FLT_MAX;
	_radioMaxIndex = 0;
	float pPos;
	float pNeg;
	int sampleBoxNum = _sampleFeatureValue.cols;

	for (int j = 0; j<sampleBoxNum; j++)//对每一个Box计算其分数sumRadio，得分最高的即为表明其属于正样本的概率越大，即为目标的可能性越大
	{
		sumRadio = 0.0f;
		for (int i = 0; i<featureNum; i++)//featureNum对应为论文中公式4的n
		{
			pPos = exp((_sampleFeatureValue.at<float>(i, j) - _muPos[i])*(_sampleFeatureValue.at<float>(i, j) - _muPos[i]) / -(2.0f*_sigmaPos[i] * _sigmaPos[i] + 1e-30)) / (_sigmaPos[i] + 1e-30);
			pNeg = exp((_sampleFeatureValue.at<float>(i, j) - _muNeg[i])*(_sampleFeatureValue.at<float>(i, j) - _muNeg[i]) / -(2.0f*_sigmaNeg[i] * _sigmaNeg[i] + 1e-30)) / (_sigmaNeg[i] + 1e-30);
			sumRadio += log(pPos + 1e-30) - log(pNeg + 1e-30);	// equation 4
		}
		if (_radioMax < sumRadio)
		{
			_radioMax = sumRadio;
			_radioMaxIndex = j;
		}
	}

	PRINT(sumRadio);

}
void CompressiveKLTracker::init(Mat& _frame, Rect _objectBox)
{
	box1 = _objectBox;
	box2 = box1;

	vp1.reserve(nMaxPoints);
	vp2.reserve(nMaxPoints);

	//bbPoints(vp1, box1);
	bbPointsharris(_frame, vp1, box1);

	// compute feature template
	HaarFeature(box1, featureNum);

	// compute sample templates
	sampleRect(_frame, box1, rOuterPositive, 0, 1000000, samplePositiveBox);//0-rOuterPositive(4)范围内采样最大为1000000个正样本
	sampleRect(_frame, box1, rSearchWindow*1.5, rOuterPositive + 4.0, 100, sampleNegativeBox);//在rOutPositive(4)+4-25*1.5（37.5）范围内找最大为100个负样本

	integral(_frame, imageIntegral, CV_32F);

	getFeatureValue(imageIntegral, samplePositiveBox, samplePositiveFeatureValue);
	getFeatureValue(imageIntegral, sampleNegativeBox, sampleNegativeFeatureValue);
	classifierUpdate(samplePositiveFeatureValue, muPositive, sigmaPositive, learnRate);
	classifierUpdate(sampleNegativeFeatureValue, muNegative, sigmaNegative, learnRate);
	
	_frame.copyTo(preGrayFrame);
}


void CompressiveKLTracker::processFrame(Mat& _frame)
{
	//bbPoints(vp1, box1);
	bbPointsharris(_frame, vp1, box1);

	status = 0;
	status = (int)lkt.trackf2f(preGrayFrame, _frame, vp1, vp2);
	if (1 == status)
	{
		scaleRatio=bbPredict(vp1, vp2, box1, box2);
		cout << "FB Error: " << lkt.getFB() << endl;
		if (lkt.getFB() > 10 || box2.x > _frame.cols || box2.y > _frame.rows || box2.br().x < 10 || box2.br().y < 10)// origin: br().x<1
		{
			status = 0;
			cout << "KLT Tracker Failed: FB Error " << lkt.getFB() << " ,box2: " << box2 << endl;
		}
	}
	else
	{
		box2 = box1;
	}

	 
	//PRINT(scaleRatio);

	//update the features
	if (scaleRatio - 1.0>fminFloat || scaleRatio - 1.0 < -fminFloat)// 与1不等
	{
		for (int i = 0; i < featureNum; i++)
		{
			for (int k = 0; k < features.at(i).size();k++)
			{
				Rect& rec = features.at(i).at(k);
				float s1 = 0.5*(scaleRatio - 1)*rec.width;
				float s2 = 0.5*(scaleRatio - 1)*rec.height;
				
				//rec.x = (rec.x - s1);
				//rec.y = (rec.y - s2);

				rec.x *= scaleRatio;
				rec.y *= scaleRatio;

				rec.width = (rec.width*scaleRatio);
				rec.height = (rec.height*scaleRatio);
			}
		}
	}



	// predict
	sampleRect(_frame, box2, rSearchWindow, detectBox);
	integral(_frame, imageIntegral, CV_32F);
	getFeatureValue(imageIntegral, detectBox, detectFeatureValue);
	int radioMaxIndex;
	float radioMax;
	radioClassifier(muPositive, sigmaPositive, muNegative, sigmaNegative, detectFeatureValue, radioMax, radioMaxIndex);
	box1 = detectBox[radioMaxIndex];//具有最大概率的Box设为objectBox，更新数据

	// update
	sampleRect(_frame, box1, rOuterPositive, 0.0, 1000000, samplePositiveBox);//0-rOuterPositive(4)范围内采样最大为1000000个正样本
	sampleRect(_frame, box1, rSearchWindow*1.5, rOuterPositive + 4.0, 100, sampleNegativeBox);//在rOutPositive(4)+4-25*1.5（37.5）范围内找最大为100个负样本

	getFeatureValue(imageIntegral, samplePositiveBox, samplePositiveFeatureValue);
	getFeatureValue(imageIntegral, sampleNegativeBox, sampleNegativeFeatureValue);
	classifierUpdate(samplePositiveFeatureValue, muPositive, sigmaPositive, learnRate);
	classifierUpdate(sampleNegativeFeatureValue, muNegative, sigmaNegative, learnRate);

	box2 = box1;
	_frame.copyTo(preGrayFrame);
}



















//-----------------------------------------functions for KLT--------------------------------------------------
void CompressiveKLTracker::bbPoints(std::vector<cv::Point2f>& points, const cv::Rect& bb)
{
	points.clear();

	//int max_pts = 10;
	int _pts = MIN(bb.width, bb.height);
	int max_pts = (_pts> 10 ? 10 : _pts); // 当采样框长度或宽度小于10

	int margin_h = 0;
	int margin_v = 0;
	int stepx = ceil((bb.width - 2 * margin_h) / max_pts);
	int stepy = ceil((bb.height - 2 * margin_v) / max_pts);
	for (int y = bb.y + margin_v; y < bb.y + bb.height - margin_v; y += stepy)
	{
		for (int x = bb.x + margin_h; x < bb.x + bb.width - margin_h; x += stepx)
		{
			points.push_back(cv::Point2f(x, y));
		}
	}
}


void CompressiveKLTracker::bbPointsharris(cv::Mat& img, std::vector<cv::Point2f>& points, const cv::Rect& bb)
{
	points.clear();

	std::cout << bb << std::endl;

	cv::Mat mask;
	mask = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
	cv::Mat roiImage = mask(bb);
	roiImage.setTo(1);

	goodFeaturesToTrack(img, points, 100, 0.01, 1, mask);//100 0.001 5 
	//printf("%d\n", points.size());
}



float CompressiveKLTracker::bbPredict(const std::vector<cv::Point2f>& points1, const std::vector<cv::Point2f>& points2,
	const cv::Rect& bb1, cv::Rect& bb2)
{
	int npoints = (int)points1.size();

	if (npoints < nminPoints) // if the number of points less than nminPoints, we assume the prediction is inaccuracy and do not update the size of box 
	{
		bb2 = bb1;
		return 1.0;
	}


	std::vector<float> xoff(npoints);
	std::vector<float> yoff(npoints);
	//printf("tracked points : %d\n", npoints);
	for (int i = 0; i<npoints; i++)
	{
		xoff[i] = points2[i].x - points1[i].x;
		yoff[i] = points2[i].y - points1[i].y;
	}
	float dx = median(xoff);
	float dy = median(yoff);
	float s;
	if (npoints>1)
	{
		std::vector<float> d;
		d.reserve(npoints*(npoints - 1) / 2);
		for (int i = 0; i < npoints; i++)
		{
			for (int j = i + 1; j < npoints; j++)
			{
				d.push_back(norm(points2[i] - points2[j]) / norm(points1[i] - points1[j]));
			}
		}
		s = median(d);
	}
	else {
		s = 1.0;
	}
	float s1 = 0.5*(s - 1)*bb1.width;
	float s2 = 0.5*(s - 1)*bb1.height;
	//printf("s= %f s1= %f s2= %f \n", s, s1, s2);
	bb2.x = round(bb1.x + dx - s1);
	bb2.y = round(bb1.y + dy - s2);
	bb2.width = round(bb1.width*s);
	bb2.height = round(bb1.height*s);
	//printf("predicted bb: %d %d %d %d\n", bb2.x, bb2.y, bb2.br().x, bb2.br().y);

	return s;
}

