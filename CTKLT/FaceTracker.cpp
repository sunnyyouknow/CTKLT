#include "stdafx.h"
#include "FaceTracker.h"

extern const cv::Scalar obColors[];
extern const float fNotSure;


void initCTKLTS(cv::Mat &img, std::vector<CompressiveKLTracker> & _ctklts)
{
	for (int i = 0; i < _ctklts.size(); i++)
	{
		_ctklts.at(i).init(img, _ctklts.at(i).box0);
	}
}



void rectangleCTKLTS(cv::Mat &img, const std::vector<CompressiveKLTracker> & _ctklts)
{
	for (int i = 0; i < _ctklts.size(); i++)
	{
		//cv::rectangle(img, _ctklts.at(i).box0, obColors[2], 2);
		//cv::rectangle(img, _ctklts.at(i).box2, obColors[3], 2);

		//cv::rectangle(img, _ctklts.at(i).box0, obColors[_ctklts.at(i).id], 2);
		if (_ctklts.at(i).confidence < fNotSure)
		{
			char strName[256];
			//sprintf(strName, "%s %f?", obNames[_ctklts.at(i).id].c_str(), _ctklts.at(i).confidence);
			sprintf(strName, "%s ?", obNames[_ctklts.at(i).id].c_str());
			cv::putText(img, strName, cv::Point(_ctklts.at(i).box2.x, _ctklts.at(i).box2.y - 10), 2, 0.8, obColors[_ctklts.at(i).id]);
		}
		else
		{
			cv::putText(img, obNames[_ctklts.at(i).id], cv::Point(_ctklts.at(i).box2.x, _ctklts.at(i).box2.y - 10), 2, 0.8, obColors[_ctklts.at(i).id]);
		}
		cv::rectangle(img, _ctklts.at(i).box2, obColors[_ctklts.at(i).id], 2);
	}
}


void rectangleDET(cv::Mat &img, const std::vector<cv::Rect> & rec)
{
	for (int i = 0; i < rec.size(); i++)
	{
		cv::rectangle(img, rec.at(i), cv::Scalar(0, 255, 0), 2);
	}
}


void addDescription(cv::Mat &img, const int _frameIdx, const std::string _description)
{
	char strFrame[256];
	sprintf(strFrame, "#%d", _frameIdx);
	cv::putText(img, strFrame, cv::Point(0, 20), 2, 1, CV_RGB(25, 200, 25));
	cv::putText(img, _description, cv::Point(150, 20), 2, 1, CV_RGB(25, 200, 25));
}




int trackAndCheck(cv::Mat & grayImg, std::vector<CompressiveKLTracker> & _ctklts)
{
	for (int i = 0; i < _ctklts.size(); i++)
	{
		CompressiveKLTracker &ctklt = _ctklts.at(i);
		ctklt.status = 0;

		ctklt.status = ctklt.processFrame(grayImg);

		if (0 == ctklt.status)
		{
			printf("tracked failure: fail to track the object\n");
		}
	}



	int failCount = 0;


	std::vector<CompressiveKLTracker>::iterator ite;
	for (ite = _ctklts.begin(); ite != _ctklts.end();)
	{
		if (0 == (*ite).status)//status
		{
			ite = _ctklts.erase(ite);
			failCount++;
		}
		else
		{
			ite++;
		}
	}


	return failCount;
}



void rectangleDecIden(cv::Mat &img, const std::vector<LocationIdentification> & lid)
{
	for (int i = 0; i < lid.size(); i++)
	{
		cv::rectangle(img, lid.at(i).rec, cv::Scalar(0, 255, 0), 2);
	}
}