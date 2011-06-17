// Sudoku Solver.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <cv.h>
#include <highgui.h>

#include "digitrecognizer.h"

using namespace cv;

// Record the execution time of some code, in milliseconds.
#define DECLARE_TIMING(s)  int64 timeStart_##s; double timeDiff_##s; double timeTally_##s = 0; int countTally_##s = 0
#define START_TIMING(s)    timeStart_##s = cvGetTickCount()
#define STOP_TIMING(s) 	   timeDiff_##s = (double)(cvGetTickCount() - timeStart_##s); timeTally_##s += timeDiff_##s; countTally_##s++
#define GET_TIMING(s) 	   (double)(timeDiff_##s / (cvGetTickFrequency()*1000.0))
#define GET_AVERAGE_TIMING(s)   (double)(countTally_##s ? timeTally_##s/ ((double)countTally_##s * cvGetTickFrequency()*1000.0) : 0)
#define CLEAR_AVERAGE_TIMING(s) timeTally_##s = 0; countTally_##s = 0

void drawLine(Vec2f line, Mat &img, Scalar rgb = CV_RGB(0,0,255))
{
	if(line[1]!=0)
	{
		float m = -1/tan(line[1]);
		float c = line[0]/sin(line[1]);

		cv::line(img, Point(0, c), Point(img.size().width, m*img.size().width+c), rgb);
	}
	else
	{
		cv::line(img, Point(line[0], 0), Point(line[0], img.size().height), rgb);
	}
}

void drawLine(Vec4i line, Mat &img)
{
	cv::line(img, Point(line[0], line[1]), Point(line[2], line[3]), CV_RGB(255,0,255));
}

void mergeRelatedLines(vector<Vec2f> *lines, Mat &img)
{
	vector<Vec2f>::iterator current;
	vector<Vec4i> points(lines->size());
	for(current=lines->begin();current!=lines->end();current++)
	{
		if((*current)[0]==0 && (*current)[1]==-100)
			continue;

		float p1 = (*current)[0];
		float theta1 = (*current)[1];

		Point pt1current, pt2current;
		if(theta1>CV_PI*45/180 && theta1<CV_PI*135/180)
		{
			pt1current.x=0;
			pt1current.y = p1/sin(theta1);

			pt2current.x=img.size().width;
			pt2current.y=-pt2current.x/tan(theta1) + p1/sin(theta1);
		}
		else
		{
			pt1current.y=0;
			pt1current.x=p1/cos(theta1);

			pt2current.y=img.size().height;
			pt2current.x=-pt2current.y/tan(theta1) + p1/cos(theta1);
		}

		vector<Vec2f>::iterator	pos;
		for(pos=lines->begin();pos!=lines->end();pos++)
		{
			if(*current==*pos)
				continue;

			if(fabs((*pos)[0]-(*current)[0])<20 && fabs((*pos)[1]-(*current)[1])<CV_PI*10/180)
			{
				float p = (*pos)[0];
				float theta = (*pos)[1];

				Point pt1, pt2;
				if((*pos)[1]>CV_PI*45/180 && (*pos)[1]<CV_PI*135/180)
				{
					pt1.x=0;
					pt1.y = p/sin(theta);

					pt2.x=img.size().width;
					pt2.y=-pt2.x/tan(theta) + p/sin(theta);
				}
				else
				{
					pt1.y=0;
					pt1.x=p/cos(theta);

					pt2.y=img.size().height;
					pt2.x=-pt2.y/tan(theta) + p/cos(theta);
				}

				if(((double)(pt1.x-pt1current.x)*(pt1.x-pt1current.x) + (pt1.y-pt1current.y)*(pt1.y-pt1current.y)<64*64) && ((double)(pt2.x-pt2current.x)*(pt2.x-pt2current.x) + (pt2.y-pt2current.y)*(pt2.y-pt2current.y)<64*64))
				{
					printf("Merging\n");
					// Merge the two
					(*current)[0] = ((*current)[0]+(*pos)[0])/2;
					(*current)[1] = ((*current)[1]+(*pos)[1])/2;

					(*pos)[0]=0;
					(*pos)[1]=-100;
					//lines->erase(pos);
				}
			}
		}
	}

	//return lines;
}

void findX(IplImage* imgSrc,int* min, int* max){
	int i;
	int minFound=0;
	CvMat data;
	CvScalar maxVal=cvRealScalar(imgSrc->width * 255);
	CvScalar val=cvRealScalar(0);
	//For each col sum, if sum < width*255 then we find the min 
	//then continue to end to search the max, if sum< width*255 then is new max
	for (i=0; i< imgSrc->width; i++){
		cvGetCol(imgSrc, &data, i);
		val= cvSum(&data);
		if(val.val[0] < maxVal.val[0]){
			*max= i;
			if(!minFound){
				*min= i;
				minFound= 1;
			}
		}
	}
}

void findY(IplImage* imgSrc,int* min, int* max){
	int i;
	int minFound=0;
	CvMat data;
	CvScalar maxVal=cvRealScalar(imgSrc->width * 255);
	CvScalar val=cvRealScalar(0);
	//For each col sum, if sum < width*255 then we find the min 
	//then continue to end to search the max, if sum< width*255 then is new max
	for (i=0; i< imgSrc->height; i++)
	{
		cvGetRow(imgSrc, &data, i);
		val= cvSum(&data);
		if(val.val[0] < maxVal.val[0]){
			*max=i;
			if(!minFound){
				*min= i;
				minFound= 1;
			}
		}
	}
}

CvRect findBB(Mat imgSrc)
{
	CvRect aux;
	int xmin, xmax, ymin, ymax;
	xmin=xmax=ymin=ymax=0;

	IplImage toCheck = imgSrc;

	findX(&toCheck, &xmin, &xmax);
	findY(&toCheck, &ymin, &ymax);
	
	aux=cvRect(xmin, ymin, xmax-xmin, ymax-ymin);
	
	return aux;
	
}

int main()
{
	Mat sudoku = imread("sudoku.jpg",0);

	//Mat sudoku;
	//pyrDown(imread("sudoku3.jpg", 0), sudoku);

	Mat original = sudoku.clone();

	DECLARE_TIMING(sudTimer);
	START_TIMING(sudTimer);

	// Create a duplicate. We'll try to extract grid lines in this image
	Mat outerBox = Mat(sudoku.size(), CV_8UC1);
	
	
	//erode(sudoku, sudoku, kernel);

	GaussianBlur(sudoku, sudoku, Size(11,11), 0);
	adaptiveThreshold(sudoku, outerBox, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 5, 2);

	bitwise_not(outerBox, outerBox);

	Mat kernel = (Mat_<uchar>(3,3) << 0,1,0,1,1,1,0,1,0);
	dilate(outerBox, outerBox, kernel);

	
	
	int count=0;
	int max=-1;
	Point maxPt;

	Mat cloneOuterBox = outerBox.clone();

	for(int y=0;y<outerBox.size().height;y++)
	{
		uchar *row = outerBox.ptr(y);
		for(int x=0;x<outerBox.size().width;x++)
		{
			if(row[x]>=128)
			{
				int area = floodFill(outerBox, Point(x,y), CV_RGB(0,0,64));

				if(area>max)
				{
					maxPt = Point(x,y);
					max = area;
				}
			}
		}

			
		//printf("Current row: %d\n", y);
	}

	floodFill(outerBox, maxPt, CV_RGB(255,255,255));
	
	for(int y=0;y<outerBox.size().height;y++)
	{
		uchar *row = outerBox.ptr(y);
		for(int x=0;x<outerBox.size().width;x++)
		{
			if(row[x]==64 && x!=maxPt.x && y!=maxPt.y)
			{
				int area = floodFill(outerBox, Point(x,y), CV_RGB(0,0,0));
			}
		}
		//printf("Current row: %d\n", y);
	}

	erode(outerBox, outerBox, kernel);

	//imshow("thresholded", outerBox);

	vector<Vec2f> lines;
	HoughLines(outerBox, lines, 1, CV_PI/180, 200);

	//vector<Vec2f>::iterator pos;
	mergeRelatedLines(&lines, sudoku);

	printf("Size of lines: %d\n", lines.size());
	for(int i=0;i<lines.size();i++)
	{
		drawLine(lines[i], outerBox, CV_RGB(0,0,128));
	}

	imshow("thresholded", outerBox);

	// Now detect the lines on extremes
	Vec2f topEdge = Vec2f(1000,1000);	double topYIntercept=100000, topXIntercept=0;
	Vec2f bottomEdge = Vec2f(-1000,-1000);		double bottomYIntercept=0, bottomXIntercept=0;
	Vec2f leftEdge = Vec2f(1000,1000);	double leftXIntercept=100000, leftYIntercept=0;
	Vec2f rightEdge = Vec2f(-1000,-1000);		double rightXIntercept=0, rightYIntercept=0;
	for(int i=0;i<lines.size();i++)
	{
		Vec2f current = lines[i];

		float p=current[0];
		float theta=current[1];

		if(p==0 && theta==-100)
			continue;

		double xIntercept, yIntercept;
		xIntercept = p/cos(theta);
		yIntercept = p/(cos(theta)*sin(theta));

		if(theta>CV_PI*80/180 && theta<CV_PI*100/180)
		{
			if(p<topEdge[0])
				topEdge = current;

			if(p>bottomEdge[0])
				bottomEdge = current;

			//printf("X: %f, Y: %f\n", xIntercept, yIntercept);
			
		}
		else if(theta<CV_PI*10/180 || theta>CV_PI*170/180)
		{
			/*if(p<leftEdge[0])
				leftEdge = current;
			
			if(p>rightEdge[0])
				rightEdge = current;*/

			if(xIntercept>rightXIntercept)
			{
				rightEdge = current;
				rightXIntercept = xIntercept;
			} 
			else if(xIntercept<=leftXIntercept)
			{
				leftEdge = current;
				leftXIntercept = xIntercept;
			}
		}
	}

	
	drawLine(topEdge, sudoku, CV_RGB(0,0,0));
	drawLine(bottomEdge, sudoku, CV_RGB(0,0,0));
	drawLine(leftEdge, sudoku, CV_RGB(0,0,0));
	drawLine(rightEdge, sudoku, CV_RGB(0,0,0));

	Point left1, left2, right1, right2, bottom1, bottom2, top1, top2;

	int height=outerBox.size().height;
	int width=outerBox.size().width;

	if(leftEdge[1]!=0)
	{
		left1.x=0;		left1.y=leftEdge[0]/sin(leftEdge[1]);
		left2.x=width;	left2.y=-left2.x/tan(leftEdge[1]) + left1.y;
	}
	else
	{
		left1.y=0;		left1.x=leftEdge[0]/cos(leftEdge[1]);
		left2.y=height;	left2.x=left1.x - height*tan(leftEdge[1]);
	}

	if(rightEdge[1]!=0)
	{
		right1.x=0;		right1.y=rightEdge[0]/sin(rightEdge[1]);
		right2.x=width;	right2.y=-right2.x/tan(rightEdge[1]) + right1.y;
	}
	else
	{
		right1.y=0;		right1.x=rightEdge[0]/cos(rightEdge[1]);
		right2.y=height;	right2.x=right1.x - height*tan(rightEdge[1]);
	}

	bottom1.x=0;	bottom1.y=bottomEdge[0]/sin(bottomEdge[1]);
	bottom2.x=width;bottom2.y=-bottom2.x/tan(bottomEdge[1]) + bottom1.y;

	top1.x=0;		top1.y=topEdge[0]/sin(topEdge[1]);
	top2.x=width;	top2.y=-top2.x/tan(topEdge[1]) + top1.y;

	// Next, we find the intersection of  these four lines
    double leftA = left2.y-left1.y;
    double leftB = left1.x-left2.x;
    double leftC = leftA*left1.x + leftB*left1.y;
 
    double rightA = right2.y-right1.y;
    double rightB = right1.x-right2.x;
    double rightC = rightA*right1.x + rightB*right1.y;
 
    double topA = top2.y-top1.y;
    double topB = top1.x-top2.x;
    double topC = topA*top1.x + topB*top1.y;
 
    double bottomA = bottom2.y-bottom1.y;
    double bottomB = bottom1.x-bottom2.x;
    double bottomC = bottomA*bottom1.x + bottomB*bottom1.y;
 
    // Intersection of left and top
    double detTopLeft = leftA*topB - leftB*topA;
    CvPoint ptTopLeft = cvPoint((topB*leftC - leftB*topC)/detTopLeft, (leftA*topC - topA*leftC)/detTopLeft);
 
    // Intersection of top and right
    double detTopRight = rightA*topB - rightB*topA;
    CvPoint ptTopRight = cvPoint((topB*rightC-rightB*topC)/detTopRight, (rightA*topC-topA*rightC)/detTopRight);
 
    // Intersection of right and bottom
    double detBottomRight = rightA*bottomB - rightB*bottomA;
    CvPoint ptBottomRight = cvPoint((bottomB*rightC-rightB*bottomC)/detBottomRight, (rightA*bottomC-bottomA*rightC)/detBottomRight);
 
    // Intersection of bottom and left
    double detBottomLeft = leftA*bottomB-leftB*bottomA;
    CvPoint ptBottomLeft = cvPoint((bottomB*leftC-leftB*bottomC)/detBottomLeft, (leftA*bottomC-bottomA*leftC)/detBottomLeft);

	cv::line(sudoku, ptTopRight, ptTopRight, CV_RGB(255,0,0), 10);
	cv::line(sudoku, ptTopLeft, ptTopLeft, CV_RGB(255,0,0), 10);
	cv::line(sudoku, ptBottomRight, ptBottomRight, CV_RGB(255,0,0), 10);
	cv::line(sudoku, ptBottomLeft, ptBottomLeft, CV_RGB(255,0,0), 10);

	// Correct the perspective transform
	int maxLength = (ptBottomLeft.x-ptBottomRight.x)*(ptBottomLeft.x-ptBottomRight.x) + (ptBottomLeft.y-ptBottomRight.y)*(ptBottomLeft.y-ptBottomRight.y);
	int temp = (ptTopRight.x-ptBottomRight.x)*(ptTopRight.x-ptBottomRight.x) + (ptTopRight.y-ptBottomRight.y)*(ptTopRight.y-ptBottomRight.y);
	if(temp>maxLength) maxLength = temp;

	temp = (ptTopRight.x-ptTopLeft.x)*(ptTopRight.x-ptTopLeft.x) + (ptTopRight.y-ptTopLeft.y)*(ptTopRight.y-ptTopLeft.y);
	if(temp>maxLength) maxLength = temp;

	temp = (ptBottomLeft.x-ptTopLeft.x)*(ptBottomLeft.x-ptTopLeft.x) + (ptBottomLeft.y-ptTopLeft.y)*(ptBottomLeft.y-ptTopLeft.y);
	if(temp>maxLength) maxLength = temp;

	maxLength = sqrt((double)maxLength);
	

	Point2f src[4], dst[4];
	src[0] = ptTopLeft;			dst[0] = Point2f(0,0);
	src[1] = ptTopRight;		dst[1] = Point2f(maxLength-1, 0);
	src[2] = ptBottomRight;		dst[2] = Point2f(maxLength-1, maxLength-1);
	src[3] = ptBottomLeft;		dst[3] = Point2f(0, maxLength-1);

	Mat undistorted = Mat(Size(maxLength, maxLength), CV_8UC1);
	cv::warpPerspective(original, undistorted, cv::getPerspectiveTransform(src, dst), Size(maxLength, maxLength));

	

STOP_TIMING(sudTimer);

	printf("Time taken: %f\n", GET_TIMING(sudTimer));
	imshow("Lines", outerBox);
	imshow("SuDoKu", sudoku);

	Mat undistortedThreshed = undistorted.clone();

	// Show this sample
	//threshold(undistorted, undistortedThreshed, 128, 255, CV_THRESH_BINARY_INV);
	adaptiveThreshold(undistorted, undistortedThreshed, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 101, 1);

	imshow("undistorted", undistortedThreshed);
	waitKey(0);

	DigitRecognizer *dr = new DigitRecognizer();
	bool b = dr->train("D:/Test/Character Recognition/train-images.idx3-ubyte", "D:/Test/Character Recognition/train-labels.idx1-ubyte");
	printf("Trained: %d\n", b);

	int dist = ceil((double)maxLength/9);
	Mat currentCell = Mat(dist, dist, CV_8UC1);
	for(int j=0;j<9;j++)
	{
		for(int i=0;i<9;i++)
		{
			// We copy the current cell into a separate image
			// undistortedThreshed.adjustROI(i*dist, i*dist+dist, j*dist, j*dist+dist);
			for(int y=0;y<dist && j*dist+y<undistortedThreshed.cols;y++)
			{
				uchar* ptr = currentCell.ptr(y);
				// uchar* ptr2 = &(undistortedThreshed.ptr<uchar>(i*dist)[j*dist]);
				for(int x=0;x<dist && i*dist+x<undistortedThreshed.rows;x++)
				{
					ptr[x] = undistortedThreshed.at<uchar>(j*dist+y, i*dist+x);
				}
			}

			// Fill edges with black color
			/*for(int l=0;l<dist/10;l++)
			{
				floodFill(currentCell, Point(l,l), cvScalar(0));
				floodFill(currentCell, Point(l,dist-l-1), cvScalar(0));
				floodFill(currentCell, Point(dist-l-1,l), cvScalar(0));
				floodFill(currentCell, Point(dist-l-1,dist-l-1), cvScalar(0));
			}*/

			/*CvRect bb = findBB(currentCell);
			Mat currentToProcess = Mat(bb.height-bb.x, bb.width-bb.x, CV_8UC1);
			for(int y=bb.y;y<bb.height;y++)
			{
				uchar* ptr = currentToProcess.ptr(y);
				// uchar* ptr2 = &(undistortedThreshed.ptr<uchar>(i*dist)[j*dist]);
				for(int x=bb.x;x<bb.width;x++)
				{
					ptr[x] = currentCell.at<uchar>(y, x);
				}
			}*/

			Moments m = cv::moments(currentCell, true);
			int area = m.m00;
			if(area > currentCell.rows*currentCell.cols/5)
			{
				int number = dr->classify(currentCell);
				//printf("Classified as: %d\n", number);

				// printf("Shown\n");
				printf("%d ", number);
				/*imshow("test", currentCell);
				waitKey(0);*/
			}
			else
			{
				printf("  ");
			}

			
		}
		printf("\n");
	}

	waitKey(0);

	return 0;
}

