#include "stdafx.h"
#include "digitrecognizer.h"

DigitRecognizer::DigitRecognizer()
{
	knn = new KNearest();
}

DigitRecognizer::~DigitRecognizer()
{
	delete knn;
}



cv::Mat DigitRecognizer::preprocessImage(cv::Mat img)
{
	//Mat cloneImg = Mat(numRows, numCols, CV_8UC1);
	//resize(img, cloneImg, Size(numCols, numRows));

	// Try to position the given image so that the 
	// character is at the center

	// Step 1: Find a good enough bounding box
	// How? Starting at the center, find the first rows and colums
	// in 4 directions such that less than 10% of the pixels are 
	// bright

	int rowTop=-1, rowBottom=-1, colLeft=-1, colRight=-1;

	Mat temp;
	int thresholdBottom = 50;
	int thresholdTop = 50;
	int thresholdLeft = 50;
	int thresholdRight = 50;
	int center = img.rows/2;
	for(int i=center;i<img.rows;i++)
	{
		if(rowBottom==-1)
		{
			temp = img.row(i);
			IplImage stub = temp;
			if(cvSum(&stub).val[0] < thresholdBottom || i==img.rows-1)
				rowBottom = i;
		}

		if(rowTop==-1)
		{
			temp = img.row(img.rows-i);
			IplImage stub = temp;
			if(cvSum(&stub).val[0] < thresholdTop || i==img.rows-1)
				rowTop = img.rows-i;
		}

		if(colRight==-1)
		{
			temp = img.col(i);
			IplImage stub = temp;
			if(cvSum(&stub).val[0] < thresholdRight|| i==img.cols-1)
				colRight = i;
		}

		if(colLeft==-1)
		{
			temp = img.col(img.cols-i);
			IplImage stub = temp;
			if(cvSum(&stub).val[0] < thresholdLeft|| i==img.cols-1)
				colLeft = img.cols-i;
		}
	}

	// Point2i pt = Point((colLeft+colRight)/2, (rowTop+rowBottom)/2);
	/*line(img, Point(0, rowTop), Point(img.cols, rowTop), cvScalar(255,255,255));
	line(img, Point(0, rowBottom), Point(img.cols, rowBottom), cvScalar(255,255,255));
	line(img, Point(colLeft, 0), Point(colLeft, img.rows), cvScalar(255,255,255));
	line(img, Point(colRight, 0), Point(colRight, img.rows), cvScalar(255,255,255));

	imshow("Testing the image", img);
	cvWaitKey(0);*/

	// Now, position this into the center

	Mat newImg;
	newImg = newImg.zeros(img.rows, img.cols, CV_8UC1);

	int startAtX = (newImg.cols/2)-(colRight-colLeft)/2;
	int startAtY = (newImg.rows/2)-(rowBottom-rowTop)/2;

	for(int y=startAtY;y<(newImg.rows/2)+(rowBottom-rowTop)/2;y++)
	{
		uchar *ptr = newImg.ptr<uchar>(y);
		for(int x=startAtX;x<(newImg.cols/2)+(colRight-colLeft)/2;x++)
		{
			ptr[x] = img.at<uchar>(rowTop+(y-startAtY),colLeft+(x-startAtX));
		}
	}

	Mat cloneImg = Mat(numRows, numCols, CV_8UC1);
	resize(newImg, cloneImg, Size(numCols, numRows));

	// Now fill along the borders
	for(int i=0;i<cloneImg.rows;i++)
	{
		floodFill(cloneImg, cvPoint(0, i), cvScalar(0,0,0));
		floodFill(cloneImg, cvPoint(cloneImg.cols-1, i), cvScalar(0,0,0));

		floodFill(cloneImg, cvPoint(i, 0), cvScalar(0));
		floodFill(cloneImg, cvPoint(i, cloneImg.rows-1), cvScalar(0));
	}

	imshow("testing image", cloneImg);

	cloneImg = cloneImg.reshape(1, 1);

	return cloneImg;
}
int DigitRecognizer::classify(cv::Mat img)
{
	Mat cloneImg = preprocessImage(img);

	return knn->find_nearest(Mat_<float>(cloneImg), 1);
}

int DigitRecognizer::readFlippedInteger(FILE *fp)
{
	int ret = 0;
	BYTE *temp;

	temp = (BYTE*)(&ret);
	fread(&temp[3], sizeof(BYTE), 1, fp);
	fread(&temp[2], sizeof(BYTE), 1, fp);
	fread(&temp[1], sizeof(BYTE), 1, fp);
	fread(&temp[0], sizeof(BYTE), 1, fp);

	return ret;
}

bool DigitRecognizer::train(char *trainPath, char *labelsPath)
{
	FILE *fp = fopen(trainPath, "rb");
	FILE *fp2 = fopen(labelsPath, "rb");

	if(!fp || !fp2)
		return false;

	// Read bytes in flipped order
	int magicNumber = readFlippedInteger(fp);	
	numImages = readFlippedInteger(fp);
	numRows = readFlippedInteger(fp);
	numCols = readFlippedInteger(fp);

	// printf("Magic number: %4x\n", magicNumber);
	//printf("Number of images: %d\n", numImages);
	//printf("Number of rows: %d\n", numRows);
	//printf("Number of columns: %d\n", numCols);

	fseek(fp2, 0x08, SEEK_SET);

	if(numImages > MAX_NUM_IMAGES) numImages = MAX_NUM_IMAGES;

	//////////////////////////////////////////////////////////////////
	// Go through each training data entry and figure out a 
	// center for each digit

	int size = numRows*numCols;
	CvMat *trainingVectors = cvCreateMat(numImages, size, CV_32FC1);
	CvMat *trainingClasses = cvCreateMat(numImages, 1, CV_32FC1);

	memset(trainingClasses->data.ptr, 0, sizeof(float)*numImages);

	BYTE *temp = new BYTE[size];
	BYTE tempClass=0;
	for(int i=0;i<numImages;i++)
	{
		fread((void*)temp, size, 1, fp);
		fread((void*)(&tempClass), sizeof(BYTE), 1, fp2);

		trainingClasses->data.fl[i] = tempClass;

		// Normalize the vector
		/*float sumofsquares = 0;
		for(int k=0;k<size;k++)
			sumofsquares+=temp[k]*temp[k];
		sumofsquares = sqrt(sumofsquares);*/

		for(int k=0;k<size;k++)
			trainingVectors->data.fl[i*size+k] = temp[k]; ///sumofsquares;
	}

	knn->train(trainingVectors, trainingClasses);
	fclose(fp);
	fclose(fp2);

	return true;
}