#include <bits/stdc++.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
//Library to read images
#include <dirent.h>
using namespace cv;
using namespace std;

//vector to store image names
vector<string> imgs_name;
int img_number = 0;
int max_images = 0;
int structuring_element= 0;
int operation = 0;
int total_operation = 4; 

Mat input_img;

//Structuring Element
vector<vector<int> > SE_1D(1, vector<int>(2, 1));
vector<vector<int> > SE_3(3, vector<int>(3, 1));
vector<vector<int> > SE_9(9, vector<int>(9, 1));
vector<vector<int> > SE_15(15, vector<int>(15, 1));
vector<vector<int> > SE_3_with0(3, vector<int>(3, 1));

/*
It takes binary image and structuring elements as input and returns the Dilated binary image in Mat format. 
Inside it, it iterates over every white pixel and places the structuring element center at that white pixel 
and will replace all the pixels that overlap with 1 value of the structuring elements with the white value.
*/

Mat DilateBinary(Mat img, vector<vector<int> > SE)
{
	Mat output_image(img.rows, img.cols, CV_8UC1, Scalar(0));
	int y = SE.size();
	int x = SE[0].size();
	int x_center = (x-1)/2;
	int y_center = (y-1)/2;
	int ii, jj;
	for(int h=0; h< img.rows; h++)
		for(int w =0; w<img.cols; w++)
		{
			if(int(img.at<uchar>(Point(h,w))) == 255)
			{
				for(int yy=0; yy<y; yy++)
					for(int xx=0; xx<x; xx++)
					{
						if(SE[yy][xx] ==1)
						{
							ii = h + yy - y_center;
							jj = w + xx - x_center;
							if((ii >=0 && jj>=0) && (ii<img.rows && jj<img.cols))
							{
								output_image.at<uchar>(Point(ii, jj)) = 255;
							}
						}

					}
			}
		}

	return output_image;
}
/*
It takes binary image and structuring elements as input and returns the Eroded binary image in Mat format. 
Inside it, it iterates over every pixel of the input image, 
places the structuring element center at that pixel 
and then if any 1 value of the structuring element coincides with 
black value of the image it replaces that center pixel with black value 
else keep it white pixel.
*/
Mat ErodeBinary(Mat img, vector<vector<int> > SE)
{
	Mat output_image(img.rows, img.cols, CV_8UC1, Scalar(0));
	int y = SE.size();
	int x = SE[0].size();
	int x_center = (x-1)/2;
	int y_center = (y-1)/2;
	int ii, jj;
	for(int h=0; h< img.rows; h++)
		for(int w =0; w<img.cols; w++)
		{
			bool fill_white = true;
			for(int yy=0; yy<y; yy++)
				for(int xx=0; xx<x; xx++)
				{
					ii = h + yy - y_center;
					jj = w + xx - x_center;
					if((ii < 0) || (jj<0) || (ii>=img.rows) || (jj>=img.cols) || (SE[yy][xx] != 0 && int(img.at<uchar>(Point(ii, jj))) != 255))
						fill_white = false;
				}
		
			if(fill_white)
				output_image.at<uchar>(Point(h, w)) = 255;
			else
				output_image.at<uchar>(Point(h, w)) = 0;
		}

	return output_image;
}

//Function to perform opening operation
//It simply perform erosion then dilation
Mat OpenBinary(Mat img, vector<vector<int> > SE)
{
	Mat output_image(img.rows, img.cols, CV_8UC1, Scalar(0));
	output_image = ErodeBinary(img, SE);
	output_image = DilateBinary(output_image, SE);
	return output_image;
}

//Function to perform closing operation
//It simply perform dilation then erosion
Mat CloseBinary(Mat img, vector<vector<int> > SE)
{
	Mat output_image(img.rows, img.cols, CV_8UC1, Scalar(0));
	output_image = DilateBinary(img, SE);
	output_image = ErodeBinary(output_image, SE);
	return output_image;
}

//Function to read images from the folder
vector<string> read_images(string path)
{
	vector<string> imgs;
	string img_name;
	string temp;
	string img_path = path;
	const char* path_img = &img_path[0];
	DIR *dir = opendir(path_img);
	struct dirent *dp;
	//Make vector of images name placed in path
	if(dir != NULL)
	{
		while(( dp = readdir(dir)) != NULL)
		{
			img_name = dp->d_name;
			//check for garbage value
			if(img_name.length() >=4)
			{
				temp = img_name.substr(img_name.length() - 3, 3);
				if(temp == "bmp")
				imgs.push_back(img_path + "/" + img_name);
			}
		}
	}
	(void)closedir(dir);

	return imgs;
}
//Function to perform the selected operation using structuring element 
//as given by trackbar
static void Filter_image(int, void*)
{

	Mat out_image;

	if(operation == 0 || structuring_element ==0)
	{
		imshow("Images", input_img);
		imshow("Output Image", input_img);
		imwrite("original_image.jpg", input_img);
	}
	else
	{
		if(operation==1)
		{
			if(structuring_element ==1)
				out_image = ErodeBinary(input_img, SE_1D);
			else if(structuring_element ==2)
				out_image = ErodeBinary(input_img, SE_3);
			else if(structuring_element == 3)
				out_image = ErodeBinary(input_img, SE_3_with0);
			else if(structuring_element == 4)
				out_image = ErodeBinary(input_img, SE_9);
			else
				out_image = ErodeBinary(input_img, SE_15);

			imshow("Images", input_img);
			imshow("Output Image", out_image);

		}
		else if(operation==2)
		{
			if(structuring_element ==1)
				out_image = DilateBinary(input_img, SE_1D);
			else if(structuring_element ==2)
				out_image = DilateBinary(input_img, SE_3);
			else if(structuring_element == 3)
				out_image = DilateBinary(input_img, SE_3_with0);
			else if(structuring_element == 4)
				out_image = DilateBinary(input_img, SE_9);
			else
				out_image = DilateBinary(input_img, SE_15);

			imshow("Images", input_img);
			imshow("Output Image", out_image);
		}
		else if(operation ==3)
		{
			if(structuring_element ==1)
				out_image = OpenBinary(input_img, SE_1D);
			else if(structuring_element ==2)
				out_image = OpenBinary(input_img, SE_3);
			else if(structuring_element == 3)
				out_image = OpenBinary(input_img, SE_3_with0);
			else if(structuring_element == 4)
				out_image = OpenBinary(input_img, SE_9);
			else
				out_image = OpenBinary(input_img, SE_15);

			imshow("Images", input_img);
			imshow("Output Image", out_image);
		}
		else if(operation ==4)
		{
			if(structuring_element ==1)
				out_image = CloseBinary(input_img, SE_1D);
			else if(structuring_element ==2)
				out_image = CloseBinary(input_img, SE_3);
			else if(structuring_element == 3)
				out_image = CloseBinary(input_img, SE_3_with0);
			else if(structuring_element == 4)
				out_image = CloseBinary(input_img, SE_9);
			else
				out_image = CloseBinary(input_img, SE_15);
			

			imshow("Images", input_img);
			imshow("Output Image", out_image);
		}
		else
		{
			imshow("Images", input_img);
			imshow("Output Image", input_img);
		}
	}
}

//Function to read the image to global Mat image
static void Select_img(int, void*)
{
	input_img = imread(imgs_name[img_number], IMREAD_UNCHANGED);
	imshow("Images", input_img);
	Filter_image(0,0);
}


int main(int argc, char *argv[])
{
	//Check for valid argument
	if(argc != 2)
	{
		cout << "Please provide single argument specifying path to image stack" << endl;
	}
	else
	{
		//store the path to image folders
		string img_path = argv[1];
		
		//Read images
		imgs_name = read_images(img_path);
		max_images = imgs_name.size();
		namedWindow("Images", CV_WINDOW_AUTOSIZE);
		namedWindow("Output Image", CV_WINDOW_AUTOSIZE);
		SE_3_with0[0][0] = 0;
		SE_3_with0[0][2] = 0;
		SE_3_with0[2][0] = 0;
		SE_3_with0[2][2] = 0;
		if(max_images ==0)
			cout << "No .bmp image in path provided" << endl;
		//if only 1 image is found no need to create select image trackbar, so create only
		//select operation and select structuring element trackbar.
		else if(max_images ==1)
		{
			input_img = imread(imgs_name[0], IMREAD_UNCHANGED);
			createTrackbar("Select Operation", "Images", &operation, total_operation, Filter_image);
			createTrackbar("Select structuring element", "Images", &structuring_element, 5 , Filter_image);
			Filter_image(0,0);
			waitKey();
		}
		//else create select image trackbar also
		else
		{
			createTrackbar("Select images", "Images", &img_number, max_images, Select_img);
			createTrackbar("Select Operation", "Images", &operation, total_operation, Filter_image);
			createTrackbar("Select structuring element", "Images", &structuring_element, 5 , Filter_image);
			//Call one of the call back function for initialization 
			Select_img(0, 0);
			waitKey();
		}
	}
	return 0;
}
