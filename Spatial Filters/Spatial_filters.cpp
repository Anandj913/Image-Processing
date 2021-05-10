#include <bits/stdc++.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
//Library to read images
#include <dirent.h>
using namespace cv;
using namespace std;

//Folder name in which images are kept
string noise_folder_name = "Noisy Images";
string normal_folder_name = "Normal Images";

vector<string> imgs_name;
int img_number = 0;
int filter_size = 0	;
int filter_number = 0;
int max_filter_size = 9; 
Mat input_img;

//LOG filter coffecient
double log_3[3][3] = {{0.0, 1.0/4, 0.0}, {1.0/4, -1.0, 1.0/4}, {0.0, 1.0/4, 0.0}};
double log_5[5][5] = {{0.0, 0.0, 1.0/16, 0.0, 0.0}, 
					  {0.0, 1.0/16, 2.0/16, 1.0/16, 0.0}, 
					  {1.0/16, 2.0/16, -1.0, 2.0/16, 1.0/16}, 
					  {0.0, 1.0/16, 2.0/16, 1.0/16, 0.0}, 
					  {0.0, 0.0, 1.0/16, 0.0, 0.0}};

double log_7[7][7] = {{0.0, 0.0, 1.0/52, 1.0/52, 1.0/52, 0.0, 0.0}, 
                            {0.0, 1.0/52, 3.0/52, 3.0/52, 3.0/52, 1.0/52, 0.0},
                            {1.0/52, 3.0/52, 0.0, -7.0/52, 0.0 ,3.0/52, 1.0/52},
                            {1.0/52, 3.0/52, -7.0/52, -24.0/52 ,-7.0/52, 3.0/52, 1.0/52},
                            {1.0/52, 3.0/52, 0.0, -7.0/52, 0.0 ,3.0/52, 1.0/52},
                            {0.0, 1.0/52, 3.0/52, 3.0/52, 3.0/52, 1.0/52, 0.0},
                            {0.0, 0.0, 1.0/52, 1.0/52, 1.0/52, 0.0, 0.0}};

double log_9[9][9] = {{0.0, 0.0, 3.0, 2.0, 2.0, 2.0, 3.0, 0.0, 0.0},
							{0.0, 2.0, 3.0, 5.0, 5.0, 5.0, 3.0, 2.0, 0.0},
							{3.0, 3.0, 5.0, 3.0, 0.0, 3.0, 5.0, 3.0, 3.0},
							{2.0, 5.0, 3.0, -12.0, -23.0, -12.0, 3.0, 5.0, 2.0},
							{2.0, 5.0, 0.0, -23.0, -40.0, -23.0, 0.0, 5.0, 2.0},
							{2.0, 5.0, 3.0, -12.0, -23.0, -12.0, 3.0, 5.0, 2.0},
							{3.0, 3.0, 5.0, 3.0, 0.0, 3.0, 5.0, 3.0, 3.0},
							{0.0, 2.0, 3.0, 5.0, 5.0, 5.0, 3.0, 2.0, 0.0},
							{0.0, 0.0, 3.0, 2.0, 2.0, 2.0, 3.0, 0.0, 0.0}};

//Laplacian filter coffecient
double laplacian_3[3][3] = {{-1.0, -1.0, -1.0}, {-1.0, 8.0, -1.0}, {-1.0, -1.0, -1.0}};
double laplacian_5[5][5] = {{-1.0, -3.0, -4.0, -3.0, -1.0}, 
					        {-3.0, 0.0, 6.0, 0.0, -3.0}, 
					        {-4.0, 6.0, 20.0, 6.0, -4.0}, 
					        {-3.0, 0.0, 6.0, 0.0, -3.0}, 
					        {-1.0, -3.0, -4.0, -3.0, -1.0}};
					  
double laplacian_7[7][7] = {{-2.0, -3.0, -4.0, -6.0, -4.0, -3.0, -2.0}, 
                            {-3.0, -5.0, -4.0, -3.0, -4.0, -5.0, -3.0},
                            {-4.0, -4.0, 9.0, 20.0, 9.0, -4.0, -4.0},
                            {-6.0, -3.0, 20.0, 36.0, 20.0, -3.0, -6.0},
                            {-4.0, -4.0, 9.0, 20.0, 9.0, -4.0, -4.0},
                            {-3.0, -5.0, -4.0, -3.0, -4.0, -5.0, -3.0},
                            {-2.0, -3.0, -4.0, -6.0, -4.0, -3.0, -2.0}};

double laplacian_9[9][9] = {{0.0, 0.0, 3.0, 2.0, 2.0, 2.0, 3.0, 0.0, 0.0},
							{0.0, 2.0, 3.0, 5.0, 5.0, 5.0, 3.0, 2.0, 0.0},
							{3.0, 3.0, 5.0, 3.0, 0.0, 3.0, 5.0, 3.0, 3.0},
							{2.0, 5.0, 3.0, -12.0, -23.0, -12.0, 3.0, 5.0, 2.0},
							{2.0, 5.0, 0.0, -23.0, -40.0, -23.0, 0.0, 5.0, 2.0},
							{2.0, 5.0, 3.0, -12.0, -23.0, -12.0, 3.0, 5.0, 2.0},
							{3.0, 3.0, 5.0, 3.0, 0.0, 3.0, 5.0, 3.0, 3.0},
							{0.0, 2.0, 3.0, 5.0, 5.0, 5.0, 3.0, 2.0, 0.0},
							{0.0, 0.0, 3.0, 2.0, 2.0, 2.0, 3.0, 0.0, 0.0}};

//Sobel diagnol filter coffecient
double sobel_diagnol3[3][3] = {{-1.0, -0.5, 0.0},
							  {-0.5,  0.0, 0.5},
							  { 0.0,  0.5, 1.0}}; 

double sobel_diagnol5[5][5] = {{-0.50, -0.4, -0.25, -0.2, 0.0}, 
					           {-0.40, -1.0, -0.50, 0.0, 0.2}, 
					           {-0.25, -0.5,  0.00, 0.5, 0.25}, 
					           {-0.20,  0.0,  0.50, 1.0, 0.4}, 
					           { 0.00,  0.2,  0.25, 0.4, 0.5}};

double sobel_diagnol7[7][7] = {{-0.333333, -0.3, -0.230769, -0.166667, -0.153846, -0.1, 0.0}, 
					        	{-0.3, -0.50, -0.4, -0.25, -0.2, 0, 0.1},
					        	{-0.230769, -0.40, -1, -0.5, 0.0, 0.2, 0.153846},
					        	{-0.166667, -0.25, -0.5, 0, 0.5, 0.25, 0.166667},
					        	{-0.153846, -0.2, 0.0, 0.5, 1, 0.4, 0.230769 },
					        	{-0.1, 0.0, 0.2, 0.25, 0.4, 0.5, 0.3},
					        	{0, 0.1, 0.153846, 0.166667, 0.230769, 0.3, 0.333333}};

double sobel_diagnol9[9][9] = {{-0.25,-0.235294,-0.2,-0.16,-0.125,-0.12,-0.1,-0.0588235,0},
								{-0.235294,-0.333333, -0.3, -0.230769, -0.166667, -0.153846, -0.1, 0.0,0.1}, 
					        	{-0.2,-0.3, -0.50, -0.4, -0.25, -0.2, 0, 0.1,0.1},
					        	{-0.16,-0.230769, -0.40, -1, -0.5, 0.0, 0.2, 0.153846,0.12},
					        	{-0.125,-0.166667, -0.25, -0.5, 0, 0.5, 0.25, 0.166667,0.125},
					        	{-0.12,-0.153846, -0.2, 0.0, 0.5, 1, 0.4, 0.230769, 0.16},
					        	{-0.1,-0.1, 0.0, 0.2, 0.25, 0.4, 0.5, 0.3,0.2},
					        	{-0.0588235,0, 0.1, 0.153846, 0.166667, 0.230769, 0.3, 0.333333,0.235294},
								{0,0.0588235,0.1,0.12,0.125,0.16,0.2,0.235294,0.25}};


//This function simply return sobel diagnol filter coefficient
double sobel_diagnol_filter_coff(int f_size, int x, int y)
{
	if(f_size ==3)
		return sobel_diagnol3[x][y];
	else if(f_size==5)
		return sobel_diagnol5[x][y];
	else if(f_size == 7)
		return sobel_diagnol7[x][y];
	else
		return sobel_diagnol9[x][y];
}

//This function simply return log filter coefficient
double log_filter_coff(int f_size, int x, int y)
{
	if(f_size ==3)
		return log_3[x][y];
	else if(f_size==5)
		return log_5[x][y];
	else if(f_size == 7)
		return log_7[x][y];
	else
		return log_9[x][y]/180.0;
}

//This function simply return laplacian filter coefficient
double laplacian_filter_coff(int f_size, int x, int y)
{
	if(f_size ==3)
		return laplacian_3[x][y]/8.0;
	else if(f_size==5)
		return laplacian_5[x][y]/44.0;
	else if(f_size == 7)
		return laplacian_7[x][y]/152.0;
	else
		return laplacian_9[x][y]/180.0;
}
//This function simply return Gaussian filter coefficient
double gaussian_filter_coff(int f_size, int x, int y)
{
	//Sigma is set to be filter_size/4 
	double sigma = f_size/4.0;
	int mean = f_size/2;
	return exp(-0.5*(pow((x-mean)/sigma, 2.0) + pow((y-mean)/sigma, 2.0)));
}
// double log_filter_coff(int f_size, int x, int y)
// {
// 	double sigma = 1.4;
// 	int mean = f_size/2;
// 	return  (pow(x-mean, 2.0) + pow(y-mean, 2.0) - 2*sigma*sigma)*exp(-0.5*(pow((x-mean)/sigma, 2.0) + pow((y-mean)/sigma, 2.0)));
// }

//This function simply return sobel filter coefficient for horizontal and vertical sobel filter
//Depending on axis
double sobel_filter_coff(int size, int i, int j, int axis)
{
	if(i==0 && j==0)
        return 0.0;

    return (axis==0)?i/double(i*i+j*j):j/double(i*i+j*j);
}

//This function simply return the img intensity value if i,j is in image range else it return 0;
//This function act as our 0 padder. 
int pixel_value(int i, int j)
{
	int val = 0;
	if((i>=0 && i<input_img.rows) && (j>=0 && j<input_img.cols))
		val = (int)input_img.at<uchar>(Point(i,j));

	return val;
}
//This function takes the filter size and apply mean filter 
Mat mean_filter(int f_size)
{
	Mat out_img(input_img.rows, input_img.cols , CV_8UC1, Scalar(0));
	int sum = 0;
	for(int i=0; i<input_img.rows; i++)
		for(int j=0; j<input_img.cols; j++)
		{
			sum = 0;
			for(int k=-1*((f_size-1)/2); k<=((f_size-1)/2); k++)
				for(int l=-1*((f_size-1)/2); l<=((f_size-1)/2); l++)
					sum+= pixel_value(i+k, j+l);
			out_img.at<uchar>(Point(i,j)) = int(sum/(f_size*f_size));
		}
	return out_img;
}
//This function takes the filter size and apply median filter 
Mat median_filter(int f_size)
{
	Mat out_img(input_img.rows, input_img.cols , CV_8UC1, Scalar(0));
	for(int i=0; i<input_img.rows; i++)
		for(int j=0; j<input_img.cols; j++)
		{
			vector<int> ar;
			for(int k=-1*((f_size-1)/2); k<=((f_size-1)/2); k++)
				for(int l=-1*((f_size-1)/2); l<=((f_size-1)/2); l++)
					ar.push_back(pixel_value(i+k, j+l));
			sort(ar.begin(), ar.end());
			out_img.at<uchar>(Point(i,j)) = ar[(f_size*f_size-1)/2];
		}
	return out_img;
}
//This function takes the filter size and apply Prewitt filter 
//It calculates edge in both horizontal and vertical direction and then take root square value of them as final value
Mat prewitt_filter(int f_size)
{
	Mat out_img(input_img.rows, input_img.cols , CV_8UC1, Scalar(0));
	int norm_factor = 0;
	for(int i=1; i<=(f_size-1)/2; i++)
		norm_factor+= i*f_size;

	for(int i=0; i<input_img.rows; i++)
		for(int j=0; j<input_img.cols; j++)
		{
			int Gx=0, Gy=0;
			for(int k=-1*((f_size-1)/2); k<=((f_size-1)/2); k++)
				for(int l=-1*((f_size-1)/2); l<=((f_size-1)/2); l++)
				{
					Gx += -1*l*pixel_value(i+k, j+l);
					Gy += -1*k*pixel_value(i+k, j+l);
				}
			Gx = Gx/norm_factor;
			Gy = Gy/norm_factor;
			out_img.at<uchar>(Point(i,j)) = int(sqrt(Gx*Gx + Gy*Gy));
		}
	return out_img;
}
//This function takes the filter size and apply Gaussian filter 
Mat Gaussian_filter(int f_size)
{
	Mat out_img(input_img.rows, input_img.cols , CV_8UC1, Scalar(0));
	for(int i=0; i<input_img.rows; i++)
		for(int j=0; j<input_img.cols; j++)
		{
			double sum =0;
			double norm_fact = 0;
			double temp;
			for(int k=-1*((f_size-1)/2); k<=((f_size-1)/2); k++)
				for(int l=-1*((f_size-1)/2); l<=((f_size-1)/2); l++)
				{
					temp = gaussian_filter_coff(f_size, k+(f_size-1)/2, l+(f_size-1)/2);
					sum+= temp*pixel_value(i+k, j+l);
					norm_fact+=temp;
				}
			out_img.at<uchar>(Point(i,j)) = int(sum/norm_fact);
		}
	return out_img;
}
//This function takes the filter size and apply sobel horizonal and vertical filter
//if axis ==0, then vertical else horizontal
Mat Sobel_filter(int f_size, int axis)
{
	Mat out_img(input_img.rows, input_img.cols , CV_8UC1, Scalar(0));
	for(int i=0; i<input_img.rows; i++)
		for(int j=0; j<input_img.cols; j++)
		{
			double sum =0;
			double norm_fact = 0;
			double temp;
			for(int k=-1*((f_size-1)/2); k<=((f_size-1)/2); k++)
				for(int l=-1*((f_size-1)/2); l<=((f_size-1)/2); l++)
				{
					temp = sobel_filter_coff(f_size, k, l, axis);
					sum += temp*pixel_value(i+k, j+l);
					if(temp>0)
					norm_fact+=temp;
				}
			out_img.at<uchar>(Point(i,j)) = abs(int(sum/norm_fact));
		}
	return out_img;
}
//This function takes the filter size and apply sobel diagnol filter
//Using the sobel diagnol coffecients defined above
Mat Sobel_diagnol_filter(int f_size)
{
	Mat out_img(input_img.rows, input_img.cols , CV_8UC1, Scalar(0));
	for(int i=0; i<input_img.rows; i++)
		for(int j=0; j<input_img.cols; j++)
		{
			double sum_x =0, sum_y=0;
			double temp_x, temp_y;
			double norm_fact_x = 0, norm_fact_y=0;
			for(int k=-1*((f_size-1)/2); k<=((f_size-1)/2); k++)
				for(int l=-1*((f_size-1)/2); l<=((f_size-1)/2); l++)
				{
					temp_x = sobel_diagnol_filter_coff(f_size, k+(f_size-1)/2, l+(f_size-1)/2);
					temp_y = sobel_diagnol_filter_coff(f_size, k+(f_size-1)/2, (f_size-1)/2-l);
					sum_x += temp_x*pixel_value(i+k, j+l);
					sum_y += temp_y*pixel_value(i+k, j+l);
					if(temp_x>0)
						norm_fact_x +=temp_x;
					if(temp_y >0)
						norm_fact_y +=temp_y;
				}
			out_img.at<uchar>(Point(i,j)) = int(sqrt((sum_x/norm_fact_x)*(sum_x/norm_fact_x) + (sum_y/norm_fact_y)*(sum_y/norm_fact_y)));
		}
	return out_img;
}
//This function takes the filter size and apply log filter
//Using the log coffecients defined above
Mat log_filter(int f_size)
{
	Mat out_img(input_img.rows, input_img.cols , CV_8UC1, Scalar(0));
	for(int i=0; i<input_img.rows; i++)
		for(int j=0; j<input_img.cols; j++)
		{
			double sum =0;
			double temp;
			for(int k=-1*((f_size-1)/2); k<=((f_size-1)/2); k++)
				for(int l=-1*((f_size-1)/2); l<=((f_size-1)/2); l++)
				{
					temp = log_filter_coff(f_size, k+(f_size-1)/2, l+(f_size-1)/2);
					sum+= temp*pixel_value(i+k, j+l);
				}
			out_img.at<uchar>(Point(i,j)) = abs(int(sum));
		}
	return out_img;
}
//This function takes the filter size and apply laplacian filter
//Using the laplacian coffecients defined above
Mat laplacian_filter(int f_size)
{
	Mat out_img(input_img.rows, input_img.cols , CV_8UC1, Scalar(0));
	for(int i=0; i<input_img.rows; i++)
		for(int j=0; j<input_img.cols; j++)
		{
			double sum =0;
			double temp;
			for(int k=-1*((f_size-1)/2); k<=((f_size-1)/2); k++)
				for(int l=-1*((f_size-1)/2); l<=((f_size-1)/2); l++)
				{
					temp = laplacian_filter_coff(f_size, k+(f_size-1)/2, l+(f_size-1)/2);
					sum+= temp*pixel_value(i+k, j+l);
				}
			out_img.at<uchar>(Point(i,j)) = abs(int(sum));
		}
	return out_img;
}
//Function to read images from the folder
vector<string> read_images(string path)
{
	vector<string> imgs;
	string img_name;
	string img_path_noisy = path + "/" + noise_folder_name;
	string img_path_normal = path + "/" + normal_folder_name;
	const char* path_noise = &img_path_noisy[0];
	const char* path_normal = &img_path_normal[0];
	DIR *dir = opendir(path_noise);
	struct dirent *dp;
	//Make vector of images name placed in noise_folder_name
	if(dir != NULL)
	{
		while(( dp = readdir(dir)) != NULL)
		{
			img_name = dp->d_name;
			//check for garbage value
			if(img_name.length() >=4)
				imgs.push_back(img_path_noisy + "/" + img_name);
		}
	}
	(void)closedir(dir);
	dir = opendir(path_normal);
	//Make vector of images name placed in normal_folder_name
	if(dir != NULL)
	{
		while(( dp = readdir(dir)) != NULL)
		{
			img_name = dp->d_name;
			if(img_name.length() >=4)
				imgs.push_back(img_path_normal + "/" + img_name);
		}
	}
	(void)closedir(dir);

	return imgs;

}
//Callback function for filtering
static void Filter_image(int, void*)
{

	//First calculate filter size
	int f_size = 3 + (filter_size-1)*2;
	Mat filter_image;
	//If filter size is one or no filter is selected show the original image
	if(f_size ==1 || filter_number ==0)
	{
		imshow("Images", input_img);
	}
	else
	{
		//else based on filter selected and filter size, filter the image and display it
		if(filter_number==1)
		{
			filter_image = mean_filter(f_size);
			imshow("Images", filter_image);
		}
		else if(filter_number==2)
		{
			filter_image = median_filter(f_size);
			imshow("Images", filter_image);
		}
		else if(filter_number ==3)
		{
			filter_image = prewitt_filter(f_size);
			imshow("Images", filter_image);
		}
		else if(filter_number == 4)
		{
			filter_image = laplacian_filter(f_size);
			imshow("Images", filter_image);
		}
		//filter_number=5, is for sobel horizontal filter and 6 is for vertical
		else if(filter_number == 5 || filter_number == 6)
		{
			filter_image = Sobel_filter(f_size, filter_number-5);
			imshow("Images", filter_image);
		}
		else if(filter_number == 7)
		{
			filter_image = Sobel_diagnol_filter(f_size);
			imshow("Images", filter_image);
		}
		else if(filter_number == 8)
		{
			filter_image = Gaussian_filter(f_size);
			imshow("Images", filter_image);
		}
		else if(filter_number==9)
		{
			filter_image = log_filter(f_size);
			imshow("Images", filter_image);
		}
		else
			imshow("Images", input_img);
	}
}
//Callback function to select image, it simply read the image and calls for filter operation
static void Select_img(int, void*)
{
	input_img = imread(imgs_name[img_number], IMREAD_UNCHANGED);
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
		int max_images = imgs_name.size();

		//Create name window
		namedWindow("Images", CV_WINDOW_AUTOSIZE);

		//Create the required trackbars		
		createTrackbar("Select images", "Images", &img_number, max_images-1, Select_img);
		createTrackbar("Select Filter", "Images", &filter_number, 9, Filter_image);
		createTrackbar("Filter Size", "Images", &filter_size, ((max_filter_size-3)/2)+1, Filter_image);
		//Call one of the call back function for initialization 
		Select_img(0, 0);
    	waitKey();
	}
	return 0;
}