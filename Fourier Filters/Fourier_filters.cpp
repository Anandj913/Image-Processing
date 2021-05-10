#include <bits/stdc++.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
//Library to read images
#include <dirent.h>
using namespace cv;
using namespace std;

const double e = 2.718281;
const complex<double> iota(0,1);
const complex<double> zero(0.0, 0.0);

//vector to store image names
vector<string> imgs_name;
int img_number = 0;
int cut_off = 0;
int filter_number = 0;
int max_filter_number = 6; 
//vector to store read image in complex form
vector<vector< complex<double> > > comp_img;
//vector to store fft of image
vector<vector< complex<double> > > img_fft;
Mat input_img;
//image to store magnitude and shift magnitude spectrum
Mat magnitude_spectrum;
Mat shift_magnitude_spectrum;
//vector to store the shift magnitude value
vector<vector<double> > shift_abs_value;
//image to store filter spectrum
Mat filter_coeff;

//function to return idea low_pass_coefficient
double ilpf(int i, int j, double d0)
{
  if(sqrt(pow((i-magnitude_spectrum.rows/2),2)+pow((j-magnitude_spectrum.cols/2),2))<=d0)
  {
    return 1;
  }
  else
  {
    return 0;
  }
}

//function to return gaussian low_pass_coefficient
double glpf(int i, int j, double d0)
{
  return exp((-1*(pow((i-magnitude_spectrum.rows/2),2)+pow((j-magnitude_spectrum.cols/2),2)))/(2*pow(d0,2)));
}

//function to return butterworth low_pass_coefficient
double blpf(int i, int j, double d0, int order)
{
	double D = sqrt(pow((i-magnitude_spectrum.rows/2),2)+pow((j-magnitude_spectrum.cols/2),2));
	D /=d0;
	return 1.0/(1.0 + pow(D, 2*order));
}

//This function simply performs the inverse shift operation on the magnitude of the filtered spectrum so that it can be used in finding IFFT.
//we used circular shift for doing so
vector<vector<double> > shift_filter_mag(vector<vector<double> > filter_mag)
{
	int m = magnitude_spectrum.rows;
	int n = magnitude_spectrum.cols;
	int mm = floor(double(m)/ 2);
	int nn = floor(double(n)/ 2);
	vector<vector<double> > temp(m, vector<double>(n, 0.0));
	for(int i=0; i<m; i++)
	{
		int ii = (i+mm)%m;
		for(int j=0; j<n; j++)
		{
			int jj = (j+nn)%n;
			temp[ii][jj] = filter_mag[i][j];
		}
	}
	return temp;
}

//Once we have the filtered magnitude we need to combine that with original argument of fft image 
//to form the final complex filtered image which will be used to find IFFT
//This function does the same
vector<vector<complex<double> > > combine_mag_argu(vector<vector<double> > filter_mag)
{
	vector<vector<complex<double> > > temp(input_img.rows, vector<complex<double> >(input_img.cols));
	for(int i=0; i<input_img.rows; i++)
		for(int j=0; j<input_img.cols; j++)
		{
			complex<double> temp_c(filter_mag[i][j]*cos(arg(img_fft[i][j])), filter_mag[i][j]*sin(arg(img_fft[i][j])));
			temp[i][j] = temp_c;
		}
	return temp;
}

//This function simply converts the complex image to Mat form for dispaly, 
//by making the magnitude value as final pixel value
Mat form_output_img(vector<vector<complex<double> > > filtered_value)
{
	Mat out_img = input_img.clone();
	for(int i=0; i<input_img.rows; i++)
		for(int j=0; j<input_img.cols; j++)
			out_img.at<uchar>(Point(i,j)) = saturate_cast<uchar>(abs(filtered_value[i][j]));
	return out_img;
}

//This function finds the next power of two number of given a number
//It is used by fft function as it requires H and W to be in power of 2 for fft algo to work 
int find_next_power_of_two(int x)
{
	double temp = log2(x);
	temp = ceil(temp);
	return int(pow(2, temp));
}

//This function simply returns the omega used in finding fft for both fft and ifft 
//based on value of boolean inv
complex<double> omega(double sample, bool inv)
{
	if(!inv)
		return pow(e, (-2.0*M_PI*iota)/sample);
	else
		return pow(e, (2.0*M_PI*iota)/sample);
}

//Recurrisive function to find fft of given 1D array of complex number using given Wn
vector<complex<double> > fft_rec(vector<complex<double> > a, const complex<double> wn)
{
	//base condition
	if(a.size() ==1)
		return a;

	//vector to store even and odd component of array
	vector< complex<double> > even(a.size()/2);
	vector< complex<double> > odd(a.size()/2);

	//seprating even and odd position component
	for(unsigned i=0; i<a.size(); i++)
	{
		if(i%2 == 0)
		{
			even[i/2]= a[i];
		}
		else
		{
			odd[(i-1)/2] = a[i];
		}
	}

	//recurrively finding fft for even and odd component with wn = wn*wn
	vector< complex<double> > even_fft = fft_rec(even, wn*wn);
	vector< complex<double> > odd_fft = fft_rec(odd, wn*wn);

	//Combining the above result to give final fft
	vector< complex<double> > final_fft(a.size());
	complex<double> w(1,0);
	for(unsigned i=0; i<a.size()/2; i++)
	{
		final_fft[i] = even_fft[i] + (w*odd_fft[i]);
		final_fft[i+ (a.size()/2)] =  even_fft[i] - (w*odd_fft[i]);
		w *= wn;
	}

	return final_fft;
}

//This function takes image in complex number form and a boolean value inverted
//performs the FFT on the given input if boolean is false else performs IFFT and returns the final result in complex form.
vector<vector<complex<double> > > find_img_fft(vector<vector<complex<double> > > img, bool inverse)
{
	int H = img.size();
	int W = img[0].size();

	//finding new h and w which is in power of 2 for fft also
	int H_2 = find_next_power_of_two(H);
	int W_2 = find_next_power_of_two(W);

	//vector to store column wise fft and final fft
	vector<vector< complex<double> > > half_fft(H_2, vector< complex<double> >(W));
	vector<vector< complex<double> > > fft(H, vector< complex<double> >(W));

	//we pad the column with 0 if its not in power of 2
	for(int i=0; i<W; i++)
	{
		vector< complex<double> > temp;
		for(int j=0; j<H_2; j++)
		{
			if(j<H)
				temp.push_back(img[j][i]);
			else
				temp.push_back(zero);

		}
		//find column wise fft of each column
		temp = fft_rec(temp, omega(H_2, inverse));

		for(int j=0; j<H_2; j++)
			half_fft[j][i] = temp[j];
	}

	//we pad the row with 0 if its not in power of 2, of the column wise fft output
	for(int i=0; i<H; i++)
	{
		vector< complex<double> > temp;
		for(int j=0; j<W_2; j++)
		{
			if(j<W)
				temp.push_back(half_fft[i][j]);
			else
				temp.push_back(zero);

		}
		//find row wise fft of each row of column wise fft output to find final fft
		temp = fft_rec(temp, omega(W_2, inverse));

		for(int j=0; j<W; j++)
		{
			if(!inverse)
				fft[i][j] = temp[j]/sqrt(H*W);
			else
				fft[i][j] = temp[j]/sqrt(H*W);
		}
	}

	return fft;
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
				if(temp == "jpg")
				imgs.push_back(img_path + "/" + img_name);
			}
		}
	}
	(void)closedir(dir);

	return imgs;

}

//This function performs the ideal low pass filter with given filter cutoff
Mat ideal_low_filter(double d0)
{
    vector<vector<double> > filter_mag(input_img.rows,vector<double>(input_img.cols,0));
    vector<vector<complex<double> > > filter_complex(input_img.rows, vector<complex<double> >(input_img.cols));
    filter_coeff = input_img.clone();
    double coeff;
    Mat out_img;
    double max_coeff = -100000000;
    //finding normalizing factor for filter coefficient
    for(int i=0; i< input_img.rows; i++)
		for(int j=0; j< input_img.cols; j++)
		{
			if(ilpf(i,j,d0) > max_coeff)
				max_coeff = ilpf(i,j,d0);
		}

	//Finding filtered magnitude and fiter spectrum
	for(int i=0; i<input_img.rows; i++)
		for(int j=0; j<input_img.cols; j++)
		{
            coeff = ilpf(i,j,d0)/max_coeff;
            filter_coeff.at<uchar>(Point(i,j)) = saturate_cast<uchar>(255*coeff);
            filter_mag[i][j] = (coeff)*shift_abs_value[i][j];
		}
	//finding inverse shifted spectrum
	filter_mag = shift_filter_mag(filter_mag);
	//finding final complex filtered image by combining filtered magnitude with argument 
	filter_complex = combine_mag_argu(filter_mag);
	//finding IFFT
	filter_complex = find_img_fft(filter_complex, true);
	//Forming final filtered image
	out_img = form_output_img(filter_complex);

	return out_img;
}

//This function performs the ideal high pass filter with given filter cutoff
Mat ideal_high_filter(double d0)
{
    vector<vector<double> > filter_mag(input_img.rows,vector<double>(input_img.cols,0));
    vector<vector<complex<double> > > filter_complex(input_img.rows, vector<complex<double> >(input_img.cols));
    filter_coeff = input_img.clone();
    double coeff;
    Mat out_img;
    double max_coeff = -100000000;
    //finding normalizing factor for filter coefficient
    for(int i=0; i<input_img.rows; i++)
		for(int j=0; j<input_img.cols; j++)
		{
			if(ilpf(i,j,d0) > max_coeff)
				max_coeff = ilpf(i,j,d0);
		}

    //Finding filtered magnitude and fiter spectrum
	for(int i=0; i<input_img.rows; i++)
		for(int j=0; j<input_img.cols; j++)
		{
            coeff = 1.0 - (ilpf(i,j,d0)/max_coeff); // 1-normalized_low_pass_filter
            filter_coeff.at<uchar>(Point(i,j)) = saturate_cast<uchar>(255*coeff);
            filter_mag[i][j] = coeff*shift_abs_value[i][j];
		}
	//finding inverse shifted spectrum	
	filter_mag = shift_filter_mag(filter_mag);
	//finding final complex filtered image by combining filtered magnitude with argument 
	filter_complex = combine_mag_argu(filter_mag);
	//finding IFFT
	filter_complex = find_img_fft(filter_complex, true);
	//Forming final filtered image
	out_img = form_output_img(filter_complex);
	return out_img;
}
//This function performs the gaussian low pass filter with given filter cutoff
Mat Gaussian_LPF(double d0)
{
	vector<vector<double> > filter_mag(input_img.rows,vector<double>(input_img.cols,0));
    vector<vector<complex<double> > > filter_complex(input_img.rows, vector<complex<double> >(input_img.cols));
    filter_coeff = input_img.clone();
    double coeff;
    Mat out_img;
    double max_coeff = -100000000;
    //finding normalizing factor for filter coefficient
    for(int i=0; i<input_img.rows; i++)
		for(int j=0; j<input_img.cols; j++)
		{
			if(ilpf(i,j,d0) > max_coeff)
				max_coeff = glpf(i,j,d0);
		}
	//Finding filtered magnitude and fiter spectrum
	for(int i=0; i<input_img.rows; i++)
		for(int j=0; j<input_img.cols; j++)
		{
            coeff = (glpf(i,j,d0)/max_coeff);
            filter_coeff.at<uchar>(Point(i,j)) = saturate_cast<uchar>(255*coeff);
            filter_mag[i][j] = coeff*shift_abs_value[i][j];
		}
	//finding inverse shifted spectrum	
	filter_mag = shift_filter_mag(filter_mag);
	//finding final complex filtered image by combining filtered magnitude with argument 
	filter_complex = combine_mag_argu(filter_mag);
	//finding IFFT
	filter_complex = find_img_fft(filter_complex, true);
	//Forming final filtered image
	out_img = form_output_img(filter_complex);
	return out_img;
}

//This function performs the gaussian high pass filter with given filter cutoff
Mat Gaussian_HPF(double d0)
{
	vector<vector<double> > filter_mag(input_img.rows,vector<double>(input_img.cols,0));
    vector<vector<complex<double> > > filter_complex(input_img.rows, vector<complex<double> >(input_img.cols));
    filter_coeff = input_img.clone();
    double coeff;
    Mat out_img;
    double max_coeff = -100000000;
     //finding normalizing factor for filter coefficient
    for(int i=0; i<input_img.rows; i++)
		for(int j=0; j<input_img.cols; j++)
		{
			if(ilpf(i,j,d0) > max_coeff)
				max_coeff = glpf(i,j,d0);
		}
	//Finding filtered magnitude and fiter spectrum
	for(int i=0; i<input_img.rows; i++)
		for(int j=0; j<input_img.cols; j++)
		{
            coeff = 1.0 - (glpf(i,j,d0)/max_coeff); // 1-normalized_gaussian_low_pass_filter
            filter_coeff.at<uchar>(Point(i,j)) = saturate_cast<uchar>(255*coeff);
            filter_mag[i][j] = coeff*shift_abs_value[i][j];
		}
	//finding inverse shifted spectrum	
	filter_mag = shift_filter_mag(filter_mag);
	//finding final complex filtered image by combining filtered magnitude with argument 
	filter_complex = combine_mag_argu(filter_mag);
	//finding IFFT
	filter_complex = find_img_fft(filter_complex, true);
	//Forming final filtered image
	out_img = form_output_img(filter_complex);
	return out_img;
}

//This function performs the Butterworth low pass filter with given filter cutoff and order
Mat Butterworth_LPF(double d0, int order)
{
	vector<vector<double> > filter_mag(input_img.rows,vector<double>(input_img.cols,0));
    vector<vector<complex<double> > > filter_complex(input_img.rows, vector<complex<double> >(input_img.cols));
    filter_coeff = input_img.clone();
    double coeff;
    Mat out_img;
    double max_coeff = -100000000;
     //finding normalizing factor for filter coefficient
    for(int i=0; i<input_img.rows; i++)
		for(int j=0; j<input_img.cols; j++)
		{
			if(ilpf(i,j,d0) > max_coeff)
				max_coeff = blpf(i,j,d0, order);
		}
	//Finding filtered magnitude and fiter spectrum
	for(int i=0; i<input_img.rows; i++)
		for(int j=0; j<input_img.cols; j++)
		{
            coeff = (blpf(i,j,d0, order)/max_coeff);
            filter_coeff.at<uchar>(Point(i,j)) = saturate_cast<uchar>(255*coeff);
            filter_mag[i][j] = coeff*shift_abs_value[i][j];
		}
	//finding inverse shifted spectrum	
	filter_mag = shift_filter_mag(filter_mag);
	//finding final complex filtered image by combining filtered magnitude with argument 
	filter_complex = combine_mag_argu(filter_mag);
	//finding IFFT
	filter_complex = find_img_fft(filter_complex, true);
	//Forming final filtered image
	out_img = form_output_img(filter_complex);
	return out_img;
}
//This function performs the Butterworth high pass filter with given filter cutoff and order
Mat Butterworth_HPF(double d0, int order)
{
	vector<vector<double> > filter_mag(input_img.rows,vector<double>(input_img.cols,0));
    vector<vector<complex<double> > > filter_complex(input_img.rows, vector<complex<double> >(input_img.cols));
    filter_coeff = input_img.clone();
    double coeff;
    Mat out_img;
    double max_coeff = -100000000;
     //finding normalizing factor for filter coefficient
    for(int i=0; i<input_img.rows; i++)
		for(int j=0; j<input_img.cols; j++)
		{
			if(ilpf(i,j,d0) > max_coeff)
				max_coeff = blpf(i,j,d0, order);
		}
	//Finding filtered magnitude and fiter spectrum
	for(int i=0; i<input_img.rows; i++)
		for(int j=0; j<input_img.cols; j++)
		{
            coeff = 1.0 - (blpf(i,j,d0, order)/max_coeff); // 1-normalized_butterworth_low_pass_filter
            filter_coeff.at<uchar>(Point(i,j)) = saturate_cast<uchar>(255*coeff);
            filter_mag[i][j] = coeff*shift_abs_value[i][j];
		}
	//finding inverse shifted spectrum	
	filter_mag = shift_filter_mag(filter_mag);
	//finding final complex filtered image by combining filtered magnitude with argument 
	filter_complex = combine_mag_argu(filter_mag);
	//finding IFFT
	filter_complex = find_img_fft(filter_complex, true);
	//Forming final filtered image
	out_img = form_output_img(filter_complex);
	return out_img;
}

//Callback function for filtering
static void Filter_image(int, void*)
{

	Mat filter_image;
	//First calculate filter cutoff
	double f_cutoff = cut_off*10.0;

	//If filter cutoff is zero or no filter is selected show the original image
	if(f_cutoff == 0.0 || filter_number ==0)
	{
		imshow("Images", input_img);
		imshow("Magnitude Spectrum", shift_magnitude_spectrum);
		imshow("Filter Spectrum", input_img);
		imshow("Filter Output", input_img);
	}
	else
	{
		//else based on filter selected and filter cutoff, filter the image and display it
		if(filter_number==1)
		{
			filter_image = ideal_low_filter(f_cutoff);
			imshow("Images", input_img);
			imshow("Magnitude Spectrum", shift_magnitude_spectrum);
			imshow("Filter Spectrum", filter_coeff);
			imshow("Filter Output", filter_image);

		}
		else if(filter_number==2)
		{
			filter_image = ideal_high_filter(f_cutoff);
			imshow("Images", input_img);
			imshow("Magnitude Spectrum", shift_magnitude_spectrum);
			imshow("Filter Spectrum", filter_coeff);
			imshow("Filter Output", filter_image);
		}
		else if(filter_number ==3)
		{
			filter_image = Gaussian_LPF(f_cutoff);
			imshow("Images", input_img);
			imshow("Magnitude Spectrum", shift_magnitude_spectrum);
			imshow("Filter Spectrum", filter_coeff);
			imshow("Filter Output", filter_image);
		}
		else if(filter_number == 4)
		{
			filter_image = Gaussian_HPF(f_cutoff);
			imshow("Images", input_img);
			imshow("Magnitude Spectrum", shift_magnitude_spectrum);
			imshow("Filter Spectrum", filter_coeff);
			imshow("Filter Output", filter_image);
		}
		else if(filter_number == 5)
		{
			filter_image = Butterworth_LPF(f_cutoff, 2);
			imshow("Images", input_img);
			imshow("Magnitude Spectrum", shift_magnitude_spectrum);
			imshow("Filter Spectrum", filter_coeff);
			imshow("Filter Output", filter_image);
		}
		else if(filter_number == 6)
		{
			filter_image = Butterworth_HPF(f_cutoff, 2);
			imshow("Images", input_img);
			imshow("Magnitude Spectrum", shift_magnitude_spectrum);
			imshow("Filter Spectrum", filter_coeff);
			imshow("Filter Output", filter_image);
		}
		else
		{
			imshow("Images", input_img);
			imshow("Magnitude Spectrum", shift_magnitude_spectrum);
			imshow("Filter Spectrum", input_img);
			imshow("Filter Output", input_img);
		}
	}
}
//This function simply performs the magnitude shift of fft using circular shift
vector<vector<double> > mag_shift()
{
	int m = magnitude_spectrum.rows;
	int n = magnitude_spectrum.cols;
	int mm = floor(double(m)/ 2);
	int nn = floor(double(n)/ 2);
	vector<vector<double> > temp(m, vector<double>(n, 0.0));
	for(int i=0; i<m; i++)
	{
		int ii = (i+mm)%m;
		for(int j=0; j<n; j++)
		{
			int jj = (j+nn)%n;
			shift_magnitude_spectrum.at<uchar>(Point(ii,jj)) = magnitude_spectrum.at<uchar>(Point(i,j));
			temp[ii][jj] = abs(img_fft[i][j]);
		}
	}
	return temp;
}
//This function find the magnitude of fft for display
void magnitude_cal()
{
	for(int i=0; i<input_img.rows; i++)
		for(int j=0; j<input_img.cols; j++)
		{
			magnitude_spectrum.at<uchar>(Point(i,j)) = saturate_cast<uchar>(abs(img_fft[i][j]));
		}

	shift_abs_value = mag_shift();
}
//Callback function to select image, it simply read the image and calls for filter operation
static void Select_img(int, void*)
{
	input_img = imread(imgs_name[img_number], IMREAD_UNCHANGED);
	magnitude_spectrum = input_img.clone();
	shift_magnitude_spectrum = input_img.clone();
	comp_img.clear();
	for(unsigned i=0; i<input_img.rows; i++)
	{
		vector< complex<double> > temp;

		for(unsigned j=0; j<input_img.cols; j++)
		{
			complex<double> temp_c(((int)input_img.at<uchar>(Point(i,j))), 0.0);
			temp.push_back(temp_c);
		}

		comp_img.push_back(temp);
	}

	img_fft = find_img_fft(comp_img, false);
	magnitude_cal();
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
		namedWindow("Magnitude Spectrum", CV_WINDOW_AUTOSIZE);
		namedWindow("Filter Spectrum", CV_WINDOW_AUTOSIZE);
		namedWindow("Filter Output", CV_WINDOW_AUTOSIZE);

		//Create the required trackbars		
		createTrackbar("Select images", "Images", &img_number, max_images-1, Select_img);
		createTrackbar("Select Filter", "Images", &filter_number, max_filter_number, Filter_image);
		createTrackbar("Filter Cutoff", "Images", &cut_off, 12 , Filter_image);
		//Call one of the call back function for initialization 
		Select_img(0, 0);
    	waitKey();
	}
	return 0;
	
}