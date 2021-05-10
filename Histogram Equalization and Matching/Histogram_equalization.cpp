
#include <bits/stdc++.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
using namespace cv;
using namespace std;

//provide the name of the input image
char* input_file_name = "walkbridge.jpg"; 

char* output_file_name = "histogram_equalized_img.jpg";
char* input_hist = "input_histogram_img.jpg";
char* output_hist = "equalized_histogarm.jpg";

void histwrite(int histogram[], const char* name)
{
    int hist[256];
    for(int i = 0; i < 256; i++)
    {
        hist[i]=histogram[i];
    }
    // draw the histograms
    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound((double) hist_w/256);
 
    Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(255, 255, 255));
 
    // find the maximum intensity element from histogram
    int max = hist[0];
    for(int i = 1; i < 256; i++){
        if(max < hist[i]){
            max = hist[i];
        }
    }

    // normalize the histogram between 0 and histImage.rows
    for(int i = 0; i < 256; i++)
    {
        hist[i] = ((double)hist[i]/max)*histImage.rows;
    }
 
 
    // draw the intensity line for histogram
    for(int i = 0; i < 256; i++)
    {
        line(histImage, Point(bin_w*(i), hist_h), Point(bin_w*(i), hist_h - hist[i]),Scalar(0,0,0), 1, 8, 0);
    }

    //display and save histogram.
    namedWindow(name, CV_WINDOW_AUTOSIZE);
    imshow(name, histImage);
    imwrite(name, histImage);
}



Mat hist_channel(Mat image)
{
    int hist[256] = {0};
    int new_gray_level[256] = { 0 };

    Mat output_image = image;

    //find the original histogram
    for(int i=0;i<image.rows;i++)
    {
        for(int j=0;j<image.cols;j++)
        {
            hist[(int)image.at<uchar>(Point(j,i))]++;
        }
    }

    //form histogram image to display
    histwrite(hist,input_hist);

    int total = image.rows*image.cols;

    int curr=0;

    //Find new gray level mapping using cumulative normalized histogram
    for(int i=0;i<256;i++)
    {
        curr+=hist[i];
        new_gray_level[i] = round((((float)curr)*255)/total);
    }

    //replace each previous gray level to its new value to form output image.
    for(int i=0;i<image.rows;i++)
    {
        for(int j=0;j<image.cols;j++)
        {
            output_image.at<uchar>(Point(j,i)) = new_gray_level[(int)image.at<uchar>(Point(j,i))];
        }
    }

    //Find new equalized histogram using the found mapping
    int hist_new[256] = {0};

    for(int i=0;i<256;i++)
    {
       hist_new[new_gray_level[i]] += hist[i];
    }

    //form equalized histogram image
    histwrite(hist_new,output_hist);

    //Return the output image
    return output_image;
}


void histogram_equalisation()
{
    //Read image which needs to be equalized
    Mat image = imread(input_file_name, IMREAD_UNCHANGED);
    namedWindow("Input Image", CV_WINDOW_AUTOSIZE);
    imshow("Input Image", image);

    Mat output_image;

    //If read image is colour image
    if(image.channels() == 3)
    {
        //Convert it to YCbCr colour space
        Mat ycrcb;
        cvtColor(image, ycrcb,COLOR_BGR2YCrCb);

        vector<Mat> channels;
        split(ycrcb,channels);

        //Use the Y channel, i.e channel[0] for histogram equalization keeping rest two channel intact
        channels[0] = hist_channel(channels[0]);

        //merge the new channel output to intact channels
        merge(channels,ycrcb);

        //convert back to colour image
        cvtColor(ycrcb,output_image,COLOR_YCrCb2BGR);

    }
    else
    {
        //For grayscale image directly perform histogram equalization
        output_image = hist_channel(image);
    }

    //Display and save histogarm equalized image
    namedWindow("Output Image", CV_WINDOW_AUTOSIZE);
    imshow("Output Image", image);
    imwrite(output_file_name,output_image);

    //wait for user to press ESC to distroy all window and end code
    waitKey();
    return;
}

int main()
{
    histogram_equalisation();   
    return 0;
}