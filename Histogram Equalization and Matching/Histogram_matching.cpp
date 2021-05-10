#include <bits/stdc++.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
using namespace cv;
using namespace std;

int flag = 0;

//user set variables of input and target image
char* input_name = "mandril_color.jpg";
char* target_name = "4.1.02.tiff";
char* result_name = "histogram_matched_mandril_color.jpg";
char* hist_final_name = "Final_matched_histogram_mandril_color.jpg";
char* hist_target_name = "Target_histogram_4.1.02.jpg";
char* hist_input_name = "Input_histogram_mandril_color.jpg";

void histwrite(vector<int> histogram, const char* name)
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
    namedWindow(name, CV_WINDOW_AUTOSIZE);
    imshow(name, histImage);
    imwrite(name, histImage);
}

vector<double> cumm(Mat img)
{
    vector<int> hist(256,0);
    double size = img.rows*img.cols;

    //calculate histogram of given image
    for(int i=0;i<img.rows;i++)
    {
        for(int j=0;j<img.cols;j++)
        {
            hist[(int)img.at<uchar>(Point(i,j))]++;
        }
    }
    if(flag==2)
    {
        //when called 3rd time save matched histogram
        histwrite(hist,hist_final_name);
        flag = 0;
    }
    else if(flag==1)
    {
        //when called second time, save target histogram
        histwrite(hist,hist_target_name);
        flag = 2;
    }
    else
    {
        //when called first time same input histogram
        histwrite(hist,hist_input_name);
        flag=1;
    }

    int curri = 0;

    //find cumulative normalized histogram
    vector<double> nor_hist(256, 0.0);
    for(int i=0;i<256;i++)
    {
        curri+=hist[i];
        hist[i] = curri;
        nor_hist[i] = double(hist[i])/size;
    }


    return nor_hist;

}

Mat hist_match_channel(Mat input_img, Mat target_img)
{
    //find normalized cumulative histogram of input image
    vector<double> G = cumm(input_img);
    //find normalized cumulative histogram of target image
    vector<double> H = cumm(target_img);

    vector<int> M(256,0);

    int prev = 0;
    int j;
    //equate both cumulative histogram to its nearest one to find the gray level mapping
    for(int i=0;i<256;i++)
    {
        double diff = abs(H[prev]-G[i]);
        for(j=prev+1;j<256;j++)
        {          
            if(diff<=abs(H[j]-G[i]))
            {
                diff = abs(H[j]-G[i]);
                prev = j;
            }
            else
            {
                M[i] = prev;
                break;
            }   
        }
        if(j==256)
        {
            M[i] = 255;
        }
    }

    //replace prev graylevel value to its new gray level using mapping found above
    Mat out_img = input_img;
    for(int i=0;i<input_img.rows;i++)
    {
        for(int j=0;j<input_img.cols;j++)
        {
            out_img.at<uchar>(Point(j,i)) = M[(int)input_img.at<uchar>(Point(j,i))];
        }
    }

    //find final matched histogram
    G = cumm(out_img);

    //return final output image
    return out_img;

}

void hist_match()
{
    //read input and target image and display it
    Mat input_img = imread(input_name, IMREAD_UNCHANGED);
    Mat target_img =  imread(target_name,IMREAD_UNCHANGED);

    namedWindow("Input Image", CV_WINDOW_AUTOSIZE);
    imshow("Input Image", input_img);

    namedWindow("Target Image", CV_WINDOW_AUTOSIZE);
    imshow("Target Image", target_img);


    Mat out_img;

    if(target_img.channels()==3)
    {
        //if target is colour use Y channel histogram for matching
        Mat ycrcb_target;
        cvtColor(target_img, ycrcb_target,COLOR_BGR2YCrCb);
        vector<Mat> channels_target;
        split(ycrcb_target,channels_target);

        if(input_img.channels()==3)
        {
            //if input is colour, use Y channel histogram to match to target histogram
            cout << "target color, input color" << endl;
            Mat ycrcb_input;
            cvtColor(input_img, ycrcb_input,COLOR_BGR2YCrCb);
            vector<Mat> channels_input;
            split(ycrcb_input,channels_input);
            channels_input[0] = hist_match_channel(channels_input[0],channels_target[0]);
            merge(channels_input,ycrcb_input);
            cvtColor(ycrcb_input,out_img,COLOR_YCrCb2BGR);  
        }
        else
        {
            //if input is grayscale, use the image directly
            cout << "target color, input grayscale" << endl;
            out_img = hist_match_channel(input_img,channels_target[0]);

        }

    }
    else
    {
        if(input_img.channels()==3)
        {
            //if input is colour, and target is grayscale 
            //use Y channel histogram of input to match to target histogram 
            cout << "target grayscale, input color" << endl;
            Mat ycrcb_input;
            cvtColor(input_img, ycrcb_input,COLOR_BGR2YCrCb);
            vector<Mat> channels_input;
            split(ycrcb_input,channels_input);
            channels_input[0] = hist_match_channel(channels_input[0],target_img);
            merge(channels_input,ycrcb_input);
            cvtColor(ycrcb_input,out_img,COLOR_YCrCb2BGR);  
        }
        else
        {
            //if both are grayscale, use the images directly
            cout << "target grayscale, input grayscale" << endl;
            out_img = hist_match_channel(input_img,target_img);

        }

    } 

    //display and save final output image
    namedWindow("Output Image", CV_WINDOW_AUTOSIZE);
    imshow("Output Image", out_img);
    imwrite(result_name,out_img);

    //wait for user to press ESC to distroy all windows and exit code
    waitKey();
    return ;
}

int main()
{
    hist_match();
    return 0;
}