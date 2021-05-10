#include <bits/stdc++.h>
#include <cstdint>

using namespace std; 

//Global variables for structure of headers and headers info 

struct BmpHeader {
    char bitmapSignatureBytes[2] = {'B', 'M'};
    uint32_t sizeOfBitmapFile = 54 + 786432;
    uint32_t reservedBytes = 0;
    uint32_t pixelDataOffset = 54; // image pixel offset
} bmpHeader;

struct BmpInfoHeader {
    uint32_t sizeOfThisHeader = 40;
    int32_t width = 512; // in pixels
    int32_t height = 512; // in pixels
    uint16_t numberOfColorPlanes = 1; // must be 1
    uint16_t colorDepth = 24; //8 for grayscale, 24 for rgb
    uint32_t compressionMethod = 0;
    uint32_t rawBitmapDataSize = 0; // number of image pixels
    int32_t horizontalResolution = 3780; // in pixel per meter
    int32_t verticalResolution = 3780; // in pixel per meter
    uint32_t colorTableEntries = 0;
    uint32_t importantColors = 0;
} bmpInfoHeader;

void printheader() //for troubleshooting and showing required data
{
    // cout << "bitmapSignatureBytes: " << bmpHeader.bitmapSignatureBytes << endl;
    cout << "File Size: " << bmpHeader.sizeOfBitmapFile << endl;
    // cout << "reservedBytes: " << bmpHeader.reservedBytes << endl;
    cout << "DataOffset: " << bmpHeader.pixelDataOffset << endl;    
    return; 
}

void printInfo()
{
    // cout << "sizeOfThisHeader: " << bmpInfoHeader.sizeOfThisHeader << endl;
    cout << "Width: " << bmpInfoHeader.width << endl;
    cout << "Height: " << bmpInfoHeader.height << endl;
    // cout << "numberOfColorPlanes: "<< bmpInfoHeader.numberOfColorPlanes << endl;
    cout << "Color Depth/Bit Width: "<<bmpInfoHeader.colorDepth << endl;
    // cout << "Compression Method: "<<bmpInfoHeader.compressionMethod << endl;
    // cout << "Raw bitmap data size: "<< bmpInfoHeader.rawBitmapDataSize <<endl;
    // cout << "horizontal resolution: "<< bmpInfoHeader.horizontalResolution<<endl;
    // cout << "Vertical Resolution: " << bmpInfoHeader.verticalResolution << endl;
    // cout << "Color Table Entries: " << bmpInfoHeader.colorTableEntries << endl;
    // cout << "important colors: "<< bmpInfoHeader.importantColors <<endl;
    return;
}

vector<vector<vector<int>>> readBMP(const char* filename) //reading the file from BMP in 3 dimensional vector
{
    FILE *img;
    img = fopen(filename,"rb");

    int i=0;

    int counter = 0;

    vector<unsigned char> ch;

    unsigned char a;

    while(i<14) //reading the header data
    {
        fscanf(img,"%c",&a);
        ch.push_back(a);
        i++;
        counter++;
    }

    if (ch[0]!='B' || ch[1]!='M') //checking whether it is BMP file or not
    {
        cout << "NOT A BMP FILE " << endl;
    }

    i=0;
    while(i<4) //skipping header size field
    {
        fscanf(img,"%c",&a);
        i++;
        counter++;
    }

    long width = 0;

    i=0;
    while(i<4) //calculating width using 4 bits data
    {
        fscanf(img,"%c",&a);
        width+=((long)a*pow(256,i));
        i++;
        counter++;
    }
    bmpInfoHeader.width = width;

    long height = 0;

    i=0;
    while(i<4) //calculating height using 4 bits data
    {
        fscanf(img,"%c",&a);
        height+=(long)a*pow(256,i);
        i++;
        counter++;
    }

    bmpInfoHeader.height = height;

    int color_planes = 0 ;

    i=0;
    while(i<2)
    {
        fscanf(img, "%c",&a);
        color_planes+=(int)a*pow(256,i);
        i++;
        counter++;
    }

    long bit_width = 0 ;

    i=0;
    while(i<2) //calculating the bit width(grayscale-8/rgb-24)
    {
        fscanf(img, "%c",&a);
        bit_width+=(long)a*pow(256,i);
        i++;
        counter++;
    }

    bmpInfoHeader.colorDepth = bit_width;

    //filesize and offset value can be calculated using 3 bits as the values are less(allocated are 4 bits)
    uint32_t offset = (long)ch[12]*65536+(long)ch[11]*256 + (long)ch[10];
    uint32_t filesize = (long)ch[4]*65536 + (long)ch[3]*256+(long)ch[2];

    bmpHeader.sizeOfBitmapFile = filesize;
    bmpHeader.pixelDataOffset = offset;

    bmpInfoHeader.compressionMethod = 0;
    bmpInfoHeader.rawBitmapDataSize = 0;//can be 0 if no compression but usually the number of actual image pixels

    i=0;
    while(i<4)
    {
        fscanf(img,"%c",&a);
        bmpInfoHeader.compressionMethod +=(int)a*pow(256,i);
        i++;
        counter++;
    }

    i=0;
    while(i<4)
    {
        fscanf(img,"%c",&a);
        bmpInfoHeader.rawBitmapDataSize +=(int)a*pow(256,i);
        i++;
        counter++;
    }

    int XpixelPerM = 0;
    i=0;
    while(i<4)
    {
        fscanf(img,"%c",&a);
        counter++;
        XpixelPerM+=(int)a*pow(256,i);
        i++;
    }
    bmpInfoHeader.horizontalResolution = XpixelPerM;

    int YpixelPerM = 0;
    i=0;
    while(i<4)
    {
        fscanf(img,"%c",&a);
        
        counter++;
        YpixelPerM+=(int)a*pow(256,i);
        i++;
    }
    bmpInfoHeader.verticalResolution = YpixelPerM;

    int color_used = 0;
    i=0;

    while(i<4)
    {
        fscanf(img,"%c",&a);
        
        counter++;
        color_used+=(int)a*pow(256,i);
        i++;
    }

    bmpInfoHeader.colorTableEntries = color_used; //256 for grayscale, 0 for RGB

    int important_color = 0;
    i=0;
    while(i<4)
    {
        fscanf(img,"%c",&a);
        
        counter++;
        important_color+=(int)a*pow(256,i);
        i++;
    }
    bmpInfoHeader.importantColors = important_color;

    i = 0;
    while(i<(offset-counter)) //skipping through the fields not required(between 54 bit and offset)
    {
        fscanf(img, "%c",&a);
        i++;
    }
    counter = offset;

    vector<vector<vector<int>>> image;

    //reading the image pixel values

    if(bit_width==8) //grayscale
    {
        for(int i=0;i<height;i++)
        {
            vector<vector<int>> temp;
            for(int j=0;j<width;j++)
            {
                vector<int> temp1;
                fscanf(img, "%c",&a);
                temp1.push_back((int)a);
                temp.push_back(temp1);
            }
            image.push_back(temp);
        }
    }
    else //rgb
    {
        for(int i=0;i<height;i++)
        {
            vector<vector<int>> temp;
            for(int j=0;j<width;j++)
            {
                vector<int> temp1; //stored as bgr only
                for(int k=0;k<3;k++)
                {
                    fscanf(img, "%c",&a);
                    temp1.push_back(a);
                }
                temp.push_back(temp1);
            }
            image.push_back(temp); 
        }
    }

    fclose(img);
    return image;
}

//Function to convert to grayscale
vector<vector<vector<int>>> gray(vector<vector<vector<int>>> image)
{
    //if already a grayscale image then return
    if(image[0][0].size()==1)  
    {
        return image;
    }
    else //for colour image, use weighted average
    {
        vector<vector<vector<int>>> gray_img(image.size(), vector<vector<int>>(image[0].size(), vector<int>(0)));
        for(int i=0;i<image.size();i++)
        {
            for(int j=0;j<image[0].size();j++)
            {
                gray_img[i][j].push_back((int)(0.11*image[i][j][0]+0.59*image[i][j][1]+0.3*image[i][j][2]));
            }
        }
        return gray_img;
    }
    
}

//Function to flip the image w.r.t diagonal 
vector<vector<vector<int>>> flip_diagonal(vector<vector<vector<int>>> &img)
{
    //initialize the required image dimention
    vector<vector<vector<int>>> flip_img(img[0].size(), vector<vector<int>>(img.size(), vector<int>(0)));
    int height = img.size();
    int width = img[0].size();

    for(int i=0; i<height; i++)
    for(int j=0; j<width; j++)
    //take transpose, i.e i->j and j->i 
    if(img[i][j].size()==1)
    flip_img[j][i].push_back(img[i][j][0]);
    else
    {
         flip_img[j][i].push_back(img[i][j][0]);
         flip_img[j][i].push_back(img[i][j][1]);
         flip_img[j][i].push_back(img[i][j][2]);
    }
    
    return flip_img;
}

//Function to rotate image by 90-degree clockwise
vector<vector<vector<int>>> rotate_90(vector<vector<vector<int>>> &img)
{
    //angle by which we have to rotate
    double angle = (M_PI*90)/180;
    int height = img.size();
    int width = img[0].size();
    //calculation of new height and width
    int new_height = int(max(height, width)*sqrt(2));
    new_height -= new_height%4;
    int new_width = int(max(height, width)*sqrt(2));
    new_width -= new_width%4;
    vector<vector<vector<int>>> rot_img(new_height, vector<vector<int>>(new_width, vector<int>(0)));
    //flag to determise is image is gray scale or colour
    int flag = img[0][0].size();
    int i, j;
    for(int ii=0; ii <new_height; ii++)
    {
        for(int jj=0; jj<new_width; jj++)
        {
            i = int(cos(angle)*(ii-int(new_height/2)) + (jj - int(new_width/2))*sin(angle) + int(height/2));
            j = int(-1*sin(angle)*(ii-int(new_height/2)) + (jj - int(new_width/2))*cos(angle) + int(width/2));
            //if input pixel location is in bound then copy the value
            if((i>=0 && i<height) && (j>=0 && j<width))
            {
                if(flag==1)
                {
                    rot_img[ii][jj].push_back(img[i][j][0]);
                }
                else
                {
                     rot_img[ii][jj].push_back(img[i][j][0]);
                     rot_img[ii][jj].push_back(img[i][j][1]);
                     rot_img[ii][jj].push_back(img[i][j][2]);
                }
            }
            //else fill with black
            else
            {
                if(flag==1)
                {
                    rot_img[ii][jj].push_back(0);
                }
                else
                {
                     rot_img[ii][jj].push_back(0);
                     rot_img[ii][jj].push_back(0);
                     rot_img[ii][jj].push_back(0);
                }
            }
        }
    }
    return rot_img;

}

//Function to rotate image by 45-degree clockwise
vector<vector<vector<int>>> rotate_45(vector<vector<vector<int>>> &img)
{
    //angle by which we have to rotate
    double angle = (M_PI*45)/180;
    int height = img.size();
    int width = img[0].size();
    //calculation of new height and width
    int new_height = int(max(height, width)*sqrt(2));
    new_height -= new_height%4;
    int new_width = int(max(height, width)*sqrt(2));
    new_width -= new_width%4;
    vector<vector<vector<int>>> rot_img(new_height, vector<vector<int>>(new_width, vector<int>(0)));
    //flag to determise is image is gray scale or colour
    int flag = img[0][0].size();
    int i, j;
    for(int ii=0; ii <new_height; ii++)
    {
        for(int jj=0; jj<new_width; jj++)
        {
            i = int(cos(angle)*(ii-int(new_height/2)) + (jj - int(new_width/2))*sin(angle) + int(height/2));
            j = int(-1*sin(angle)*(ii-int(new_height/2)) + (jj - int(new_width/2))*cos(angle) + int(width/2));
            //if input pixel location is in bound then copy the value
            if((i>=0 && i<height) && (j>=0 && j<width))
            {
                if(flag==1)
                {
                    rot_img[ii][jj].push_back(img[i][j][0]);
                }
                else
                {
                     rot_img[ii][jj].push_back(img[i][j][0]);
                     rot_img[ii][jj].push_back(img[i][j][1]);
                     rot_img[ii][jj].push_back(img[i][j][2]);
                }
            }
            //else fill with black
            else
            {
                if(flag==1)
                {
                    rot_img[ii][jj].push_back(0);
                }
                else
                {
                     rot_img[ii][jj].push_back(0);
                     rot_img[ii][jj].push_back(0);
                     rot_img[ii][jj].push_back(0);
                }
            }
        }
    }
    return rot_img;

}

//Function to scale the image 2 times
vector<vector<vector<int>>> scale_img(vector<vector<vector<int>>> &img)
{
    //scaling factor, must be a integer
    int scale = 2;
    int height = img.size();
    int width = img[0].size();
    //first scaling along width
    vector<vector<vector<int>>> temp(height, vector<vector<int>>(scale*width, vector<int>(0)));
    for(int i=0; i<height; i++)
	{
		for(int j=0; j<width; j++)
		{
			for(int k=0; k<scale; k++)
			{
                if(img[i][j].size()==1)
				temp[i][j*scale + k].push_back(img[i][j][0]);
                else
                {
                    temp[i][j*scale + k].push_back(img[i][j][0]);
                    temp[i][j*scale + k].push_back(img[i][j][1]);
                    temp[i][j*scale + k].push_back(img[i][j][2]);
                }
			}
		}
	}

    //scaling along height    
    vector<vector<vector<int>>> s_img(height*scale, vector<vector<int>>(scale*width, vector<int>(0)));
    for(int i=0; i<height; i++)
	{
		for(int j=0; j<scale*width; j++)
		{
			for(int k=0; k<scale; k++)
			{
                if(img[i][j].size()==1)
				s_img[i*scale+k][j].push_back(temp[i][j][0]);
                else
                {
                    s_img[i*scale+k][j].push_back(temp[i][j][0]);
                    s_img[i*scale+k][j].push_back(temp[i][j][1]);
                    s_img[i*scale+k][j].push_back(temp[i][j][2]);
                }
			}
		}
	}
    return s_img;  
}

void writeBMP(vector<vector<vector<int>>> img, const char* filename)
{
    bmpHeader.pixelDataOffset = 4*256+14+40; //color_table+header+header_info
    bmpHeader.sizeOfBitmapFile = bmpHeader.pixelDataOffset+img.size()*img[0].size();
    bmpInfoHeader.width = img[0].size();
    bmpInfoHeader.height = img.size();
    bmpInfoHeader.colorDepth = 8; //grayscale image
    bmpInfoHeader.horizontalResolution = 0;
    bmpInfoHeader.verticalResolution = 0;
    bmpInfoHeader.colorTableEntries = 256; //total colors to be used
    bmpInfoHeader.rawBitmapDataSize = img.size()*img[0].size();

    FILE *outputi = fopen(filename,"wb");

    fwrite(&bmpHeader.bitmapSignatureBytes,2,1,outputi);
    fwrite(&bmpHeader.sizeOfBitmapFile,4,1,outputi);
    fwrite(&bmpHeader.reservedBytes,4,1,outputi);
    fwrite(&bmpHeader.pixelDataOffset,4,1,outputi);

    fwrite(&bmpInfoHeader.sizeOfThisHeader,4,1,outputi);
    fwrite(&bmpInfoHeader.width,4,1,outputi);
    fwrite(&bmpInfoHeader.height,4,1,outputi);
    fwrite(&bmpInfoHeader.numberOfColorPlanes,2,1,outputi);
    fwrite(&bmpInfoHeader.colorDepth,2,1,outputi);
    fwrite(&bmpInfoHeader.compressionMethod,4,1,outputi);
    fwrite(&bmpInfoHeader.rawBitmapDataSize,4,1,outputi);
    fwrite(&bmpInfoHeader.horizontalResolution,4,1,outputi);
    fwrite(&bmpInfoHeader.verticalResolution,4,1,outputi);
    fwrite(&bmpInfoHeader.colorTableEntries,4,1,outputi);
    fwrite(&bmpInfoHeader.importantColors,4,1,outputi);

    vector<vector<char>> new_img;
    for(int i=0;i<img.size();i++)
    {
        vector<char> temp;
        for(int j=0;j<img[0].size();j++)
        {
            temp.push_back((char)img[i][j][0]);
        }
        new_img.push_back(temp);
    }

    int zeroi = 0;
    int changei = 1;

    //creating the color table(BGRX) of the length of 4*number_of_color_used
    //grayscale so b=g=r=i 
    //X/alpha = 0 always
    for(int i=0;i<bmpHeader.pixelDataOffset-54;i++)
    {
        changei = i/4;
        if(i%4==3)
            fwrite(&zeroi,1,1,outputi);
        else
        {
            fwrite(&changei,1,1,outputi);
        }        
    }

    //writing the image in the file
    for(int i=0;i<img.size();i++)
    {
        for(int j=0;j<img[0].size();j++)
        {
           fwrite(&new_img[i][j],1,1,outputi);    
        }
    }

    fclose(outputi);
    return;
}

int main()
{
     char const *filename = "lena_colored_256.bmp";
     //char const *filename = "cameraman.bmp";

    vector<vector<vector<int>>> image;
    image = readBMP(filename);

    cout << endl;
    printheader();
    printInfo();
    cout << endl;
    
    vector<vector<vector<int>>> gray_img;
    gray_img = gray(image);
    char const *savefilename = "gray.bmp";
    writeBMP(gray_img,savefilename);    

    vector<vector<vector<int>>> diagnol_img;
    diagnol_img = flip_diagonal(gray_img);
    savefilename = "diagnol.bmp";
    writeBMP(diagnol_img,savefilename);

    vector<vector<vector<int>>> rotate_90_img;
    rotate_90_img = rotate_90(gray_img);
    savefilename = "90rotate.bmp";
    writeBMP(rotate_90_img,savefilename);

    vector<vector<vector<int>>> rotate_45_img;
    rotate_45_img = rotate_45(gray_img);
    savefilename = "45rotate.bmp";
    writeBMP(rotate_45_img,savefilename);

    vector<vector<vector<int>>> scale_img1;
    scale_img1 = scale_img(gray_img);
    savefilename = "scaled.bmp";
    writeBMP(scale_img1,savefilename);
    
    return 0;
}
