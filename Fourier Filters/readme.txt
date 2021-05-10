The code is self sufficient, just we need to specify the input image folder location as argument

Note: For this code, you will need dirent library installed in your system
This library is only used to read images name form the folder that's it. 

For compiling the code:
We need to have opencv installed in the system and below is the sample for the CMakeLists.txt.

cmake_minimum_required (VERSION 3.0.2)
project (exp2)
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable (${PROJECT_NAME} hist_match.cpp) // to be changed here which file need to be compiled.
target_link_libraries (${PROJECT_NAME} ${OpenCV_LIBS})

We can use cmake as above but we can also use a make file which includes all the required flags the
compiler needs to compile the programs. 

NOTE:
For most of the cases to include opencv you have to write like this 
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

if this gives an error that file core.hpp doesn't found then try:
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

In some systems due to instalation difference subfolders are not made. 

