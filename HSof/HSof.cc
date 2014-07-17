// Horn&Schunck Optical Flow
//
// Author: Eric Yuan
// Blog: http://eric-yuan.me
// You are FREE to use the following code for ANY purpose.
//
// To run this code, you should have OpenCV in your computer.
// Have fun with it.
// 

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <math.h>
#include <fstream>
#include <iostream>

using namespace cv;
using namespace std;

#define ATD at<double>
#define elif else if

#ifndef bool
    #define bool int
    #define false ((bool)0)
    #define true  ((bool)1)
#endif


Mat get_fx(Mat &src1, Mat &src2){
    Mat fx;
    Mat kernel = Mat::ones(2, 2, CV_64FC1);
    kernel.ATD(0, 0) = -1.0;
    kernel.ATD(1, 0) = -1.0;

    Mat dst1, dst2;
    filter2D(src1, dst1, -1, kernel);
    filter2D(src2, dst2, -1, kernel);

    fx = dst1 + dst2;
    return fx;
}

Mat get_fy(Mat &src1, Mat &src2){
    Mat fy;
    Mat kernel = Mat::ones(2, 2, CV_64FC1);
    kernel.ATD(0, 0) = -1.0;
    kernel.ATD(0, 1) = -1.0;

    Mat dst1, dst2;
    filter2D(src1, dst1, -1, kernel);
    filter2D(src2, dst2, -1, kernel);

    fy = dst1 + dst2;
    return fy;
}

Mat get_ft(Mat &src1, Mat &src2){
    Mat ft;
    Mat kernel = Mat::ones(2, 2, CV_64FC1);
    kernel = kernel.mul(-1);

    Mat dst1, dst2;
    filter2D(src1, dst1, -1, kernel);
    kernel = kernel.mul(-1);
    filter2D(src2, dst2, -1, kernel);

    ft = dst1 + dst2;
    return ft;
}

bool isInsideImage(int y, int x, Mat &m){
    int width = m.cols;
    int height = m.rows;
    if(x >= 0 && x < width && y >= 0 && y < height) return true;
    else return false;
}

double get_Average4(Mat &m, int y, int x){
    if(x < 0 || x >= m.cols) return 0;
    if(y < 0 || y >= m.rows) return 0;

    double val = 0.0;
    int tmp = 0;
    if(isInsideImage(y - 1, x, m)){
        ++ tmp;
        val += m.ATD(y - 1, x);
    }
    if(isInsideImage(y + 1, x, m)){
        ++ tmp;
        val += m.ATD(y + 1, x);
    }
    if(isInsideImage(y, x - 1, m)){
        ++ tmp;
        val += m.ATD(y, x - 1);
    }
    if(isInsideImage(y, x + 1, m)){
        ++ tmp;
        val += m.ATD(y, x + 1);
    }
    return val / tmp;
}

Mat get_Average4_Mat(Mat &m){
    Mat res = Mat::zeros(m.rows, m.cols, CV_64FC1);
    for(int i = 0; i < m.rows; i++){
        for(int j = 0; j < m.cols; j++){
            res.ATD(i, j) = get_Average4(m, i, j);
        }
    }
    return res;
}

void saveMat(Mat &M, string s){
    s += ".txt";
    FILE *pOut = fopen(s.c_str(), "w+");
    for(int i=0; i<M.rows; i++){
        for(int j=0; j<M.cols; j++){
            fprintf(pOut, "%lf", M.ATD(i, j));
            if(j == M.cols - 1) fprintf(pOut, "\n");
            else fprintf(pOut, " ");
        }
    }
    fclose(pOut);
}

void getHornSchunckOpticalFlow(Mat img1, Mat img2){

    double lambda = 0.05;
    Mat u = Mat::zeros(img1.rows, img1.cols, CV_64FC1);
    Mat v = Mat::zeros(img1.rows, img1.cols, CV_64FC1);

    Mat fx = get_fx(img1, img2);
    Mat fy = get_fy(img1, img2);
    Mat ft = get_ft(img1, img2);

    int i = 0;
    double last = 0.0;
    while(1){
        Mat Uav = get_Average4_Mat(u);
        Mat Vav = get_Average4_Mat(v);
        Mat P = fx.mul(Uav) + fy.mul(Vav) + ft;
        Mat D = fx.mul(fx) + fy.mul(fy) + lambda;
        Mat tmp;
        divide(P, D, tmp);
        Mat utmp, vtmp;
        utmp = Uav - fx.mul(tmp);
        vtmp = Vav - fy.mul(tmp);
        Mat eq = fx.mul(utmp) + fy.mul(vtmp) + ft;
        double thistime = mean(eq)[0];
        cout<<"i = "<<i<<", mean = "<<thistime<<endl;
        if(i != 0 && fabs(last) <= fabs(thistime)) break;
        i++;
        last = thistime;
        u = utmp;
        v = vtmp;
    }
    saveMat(u, "U");
    saveMat(v, "V");   
}



int main(){

//    Mat img1 = imread("car1.jpg", 0);
//    Mat img2 = imread("car2.jpg", 0);
    Mat img1 = imread("table1.jpg", 0);
    Mat img2 = imread("table2.jpg", 0);
//    Mat img1 = imread("anim.00.tif", 0);
//    Mat img2 = imread("anim.01.tif", 0);


    img1.convertTo(img1, CV_64FC1, 1.0/255, 0);
    img2.convertTo(img2, CV_64FC1, 1.0/255, 0);

    getHornSchunckOpticalFlow(img1, img2);





//    waitKey(0);

    return 0;
}