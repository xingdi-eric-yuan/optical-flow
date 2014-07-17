// Coarse to fine Lucas & Kanade Optical Flow

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <math.h>
#include <fstream>
#include <iostream>

using namespace cv;
using namespace std;

#define ATD at<double>
#define ATF at<float>
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

double get_Sum9(Mat &m, int y, int x){
    if(x < 0 || x >= m.cols) return 0;
    if(y < 0 || y >= m.rows) return 0;

    double val = 0.0;
    int tmp = 0;
    for(int i =  -1; i <= 1; i++){
        for(int j = -1; j <= 1; j++){
            if(isInsideImage(y + i, x + j, m)){
                ++ tmp;
                val += m.ATD(y + i, x + j);
            } 
        }
    }
    if(tmp == 9) return val;
    else return m.ATD(y, x) * 9;
}

Mat get_Sum9_Mat(Mat &m){
    Mat res = Mat::zeros(m.rows, m.cols, CV_64FC1);
    for(int i = 1; i < m.rows - 1; i++){
        for(int j = 1; j < m.cols - 1; j++){
            res.ATD(i, j) = get_Sum9(m, i, j);
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

void getLucasKanadeOpticalFlow(Mat &img1, Mat &img2, Mat &u, Mat &v){

    Mat fx = get_fx(img1, img2);
    Mat fy = get_fy(img1, img2);
    Mat ft = get_ft(img1, img2);

    Mat fx2 = fx.mul(fx);
    Mat fy2 = fy.mul(fy);
    Mat fxfy = fx.mul(fy);
    Mat fxft = fx.mul(ft);
    Mat fyft = fy.mul(ft);

    Mat sumfx2 = get_Sum9_Mat(fx2);
    Mat sumfy2 = get_Sum9_Mat(fy2);
    Mat sumfxft = get_Sum9_Mat(fxft);
    Mat sumfxfy = get_Sum9_Mat(fxfy);
    Mat sumfyft = get_Sum9_Mat(fyft);

    Mat tmp = sumfx2.mul(sumfy2) - sumfxfy.mul(sumfxfy);
    u = sumfxfy.mul(sumfyft) - sumfy2.mul(sumfxft);
    v = sumfxft.mul(sumfxfy) - sumfx2.mul(sumfyft);
    divide(u, tmp, u);
    divide(v, tmp, v);
}

vector<Mat> getGaussianPyramid(Mat &img, int nLevels){
    vector<Mat> pyr;
    pyr.push_back(img);
    for(int i = 0; i < nLevels - 1; i++){
        Mat tmp;
        pyrDown(pyr[pyr.size() - 1], tmp);
        pyr.push_back(tmp);
    }
    return pyr;
}

void coarseToFineEstimation(Mat &img1, Mat &img2, Mat &u, Mat &v, int nLevels){

    vector<Mat> pyr1 = getGaussianPyramid(img1, nLevels);
    vector<Mat> pyr2 = getGaussianPyramid(img2, nLevels);
    Mat upu, upv;
    for(int i = nLevels - 1; i >= 0; i--){

        Mat tmpu = Mat::zeros(pyr1[i].rows, pyr1[i].cols, CV_64FC1);
        Mat tmpv = Mat::zeros(pyr2[i].rows, pyr2[i].cols, CV_64FC1);
        getLucasKanadeOpticalFlow(pyr1[i], pyr2[i], tmpu, tmpv);
        if(i != nLevels - 1){
            tmpu += upu;
            tmpv += upv;
        }
        if(i == 0){
            u = tmpu;
            v = tmpv;
            return;
        }
        pyrUp(tmpu, upu);
        pyrUp(tmpv, upv);

        Mat map1(upu.size(), CV_32FC2);
        Mat map2(upu.size(), CV_32FC2);
        for (int y = 0; y < map1.rows; ++y){
            for (int x = 0; x < map1.cols; ++x){
                Point2f f = Point2f((float)(upu.ATD(y, x)), (float)(upv.ATD(y, x)));
                map1.at<Point2f>(y, x) = Point2f(x + f.x / 2, y + f.y / 2);
                map2.at<Point2f>(y, x) = Point2f(x - f.x / 2, y - f.y / 2);
            }
        }
        Mat warped1, warped2;
        remap(pyr1[i - 1], warped1, map1, cv::Mat(), INTER_LINEAR);
        remap(pyr2[i - 1], warped2, map2, cv::Mat(), INTER_LINEAR);
        warped1.copyTo(pyr1[i]);
        warped2.copyTo(pyr2[i]);
    }
}

int getMaxLayer(Mat &img){
    int width = img.cols;
    int height = img.rows;
    int res = 1;
    int p = 1;
    while(1){
        int tmp = pow(2, p);
        if(width % tmp == 0) ++ p;
        else break;
    }
    res = p;
    p = 1;
    while(1){
        int tmp = pow(2, p);
        if(height % tmp == 0) ++ p;
        else break;
    }
    res = res < p ? res : p;
    return res;
}

int main(){

//    Mat img1 = imread("anim.00.tif", 0);
//    Mat img2 = imread("anim.01.tif", 0);
//    Mat img1 = imread("frame07.png", 0);
//    Mat img2 = imread("frame08.png", 0);
    Mat ori1 = imread("table1.jpg", 0);
    Mat ori2 = imread("table2.jpg", 0);
    Mat img1 = ori1(Rect(0, 0, 640, 448));
    Mat img2 = ori2(Rect(0, 0, 640, 448));


//    Mat img1 = imread("car1.jpg", 0);
//    Mat img2 = imread("car2.jpg", 0);


    int maxLayer = getMaxLayer(img1);
    cout<<img1.rows<<", "<<img1.cols<<", Max layer = "<<maxLayer<<endl;


    img1.convertTo(img1, CV_64FC1, 1.0/255, 0);
    img2.convertTo(img2, CV_64FC1, 1.0/255, 0);


    Mat u = Mat::zeros(img1.rows, img1.cols, CV_64FC1);
    Mat v = Mat::zeros(img1.rows, img1.cols, CV_64FC1);
    Mat u2 = Mat::zeros(img1.rows, img1.cols, CV_64FC1);
    Mat v2 = Mat::zeros(img1.rows, img1.cols, CV_64FC1);

    if(maxLayer >= 1){
        coarseToFineEstimation(img1, img2, u, v, maxLayer);
        saveMat(u, "U");
        saveMat(v, "V");  
    }

    getLucasKanadeOpticalFlow(img1, img2, u2, v2);
    saveMat(u2, "U2");
    saveMat(v2, "V2");  

//    waitKey(0);

    return 0;
}














