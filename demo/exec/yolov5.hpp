#ifndef yolov5_hpp
#define yolov5_hpp

#pragma once

#include <string>
#include <vector>
#include <chrono>

#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/Tensor.hpp>
#include <MNN/ImageProcess.hpp>

#include <opencv2/opencv.hpp>

struct DetResult
{
    cv::Rect_<float> cvfRect;
    int iLabel;
    float fConf;

    bool operator>(const DetResult &strDetRet) const
    {
        return (fConf > strDetRet.fConf);
    }
};

static std::vector<std::string> clsNames = {"product"};

class Yolov5
{
public:
    Yolov5(const std::string &modelPath, const int inputSize, const int numThreads = 4, const float confThresh = 0.5, const float iouThresh = 0.3);
    void preprocess(cv::Mat &imgBGR, cv::Mat &resizedImg);
    void detect(cv::Mat &imgBGR, std::vector<DetResult> &vDetResults);
    void scaleCoords(std::vector<DetResult> &vDetResults);
    cv::Mat drawDetResults(const cv::Mat &rgb, const std::vector<DetResult> &vDetResults);
    ~Yolov5();

private:
    void nms(const std::vector<DetResult> &vDetResults, std::vector<int> &pickedIndices);
    float calcIntersectArea(const DetResult &cDetRet1, const DetResult &cDetRet2);
    void clipBox(cv::Rect_<float> &cvfRect);

private:
    int m_iNumThreads;

    int m_iImgWidth, m_iImgHeight;           // size of the origin img
    int m_iPadedImgWidth, m_iPadedImgHeight; // size of the padded img

    int m_iInputSize; // input size used when trained the Network
    float m_fRatio;
    int m_iPad_W, m_iPad_H;
    cv::Scalar m_pad;

    float m_fConfThresh, m_fIouThresh;

    /* Img normalization parameters */
    const float m_afMean[3] = {0.0f, 0.0f, 0.0f};
    const float m_afStd[3] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};

    const int m_iNumClasses = int(clsNames.size());

    /* MNN related */
    std::shared_ptr<MNN::Interpreter> m_intrpr;
    MNN::Session *m_sess = nullptr;
    MNN::Tensor *m_input = nullptr;
};

#endif /* yolov5_hpp */