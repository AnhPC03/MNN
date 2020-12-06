/*******************************************************************************
 * References:
 * preprocess: https://github.com/AspenDove/MNN-YOLOv3
 * sort and nms: https://github.com/wlguan/MNN-yolov3
 * scaleCoords: https://github.com/ultralytics/yolov5/blob/8918e6347683e0f2a8a3d7ef93331001985f6560/utils/general.py#L159
*******************************************************************************/

#include <iostream>
#include "yolov5.hpp"

/*----------------------------------------------------------------------------*/
Yolov5::Yolov5(const std::string &modelPath, const int inputSize, const int numThreads, const float confThresh, const float iouThresh)
{
    m_iInputSize = inputSize;
    m_iNumThreads = numThreads;
    m_fConfThresh = confThresh;
    m_fIouThresh = iouThresh;
    m_pad = cv::Scalar(128, 128, 128);

    m_intrpr = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(modelPath.c_str()));
    MNN::ScheduleConfig config;
    config.numThread = numThreads;
    MNN::BackendConfig backendConfig;
    backendConfig.precision = (MNN::BackendConfig::PrecisionMode)2;
    config.backendConfig = &backendConfig;
    m_sess = m_intrpr->createSession(config);
    m_input = m_intrpr->getSessionInput(m_sess, nullptr);
}

/*----------------------------------------------------------------------------*/
void Yolov5::preprocess(cv::Mat &rawImg, cv::Mat &resizedImg)
{
    int resizedHeight, resizedWidth;
    m_iImgWidth = rawImg.rows;
    m_iImgHeight = rawImg.cols;

    m_fRatio = std::min(1.0f * m_iInputSize / m_iImgWidth, 1.0f * m_iInputSize / m_iImgHeight);

    resizedHeight = int(m_iImgWidth * m_fRatio);
    resizedWidth = int(m_iImgHeight * m_fRatio);

    // odd number->pad size error
    if (resizedHeight % 2 != 0)
        resizedHeight -= 1;
    if (resizedWidth % 2 != 0)
        resizedWidth -= 1;

    m_iPad_W = (m_iInputSize - resizedWidth) / 2;
    m_iPad_H = (m_iInputSize - resizedHeight) / 2;

    cv::resize(rawImg, resizedImg, cv::Size(resizedWidth, resizedHeight), 0, 0, cv::INTER_LINEAR);
    cv::copyMakeBorder(resizedImg, resizedImg, m_iPad_H, m_iPad_H, m_iPad_W, m_iPad_W, cv::BORDER_CONSTANT, m_pad);

    m_iPadedImgWidth = resizedImg.rows;
    m_iPadedImgHeight = resizedImg.cols;
}

/*----------------------------------------------------------------------------*/
void Yolov5::detect(cv::Mat &resizedImg, std::vector<DetResult> &vDetResults)
{
    std::shared_ptr<MNN::CV::ImageProcess> pretreat(
        MNN::CV::ImageProcess::create(MNN::CV::BGR, MNN::CV::RGB, m_afMean, 3,
                                      m_afStd, 3));
    pretreat->convert(resizedImg.data, m_iInputSize, m_iInputSize, resizedImg.step[0], m_input);

    m_intrpr->runSession(m_sess);

    std::string output = "output";
    MNN::Tensor *tfOutput = m_intrpr->getSessionOutput(m_sess, output.c_str());

    MNN::Tensor tfOutputHost(tfOutput, tfOutput->getDimensionType());
    tfOutput->copyToHostTensor(&tfOutputHost);

    auto size = tfOutputHost.elementSize();
    std::vector<float> tempValues(size);

    auto values = tfOutputHost.host<float>();
    for (int i = 0; i < size; ++i)
    {
        tempValues[i] = values[i];
    }

    int numOutputs = int(tfOutputHost.shape()[1]);

    std::vector<std::vector<DetResult>> clsCandidates(m_iNumClasses);

    for (int i = 0; i < numOutputs; ++i)
    {
        float prob = tempValues[i * (5 + m_iNumClasses) + 4];
        auto maxcls = std::max_element(tempValues.begin() + i * (5 + m_iNumClasses) + 5, tempValues.begin() + i * (5 + m_iNumClasses) + (5 + m_iNumClasses));
        int clsidx = maxcls - (tempValues.begin() + i * (5 + m_iNumClasses) + 5);
        float score = prob * (*maxcls);
        if (score < m_fConfThresh)
            continue;

        float bx = (tempValues[i * (5 + m_iNumClasses) + 0]);
        float bw = (tempValues[i * (5 + m_iNumClasses) + 2]);
        float by = (tempValues[i * (5 + m_iNumClasses) + 1]);
        float bh = (tempValues[i * (5 + m_iNumClasses) + 3]);

        float xmin = bx - bw / 2.0f;
        float ymin = by - bh / 2.0f;

        DetResult detResult;
        detResult.cvfRect = cv::Rect_<float>(xmin, ymin, bw, bh);
        detResult.iLabel = clsidx;
        detResult.fConf = score;
        clsCandidates[clsidx].push_back(detResult);
    }

    for (int i = 0; i < (int)clsCandidates.size(); i++)
    {
        std::vector<DetResult> &candidates = clsCandidates[i];

        sortInplace(candidates);

        std::vector<int> pickedIndices;
        nms(candidates, pickedIndices);

        for (int j = 0; j < (int)pickedIndices.size(); j++)
        {
            int z = pickedIndices[j];
            vDetResults.push_back(candidates[z]);
        }
    }
}

/*----------------------------------------------------------------------------*/
void Yolov5::nms(const std::vector<DetResult> &vDetResults, std::vector<int> &pickedIndices)
{
    pickedIndices.clear();
    const int n = vDetResults.size();
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = vDetResults[i].cvfRect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const DetResult &a = vDetResults[i];
        int keep = 1;
        for (int j = 0; j < (int)pickedIndices.size(); j++)
        {
            const DetResult &b = vDetResults[pickedIndices[j]];

            // intersection over union
            float inter_area = calcIntersectArea(a, b);
            float union_area = areas[i] + areas[pickedIndices[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > m_fIouThresh)
                keep = 0;
        }

        if (keep)
            pickedIndices.push_back(i);
    }
}

/*----------------------------------------------------------------------------*/
void Yolov5::sortInplace(std::vector<DetResult> &vDetResults)
{
    if (vDetResults.empty())
        return;

    sortInplace(vDetResults, 0, vDetResults.size() - 1);
}

/*----------------------------------------------------------------------------*/
void Yolov5::sortInplace(std::vector<DetResult> &vDetResults, int left, int right)
{
    int i = left;
    int j = right;
    float p = vDetResults[(left + right) / 2].fConf;

    while (i <= j)
    {
        while (vDetResults[i].fConf > p)
            i++;

        while (vDetResults[j].fConf < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(vDetResults[i], vDetResults[j]);

            i++;
            j--;
        }
    }
}

/*----------------------------------------------------------------------------*/
float Yolov5::calcIntersectArea(const DetResult &cDetRet1, const DetResult &cDetRet2)
{
    cv::Rect_<float> inter = cDetRet1.cvfRect & cDetRet2.cvfRect;
    return inter.area();
}

/*----------------------------------------------------------------------------*/
cv::Mat Yolov5::drawDetResults(const cv::Mat &imgBGR, const std::vector<DetResult> &vDetResults)
{
    cv::Mat image = imgBGR.clone();
    for (auto &it : vDetResults)
    {
        cv::rectangle(image, it.cvfRect, cv::Scalar(0, 0, 255));
    }

    return image;
}

/*----------------------------------------------------------------------------*/
void Yolov5::scaleCoords(std::vector<DetResult> &vDetResults)
{
    float gain = std::min(1.0f * m_iPadedImgWidth / float(m_iImgWidth), 1.0f * m_iPadedImgHeight / float(m_iImgHeight));
    float padY = (m_iPadedImgWidth - m_iImgWidth * gain) / 2.0f;
    float padX = (m_iPadedImgHeight - m_iImgHeight * gain) / 2.0f;

    for (auto &it : vDetResults)
    {
        it.cvfRect.x = (it.cvfRect.x - padX) / gain;
        it.cvfRect.y = (it.cvfRect.y - padY) / gain;
        it.cvfRect.width /= gain;
        it.cvfRect.height /= gain;
    }

    for (auto &it : vDetResults)
    {
        clipBox(it.cvfRect);
    }
}

/*----------------------------------------------------------------------------*/
void Yolov5::clipBox(cv::Rect_<float> &cvfRect)
{
    cvfRect.x = std::max(float(cvfRect.x), 0.0f);
    cvfRect.y = std::max(float(cvfRect.y), 0.0f);

    cvfRect.width = std::min(float(cvfRect.width), float(m_iImgWidth - 1));
    cvfRect.height = std::min(float(cvfRect.height), float(m_iImgHeight - 1));
}

/*----------------------------------------------------------------------------*/
Yolov5::~Yolov5()
{
    m_intrpr->releaseModel();
    m_intrpr->releaseSession(m_sess);
}

/*----------------------------------------------------------------------------*/
int main(int argc, char **argv)
{
    if (argc < 6)
    {
        std::cout << "Usage: ./yolo.out model.mnn input.jpg inputSize confThresh iouThresh" << std::endl;
        return 0;
    }

    const std::string modelPath = argv[1];
    const std::string imgPath = argv[2];
    const int inputSize = int(atof(argv[3]));
    const float confThresh = atof(argv[4]);
    const float iouThresh = atof(argv[5]);

    cv::Mat rawImg, resizedImg, showedImg;
    std::vector<DetResult> vDetResults;

    // Yolov5 yolov5(modelPath, inputSize, 4, confThresh, iouThresh);
    // std::shared_ptr<Yolov5> yolov5 = std::make_shared<Yolov5>(modelPath, inputSize, 4, confThresh, iouThresh);
    auto yolov5 = std::make_shared<Yolov5>(modelPath, inputSize, 4, confThresh, iouThresh);

    rawImg = cv::imread(imgPath);

    yolov5->preprocess(rawImg, resizedImg);
    yolov5->detect(resizedImg, vDetResults);
    yolov5->scaleCoords(vDetResults);

    showedImg = yolov5->drawDetResults(rawImg, vDetResults);
    cv::imshow("yolov5", showedImg);
    cv::waitKey(0);

    return 0;
}