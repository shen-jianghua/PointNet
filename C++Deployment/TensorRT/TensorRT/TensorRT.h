#pragma once
#include<NvInfer.h>
#include<NvOnnxParser.h>
#include<NvOnnxConfig.h>
#include"common/argsParser.h"

#include<opencv2/opencv.hpp>

#include<iostream>
#include <vector>

#define CHECK(status)                                          \
    do                                                         \
    {                                                          \
        auto ret = (status);                                   \
        if (ret != 0)                                          \
        {                                                      \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)
#define DEVICE 0
#define NMS_THRESH 0.45
#define BOX_CONF_THRESH 0.3
#define RUN_FP16 true
#define RUN_INT8 false
#define BATCH_SIZE 1


class Logger : public nvinfer1::ILogger
{
public:
    Logger(Severity severity = Severity::kWARNING) : reportableSeverity(severity)
    {
    }

    void log(Severity severity, char const* msg) noexcept
        // void log(Severity severity, const char* msg) noexcept
    {
        // suppress messages with severity enum value greater than the reportable
        if (severity > reportableSeverity)
            return;

        switch (severity)
        {
        case Severity::kINTERNAL_ERROR:
            std::cerr << "INTERNAL_ERROR: ";
            break;
        case Severity::kERROR:
            std::cerr << "ERROR: ";
            break;
        case Severity::kWARNING:
            std::cerr << "WARNING: ";
            break;
        case Severity::kINFO:
            std::cerr << "INFO: ";
            break;
        default:
            std::cerr << "UNKNOWN: ";
            break;
        }
        std::cerr << msg << std::endl;
    }

    Severity reportableSeverity;
};


static Logger g_logger_;

class PointNetONNX
{
public:
    PointNetONNX(const std::string& onnx_file, const std::string& engine_file);
    std::vector<float> prepareImage(const std::vector<float>& pointcloud, const int sampleNum);
    bool onnxToTRTModel(nvinfer1::IHostMemory* trt_model_stream);

    void onnxToTRTModel(const std::string& model_file, nvinfer1::IHostMemory*& trt_model_stream);
    bool loadEngineFromFile();
    void doInference(const std::vector<float>& pointcloud, const int sampleNum);

private:
    const std::string m_onnx_file;
    const std::string m_engine_file;
    samplesCommon::Args m_gArgs;
    nvinfer1::ICudaEngine* m_engine;

    bool saveEngineFile(nvinfer1::IHostMemory* data);
    std::unique_ptr<char[]> readEngineFile(int& length);

    int64_t volume(const nvinfer1::Dims& d);
    unsigned int getElementSize(nvinfer1::DataType t);

    void resample(std::vector<float>& points, int nums);
    void pointCloudNormalize(std::vector<float>& points);
};