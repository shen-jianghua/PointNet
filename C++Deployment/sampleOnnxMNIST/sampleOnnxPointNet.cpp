/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//!
//! SampleOnnxPointNet.cpp
//! This file contains the implementation of the ONNX MNIST sample. It creates the network using
//! the MNIST onnx model.
//! It can be run with the following command line:
//! Command: ./sample_onnx_mnist [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]
//! [--useDLACore=<int>]
//!

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

using samplesCommon::SampleUniquePtr;

const std::string gSampleName = "TensorRT.sample_onnx_mnist";


struct PointXYZ
{
    float x;
    float y;
    float z;
};

void largeIdxEachRow(float** array, int rows, int cols, float* indices)
{
    for (int i = 0; i < rows; i++)
    {
        int maxElement = array[i][0];
        int idx = 0;
        for (int j = 0; j < cols; j++)
        {
            if (maxElement < array[i][j])
            {
                maxElement = array[i][j];
                idx = j;
            }
        }
        indices[i] = idx;
    }
}

void savePointCloud(const std::vector<PointXYZ>& pointcloud, std::string filepath)
{
    try
    {
        std::ofstream outfile(filepath);
        if (outfile.is_open())
        {
            for (auto point : pointcloud)
            {
                outfile << point.x << " " << point.y << " " << point.z << std::endl;
            }
            outfile.close();
        }

    }
    catch (const std::exception&)
    {
        std::cerr << "保存点云失败!" << std::endl;
    }
}

//! \brief  The SampleOnnxPointNet class implements the ONNX MNIST sample
//!
//! \details It creates the network using an ONNX model
//!
class SampleOnnxPointNet
{
public:
    SampleOnnxPointNet(const samplesCommon::OnnxSampleParams& params)
        : mParams(params)
        , mEngine(nullptr)
    {
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer(float* pc);

    bool saveEngineFile(IHostMemory* data, const std::string& engineFile);
    bool loadEngineFile(const std::string& engineFile);
    std::unique_ptr<char[]> readEngineFile(const std::string& engineFile, int& length);

private:
    samplesCommon::OnnxSampleParams mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
    int mNumber{0};             //!< The number to classify

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    //!
    //! \brief Parses an ONNX model for MNIST and creates a TensorRT network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvonnxparser::IParser>& parser);

    //!
    //! \brief Reads the input  and stores the result in a managed buffer
    //!
    bool processInput(const samplesCommon::BufferManager& buffers, float* pc);

    //!
    //! \brief Classifies digits and verify result
    //!
    bool verifyOutput(const samplesCommon::BufferManager& buffers, float*& input);


    void pointCloudNormalize(float* points, int num);

    void resample(float* points, int nums);
    
};

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the Onnx MNIST network by parsing the Onnx model and builds
//!          the engine that will be used to run MNIST (mEngine)
//!
//! \return true if the engine was created successfully and false otherwise
//!
bool SampleOnnxPointNet::build()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto parser
        = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }

    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed)
    {
        return false;
    }

    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
        return false;
    }
    config->setProfileStream(*profileStream);

    SampleUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        return false;
    }
    //saveEngineFile(plan.get(), "bin/pointnet.engine");

    SampleUniquePtr<IRuntime> runtime{createInferRuntime(sample::gLogger.getTRTLogger())};
    if (!runtime)
    {
        return false;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    ASSERT(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    ASSERT(mInputDims.nbDims == 3);

    ASSERT(network->getNbOutputs() == 3);
    mOutputDims = network->getOutput(0)->getDimensions();
    ASSERT(mOutputDims.nbDims == 3);

    return true;
}

//!
//! \brief Uses a ONNX parser to create the Onnx MNIST Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the Onnx MNIST network
//!
//! \param builder Pointer to the engine builder
//!
bool SampleOnnxPointNet::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser)
{
    auto parsed = parser->parseFromFile(locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(),
        static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllDynamicRanges(network.get(), 127.0f, 127.0f);
    }

    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool SampleOnnxPointNet::infer(float* pc)
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    // Read the input data into the managed buffers
    ASSERT(mParams.inputTensorNames.size() == 1);
    if (!processInput(buffers, pc))
    {
        return false;
    }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    // Verify results
    if (!verifyOutput(buffers, pc))
    {
        return false;
    }

    return true;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool SampleOnnxPointNet::processInput(const samplesCommon::BufferManager& buffers, float* pc)
{
    const int inputW = mInputDims.d[1];//3
    const int inputH = mInputDims.d[2];//2500
    float* pc_copy = pc;

    resample(pc_copy, inputH);
    pointCloudNormalize(pc_copy, inputH);


    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    for (int i = 0; i < inputH * inputW; i++)
    {
        hostDataBuffer[i] = pc_copy[i];
    }

    return true;
}


void SampleOnnxPointNet::pointCloudNormalize(float* points, int num)
{
    int N = num;//points.size() 等于 num / 3
    float mean_x = 0, mean_y = 0, mean_z = 0;
    for (size_t i = 0; i < N; ++i)
    {
        mean_x += points[3 * i];
        mean_y += points[3 * i + 1];
        mean_z += points[3 * i + 2];
    }
    mean_x /= N;
    mean_y /= N;
    mean_z /= N;

    for (size_t i = 0; i < N; ++i)
    {
        points[3 * i] -= mean_x;
        points[3 * i + 1] -= mean_y;
        points[3 * i + 2] -= mean_z;
    }

    float m = 0;
    for (size_t i = 0; i < N; ++i)
    {
        if (sqrt(pow(points[3 * i], 2) + pow(points[3 * i + 1], 2) + pow(points[3 * i + 2], 2)) > m)
            m = sqrt(pow(points[3 * i], 2) + pow(points[3 * i + 1], 2) + pow(points[3 * i + 2], 2));
    }

    for (size_t i = 0; i < N; ++i)
    {
        points[3 * i] /= m;
        points[3 * i + 1] /= m;
        points[3 * i + 2] /= m;
    }
}

void SampleOnnxPointNet::resample(float* points, int nums)
{
    srand((int)time(0));
    std::vector<int> choice(nums);
    for (size_t i = 0; i < nums; i++)
    {
        choice[i] = rand() % (nums / 3);
    }

    float* temp_points = new float[3 * nums];
    for (size_t i = 0; i < nums; i++)
    {
        temp_points[3 * i] = points[3 * choice[i]];
        temp_points[3 * i + 1] = points[3 * choice[i] + 1];
        temp_points[3 * i + 2] = points[3 * choice[i] + 2];
    }
    points = temp_points;
}


//!
//! \brief Classifies digits and verify result
//!
//! \return whether the classification output matches expectations
//!
bool SampleOnnxPointNet::verifyOutput(const samplesCommon::BufferManager& buffers, float*& input)
{
    const int outputSize = mOutputDims.d[1];
    const int segNum = mOutputDims.d[2];
    float* output = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));

    float** array = new float* [outputSize];
    for (int i = 0; i < mOutputDims.d[1]; i++)
    {
        array[i] = new float[segNum];
        for (int j = 0; j < segNum; j++)
        {
            array[i][j] = output[i * segNum + j];
        }
    }
    float* indices = new float[outputSize];
    largeIdxEachRow(array, outputSize, segNum, indices);

    std::vector<std::vector<PointXYZ>> segment(segNum);
    for (int i = 0; i < outputSize; i++)
    {
        int idx = indices[i];
        PointXYZ point;

        point.x = input[i * 3];
        point.y = input[i * 3 + 1];
        point.z = input[i * 3 + 2];
        segment[idx].push_back(point);
    }

   
    for (size_t i = 0; i < segment.size(); i++)
    {
        std::string newfile = "bin/seg-" + std::to_string(i) + ".txt";
        savePointCloud(segment[i], newfile);
    }


    return true;
}


bool SampleOnnxPointNet::saveEngineFile(IHostMemory* data, const std::string& engineFile)
{
    std::ofstream file;
    file.open(engineFile, std::ios::binary | std::ios::out);
    std::cout << "writing engine file..." << std::endl;
    file.write((const char*)data->data(), data->size());
    std::cout << "save engine file done" << std::endl;
    file.close();
    return true;
}


bool SampleOnnxPointNet::loadEngineFile(const std::string& engineFile)
{
    int length = 0; // 记录data的长度
    std::unique_ptr<char[]> data = readEngineFile(engineFile, length);
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger());

    mEngine = SampleUniquePtr<ICudaEngine>(runtime->deserializeCudaEngine(data.get(), length));
    if (!mEngine)
    {
        std::cout << "Failed to create engine" << std::endl;
        return false;
    }
    return true;
}


std::unique_ptr<char[]> SampleOnnxPointNet::readEngineFile(const std::string& engineFile, int& length)
{
    std::ifstream file;
    file.open(engineFile, std::ios::in | std::ios::binary);
    // 获得文件流的长度
    file.seekg(0, std::ios::end); // 把指针移到末尾
    length = file.tellg();        // 返回当前指针位置
    // 指针移到开始
    file.seekg(0, std::ios::beg);
    // 定义缓存
    std::unique_ptr<char[]> data(new char[length]);
    // 读取文件到缓存区
    file.read(data.get(), length);
    file.close();
    return data;
}


//!
//! \brief Initializes members of the params struct using the command line args
//!
samplesCommon::OnnxSampleParams initializeSampleParams(const samplesCommon::Args& args)
{
    samplesCommon::OnnxSampleParams params;
    if (args.dataDirs.empty()) // Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("weights/");
        params.dataDirs.push_back("../weights/");
    }
    else // Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }
    params.onnxFileName = "segmentation_model_Airplane_0.onnx";
    params.inputTensorNames.push_back("InputPC");
    params.outputTensorNames.push_back("OutputSeg");
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;

    return params;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout
        << "Usage: ./sample_onnx_mnist [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]"
        << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "(data/samples/mnist/, data/mnist/)"
              << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
              << std::endl;
    std::cout << "--int8          Run in Int8 mode." << std::endl;
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
}





int main(int argc, char** argv)
{
    std::vector<float> points;
    std::ifstream infile;
    float x, y, z;
    infile.open("weights/Airplane.txt");
    int point_num = 0;
    while (infile >> x >> y >> z)
    {
        points.push_back(x);
        points.push_back(y);
        points.push_back(z);
        ++point_num;
    }
    infile.close();

    std::string engine_file = "bin/pointnet.engine";


    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        sample::gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, argv);

    sample::gLogger.reportTestStart(sampleTest);

    SampleOnnxPointNet sample(initializeSampleParams(args));

    sample::gLogInfo << "Building and running a GPU inference engine for Onnx MNIST" << std::endl;

    std::fstream engine_reader;
    engine_reader.open(engine_file, std::ios::in);
    if (engine_reader)
    {
        std::cout << "found engine plan" << std::endl;
        sample.loadEngineFile(engine_file);
    }
    else
    {
        if (!sample.build())
        {
            return sample::gLogger.reportFail(sampleTest);
        }
    }



    if (!sample.infer(points.data()))
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    return sample::gLogger.reportPass(sampleTest);
}
