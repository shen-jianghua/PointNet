#include"TensorRT.h"

#include <fstream>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <stdio.h>
#include "common/buffers.h"
#include "common/argsParser.h"
#include "common/buffers.h"
#include "common/common.h"
#include "common/logger.h"
#include "common/parserOnnxConfig.h"
#include <cstdlib>
#include <iostream>
#include <sstream>

PointNetONNX::PointNetONNX(const std::string& onnx_file, const std::string& engine_file)
	: m_onnx_file(onnx_file), m_engine_file(engine_file)
{
}


std::vector<float> PointNetONNX::prepareImage(const std::vector<float>& pointcloud, const int sampleNum)
{
	std::vector<float> result(pointcloud);
	PointNetONNX::resample(result, sampleNum);
	PointNetONNX::pointCloudNormalize(result);
	return result;
}

void PointNetONNX::resample(std::vector<float>& points, int nums)
{
	srand((int)time(0));
	std::vector<int> choice(nums);
	for (size_t i = 0; i < nums; i++)
	{
		choice[i] = rand() % (points.size() / 3);
	}

	std::vector<float> temp_points(3 * nums);
	for (size_t i = 0; i < nums; i++)
	{
		temp_points[3 * i] = points[3 * choice[i]];
		temp_points[3 * i + 1] = points[3 * choice[i] + 1];
		temp_points[3 * i + 2] = points[3 * choice[i] + 2];
	}
	points = temp_points;
}

void PointNetONNX::pointCloudNormalize(std::vector<float>& points)
{
	int N = points.size() / 3;
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

bool PointNetONNX::onnxToTRTModel(nvinfer1::IHostMemory* trt_model_stream)
{
	//1.创建构建器builder
	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger());
	assert(builder != nullptr);

	//2.创建network
	const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);

	//3.创建config
	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
	config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1U << 30);
	config->setMaxWorkspaceSize(16 * (1 << 20));  // 设置最大工作空间
	if (RUN_FP16)
	{
		config->setFlag(nvinfer1::BuilderFlag::kFP16);
	}
	if (RUN_INT8)
	{
		config->setFlag(nvinfer1::BuilderFlag::kINT8);
	}

	//4.创建解析器
	auto parser = nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger());
	// 解析onnx文件
	if (!parser->parseFromFile(this->m_onnx_file.c_str(), static_cast<int>(sample::gLogger.getReportableSeverity())))
	{
		sample::gLogError << "Fail to parse ONNX file" << std::endl;
		return false;
	}


	//5.构建推理引擎
	m_engine = builder->buildEngineWithConfig(*network, *config); //创建engine  第二个断点测试

	//6.序列化
	trt_model_stream = m_engine->serialize();
	nvinfer1::IHostMemory* data = builder->buildSerializedNetwork(*network, *config);
	saveEngineFile(data);

	delete config;
	delete network;
	delete parser;
	// m_engine->destroy();
	return true;
}

void PointNetONNX::onnxToTRTModel(const std::string& model_file,            // name of the onnx model
	nvinfer1::IHostMemory*& trt_model_stream) // output buffer for the TensorRT model
{
	int verbosity = (int)nvinfer1::ILogger::Severity::kWARNING;

	// create the builder
	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(g_logger_);
	// 创建INetworkDefinition 对象
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
	// 创建解析器
	auto parser = nvonnxparser::createParser(*network, g_logger_);

	// 解析onnx文件，并填充网络
	if (!parser->parseFromFile(model_file.c_str(), verbosity))
	{
		std::string msg("failed to parse onnx file");
		g_logger_.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
		exit(EXIT_FAILURE);
	}

	// Build the engine
	builder->setMaxBatchSize(1);
	// 创建iBuilderConfig对象
	nvinfer1::IBuilderConfig* iBuilderConfig = builder->createBuilderConfig();
	// 设置engine可使用的最大GPU临时值
	iBuilderConfig->setMaxWorkspaceSize(1 << 20);


	nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *iBuilderConfig);

	// 将engine序列化，保存到文件中
	trt_model_stream = engine->serialize();
	// save engine
	std::ofstream p("../model.trt", std::ios::binary);
	if (!p) {
		std::cerr << "could not open plan output file" << std::endl;
	}
	p.write(reinterpret_cast<const char*>(trt_model_stream->data()), trt_model_stream->size());
	parser->destroy();
	engine->destroy();
	network->destroy();
	builder->destroy();
	iBuilderConfig->destroy();
}


bool PointNetONNX::loadEngineFromFile()
{
	int length = 0; // 记录data的长度
	std::unique_ptr<char[]> data = readEngineFile(length);
	nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger());
	m_engine = runtime->deserializeCudaEngine(data.get(), length);
	if (!m_engine)
	{
		std::cout << "Failed to create engine" << std::endl;
		return false;
	}
	return true;
}

std::unique_ptr<char[]> PointNetONNX::readEngineFile(int& length)
{
	std::ifstream file;
	file.open(m_engine_file, std::ios::in | std::ios::binary);
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

void PointNetONNX::doInference(const std::vector<float>& pointcloud, const int sampleNum)
{
	std::vector<float> copy_pc(pointcloud);
	nvinfer1::IExecutionContext* context = m_engine->createExecutionContext();
	assert(context != nullptr);
	int nbBindings = m_engine->getNbBindings();
	assert(nbBindings == 2); // 输入和输出，一共是2个

	// 为输入和输出创建空间
	void* buffers[2];                 // 待创建的空间  为指针数组
	std::vector<int64_t> buffer_size; // 要创建的空间大小
	buffer_size.resize(nbBindings);
	for (int i = 0; i < nbBindings; i++)
	{
		nvinfer1::Dims dims = m_engine->getBindingDimensions(i);    // (3, 224, 224)  (1000)
		nvinfer1::DataType dtype = m_engine->getBindingDataType(i); // 0, 0 也就是两个都是kFloat类型
		// std::cout << static_cast<int>(dtype) << endl;
		int64_t total_size = PointNetONNX::volume(dims) * 1 * PointNetONNX::getElementSize(dtype);
		buffer_size[i] = total_size;
		CHECK(cudaMalloc(&buffers[i], total_size));
	}

	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream)); // 创建异步cuda流
	auto out_dim = m_engine->getBindingDimensions(1);
	auto output_size = 1;
	for (int j = 0; j < out_dim.nbDims; j++)
	{
		output_size *= out_dim.d[j];
	}
	float* out = new float[output_size];

	// 开始推理
	auto t_start = std::chrono::high_resolution_clock::now();
	std::vector<float> cur_input = prepareImage(copy_pc, sampleNum);
	auto t_end = std::chrono::high_resolution_clock::now();
	float duration = std::chrono::duration<float, std::milli>(t_end - t_start).count();
	std::cout << "loading image takes " << duration << "ms" << std::endl;
	if (!cur_input.data())
	{
		std::cout << "failed to prepare image" << std::endl;
	}
	auto t_start_inference = std::chrono::high_resolution_clock::now();
	// 将输入传递到GPU
	CHECK(cudaMemcpyAsync(buffers[0], cur_input.data(), buffer_size[0], cudaMemcpyHostToDevice, stream));

	// 异步执行
	t_start = std::chrono::high_resolution_clock::now();
	context->enqueueV2(&buffers[0], stream, nullptr);

	// 输出传回给CPU
	CHECK(cudaMemcpyAsync(out, buffers[1], buffer_size[1], cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);
	auto t_end_inference = std::chrono::high_resolution_clock::now();
	float duration_inference = std::chrono::duration<float, std::milli>(t_end_inference - t_start_inference).count();
	std::cout << "Inference time cost " << duration_inference << "ms" << std::endl;
	std::vector<float> original_result;
	for (int i = 0; i < output_size; i++)
	{
		original_result.push_back(*(out + i));
	}
	
	t_end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration<float, std::milli>(t_end - t_start).count();
	std::cout << duration << " time" << std::endl;

	delete[] out;
}

// 累积乘法 对binding的维度累乘 (3,224,224) => 3*224*224
inline int64_t PointNetONNX::volume(const nvinfer1::Dims& d)
{
	return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

inline unsigned int PointNetONNX::getElementSize(nvinfer1::DataType t)
{
	switch (t)
	{
	case nvinfer1::DataType::kINT32:
		return 4;
	case nvinfer1::DataType::kFLOAT:
		return 4;
	case nvinfer1::DataType::kHALF:
		return 2;
	case nvinfer1::DataType::kINT8:
		return 1;
	}
	throw std::runtime_error("Invalid DataType.");
	return 0;
}

bool PointNetONNX::saveEngineFile(nvinfer1::IHostMemory* data)
{
	std::ofstream file;
	file.open(m_engine_file, std::ios::binary | std::ios::out);
	std::cout << "writing engine file..." << std::endl;
	file.write((const char*)data->data(), data->size());
	std::cout << "save engine file done" << std::endl;
	file.close();
	return true;
}