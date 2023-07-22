#include <iostream>
#include <vector>
#include <fstream>
#include <torch/script.h>

struct PointXYZ
{
	float x;
	float y;
	float z;
};

void pointCloudNormalize(std::vector<float>& points)
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

void resample(std::vector<float>& points, int nums)
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


void readPointCloud(std::string filepath, std::vector<PointXYZ>& pointcloud)
{
	try
	{
		std::ifstream infile;
		infile.open(filepath);
		if (infile.is_open())
		{
			PointXYZ point;
			float x, y, z;
			while (infile >> x >> y >> z)
			{
				point.x = x;
				point.y = y;
				point.z = z;

				pointcloud.push_back(point);
			}
			infile.close();
		}
	}
	catch (const std::exception&)
	{
		std::cerr << "加载点云失败!" << std::endl;
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


void classfier(std::vector<float> points, const std::string& model_path)
{
	//7500 --> （1,2500,3）张量
	torch::Tensor points_tensor = torch::from_blob(points.data(), { 1, 2500, 3 }, torch::kFloat);
	//std::cout << points_tensor << std::endl;
	points_tensor = points_tensor.to(torch::kCUDA);
	//（1,2500,3）-->（1,3,2500）
	points_tensor = points_tensor.permute({ 0, 2, 1 });
	//std::cout << points_tensor << std::endl;

	//加载模型
	torch::jit::script::Module module = torch::jit::load(model_path);
	module.eval();
	module.to(torch::kCUDA);
	std::cout << "success loading." << std::endl;

	auto outputs = module.forward({ points_tensor }).toTuple()->elements()[0].toTensor();
	std::cout << "outputs: " << outputs << std::endl;

	auto max_result = outputs.max(1, true);
	auto max_index = std::get<1>(max_result).item<int>();
	std::cout << "class index: " << max_index << std::endl;
}

void segmenter(std::vector<float> input_points, const std::string& model_path)
{
	torch::Tensor points_tensor = torch::from_blob(input_points.data(), { 1, 2500, 3 }, torch::kFloat);
	points_tensor = points_tensor.to(torch::kCUDA);
	points_tensor = points_tensor.permute({ 0,2,1 });

	torch::jit::script::Module module = torch::jit::load(model_path);
	module.to(torch::kCUDA);
	module.eval();

	std::vector<torch::jit::IValue> input;
	input.push_back(points_tensor);
	
	torch::Tensor outputs = module.forward(input).toTuple()->elements()[0].toTensor();
	std::cout << "outputs: " << outputs << std::endl;

	auto max_result = outputs.max(2, true);
	auto max_index = std::get<1>(max_result);
	std::cout << "class index: " << max_index << std::endl;

	std::cout << max_index[0][0] << std::endl;
	std::cout << "outputs.size: " << outputs.size(0) << " , " << outputs.size(1) << " , " << outputs.size(2) << std::endl;

	std::vector<std::vector<PointXYZ>> classes(outputs.size(2));
	for (int i = 0; i < outputs.size(1); i++)
	{
		int idx = max_index[0][i].item().toInt();
		PointXYZ point;
		point.x = input_points[i * 3];
		point.y = input_points[i * 3 + 1];
		point.z = input_points[i * 3 + 2];
		classes[idx].push_back(point);
	}
	std::size_t pos = model_path.find_last_of("/\\");
	std::string dir;
	std::string filename;
	if (pos != std::string::npos)
	{
		dir = model_path.substr(0, pos);
		filename = model_path.substr(pos + 1);
	}
	for (size_t i = 0; i < classes.size(); i++)
	{
		std::string newfile = dir + "\\seg-" + std::to_string(i) + ".txt";
		savePointCloud(classes[i], newfile);
	}
}




int main()
{
	std::vector<float> points;
	std::ifstream infile;
	float x, y, z;
	infile.open("..\\weights\\Airplane.txt");
	int point_num = 0;
	while (infile >> x >> y >> z)
	{
		points.push_back(x);
		points.push_back(y);
		points.push_back(z);
		++point_num;
	}
	infile.close();

	resample(points, 2500);

	pointCloudNormalize(points);

	std::string cls_model_path = "..\\weights\\classification_script_model_1.pt";
	classfier(points, cls_model_path);

	std::string seg_model_path = "..\\weights\\segmentation_model_Airplane_24.pt";
	segmenter(points, seg_model_path);

	system("pause");
	return 0;
}

int main0()
{
	try
	{
		torch::jit::script::Module module = torch::jit::load("..\\weights\\classification_script_model_1.pt");
		module.to(at::kCUDA);
		std::cout << "success loading." << std::endl;

		module.eval();//
		std::vector<torch::jit::IValue> inputs;
		inputs.push_back(torch::ones({ 1,3,2500 }).to(at::kCUDA));
		//torch::Tensor result = module.forward(inputs).toTensor(); //前向传播获取结果，还是tensor类型
		//模型返回多个结果，用toTuple,其中elements()[i-1]获取第i个返回值
		torch::Tensor output = module.forward(inputs).toTuple()->elements()[0].toTensor();
		std::cout << output << std::endl;

		auto max_result = output.max(1, true);
		auto max_index = std::get<1>(max_result).item<int>();
		std::cout << max_index << std::endl;

	}
	catch (...)
	{
		std::cout << "error loading the model!" << std::endl;
		return -1;
	}
	return 0;

}