#define _CRT_SECURE_NO_WARNINGS 

#include <iostream>
#include <opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include "./nn/nn.h"
#include "./nn/tensor.h"

using namespace std;
using namespace cv;

//totally 44426 param
tensor conv1_w; //[6, 1, 5, 5] 150param
tensor conv1_b; //[6] 6param
tensor conv2_w; //[16, 6, 5, 5] 2400param
tensor conv2_b; //[16] 16param
tensor fc1_w; //[120, 256] 30720param
tensor fc1_b; //[120] 120param
tensor fc2_w; //[84, 120] 10080param
tensor fc2_b; //[84] 84param
tensor fc3_w; //[10, 84] 840param
tensor fc3_b; //[10] 10param

tensor img;

void read_img(String path)
{
	Mat input = imread(path, 0);
	imshow("input", input);
	waitKey(0);

	tensor_init_3d(img, 1, 28, 28);
	for (int i = 0; i < 28; ++i)
		for (int j = 0; j < 28; ++j)
			t_e_3(img, 0, i, j, (float)input.at<uchar>(i, j) / 255);
}

void read_d(int dims, tensor& t, vector<int> index)
{
	tensor_init(dims, t, index);
	for (int i = 0; i < index[0]; ++i)
	{
		if (dims > 1)
		{
			tensor tt;
			vector<int> id_next = index;
			id_next.erase(id_next.begin());
			read_d(dims - 1, tt, id_next);
			for (int j = t.tensor_dim_size_mul[0] * i; j < t.tensor_dim_size_mul[0] * (i + 1); ++j)
				t.data[j] = tt.data[j - t.tensor_dim_size_mul[0] * i];
		}
		else
		{
			float f;
			scanf("%f", &f);
			t_e_1(t, i, f);
		}
	}
}

void read_wb()
{
	freopen("wb.in", "r", stdin);

	vector<int> index;
	index_gen_4d(index, 6, 1, 5, 5);
	read_d(4, conv1_w, index);
	
	index_gen_1d(index, 6);
	read_d(1, conv1_b, index);

	index_gen_4d(index, 16, 6, 5, 5);
	read_d(4, conv2_w, index);

	index_gen_1d(index, 16);
	read_d(1, conv2_b, index);

	index_gen_2d(index, 120, 256);
	read_d(2, fc1_w, index);

	index_gen_1d(index, 120);
	read_d(1, fc1_b, index);

	index_gen_2d(index, 84, 120);
	read_d(2, fc2_w, index);

	index_gen_1d(index, 84);
	read_d(1, fc2_b, index);

	index_gen_2d(index, 10, 84);
	read_d(2, fc3_w, index);

	index_gen_1d(index, 10);
	read_d(1, fc3_b, index);
}

int main()
{
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_ERROR);
	read_wb();
	read_img("./testdata/test2.jpg");

	tensor x;

	conv2d(6, 1, 5, 5, img, conv1_w, conv1_b, x);
	relu(x, x);
	avgpool(2, x, x);

	conv2d(16, 6, 5, 5, x, conv2_w, conv2_b, x);
	relu(x, x);
	avgpool(2, x, x);
	
	flatten(x, x);
	
	fc(120, 256, x, fc1_w, fc1_b, x);
	relu(x, x);

	fc(84, 120, x, fc2_w, fc2_b, x);
	relu(x, x);
	
	fc(10, 84, x, fc3_w, fc3_b, x);
	
	/*
	for (int j = 0; j < 10; ++j)
		printf("%.3f ", t_v_1(x, j));
	*/

	int maxi = 0;
	for (int i = 0; i < 10; ++i)
		if (t_v_1(x, maxi) < t_v_1(x, i))
			maxi = i;

	printf("%d", maxi);
		
}