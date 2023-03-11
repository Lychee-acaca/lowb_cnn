#include "tensor.h"

void index_gen_1d(
	vector<int>& index,
	int id_1)
{
	index.resize(0, 0);
	index.push_back(id_1);
}

void index_gen_2d(
	vector<int>& index,
	int id_1, int id_2)
{
	index.resize(0, 0);
	index.push_back(id_1);
	index.push_back(id_2);
}

void index_gen_3d(
	vector<int>& index,
	int id_1, int id_2, int id_3)
{
	index.resize(0, 0);
	index.push_back(id_1);
	index.push_back(id_2);
	index.push_back(id_3);
}

void index_gen_4d(
	vector<int>& index,
	int id_1, int id_2, int id_3, int id_4)
{
	index.resize(0, 0);
	index.push_back(id_1);
	index.push_back(id_2);
	index.push_back(id_3);
	index.push_back(id_4);
}

/*
* 返回多维张量对应的一维vector中的位置
* index的长度必须和维度相等
* 例如index={3, 1, 2, 5}，那么就相当于访问t[3][1][2][5]
*/
int tensor_index(
	tensor t, 
	vector<int> index)
{
	int t_id = 0;
	for (int i = 0; i < t.dims; ++i)
		t_id += index[i] * t.tensor_dim_size_mul[i];
	return t_id;
}

float tensor_value(
	tensor t,
	vector<int> index)
{
	return t.data[tensor_index(t, index)];
}

float tensor_value_1d(
	tensor t,
	int id_1)
{
	vector<int> index;
	index_gen_1d(index, id_1);
	return tensor_value(t, index);
}

float tensor_value_2d(
	tensor t,
	int id_1, int id_2)
{
	vector<int> index;
	index_gen_2d(index, id_1, id_2);
	return tensor_value(t, index);
}

float tensor_value_3d(
	tensor t,
	int id_1, int id_2, int id_3)
{
	vector<int> index;
	index_gen_3d(index, id_1, id_2, id_3);
	return tensor_value(t, index);
}

float tensor_value_4d(
	tensor t,
	int id_1, int id_2, int id_3, int id_4)
{
	vector<int> index;
	index_gen_4d(index, id_1, id_2, id_3, id_4);
	return tensor_value(t, index);
}

void tensor_edit(
	tensor& t,
	vector<int> index, 
	float data)
{
	t.data[tensor_index(t, index)] = data;
}

void tensor_edit_1d(
	tensor& t,
	int id_1,
	float data)
{
	vector<int> index;
	index_gen_1d(index, id_1);
	tensor_edit(t, index, data);
}

void tensor_edit_2d(
	tensor& t,
	int id_1, int id_2,
	float data)
{
	vector<int> index;
	index_gen_2d(index, id_1, id_2);
	tensor_edit(t, index, data);
}

void tensor_edit_3d(
	tensor& t,
	int id_1, int id_2, int id_3,
	float data)
{
	vector<int> index;
	index_gen_3d(index, id_1, id_2, id_3);
	tensor_edit(t, index, data);
}

void tensor_edit_4d(
	tensor& t,
	int id_1, int id_2, int id_3, int id_4,
	float data)
{
	vector<int> index;
	index_gen_4d(index, id_1, id_2, id_3, id_4);
	tensor_edit(t, index, data);
}

void tensor_add(
	tensor& t,
	vector<int> index,
	float data)
{
	t.data[tensor_index(t, index)] += data;
}

void tensor_add_1d(
	tensor& t,
	int id_1,
	float data)
{
	vector<int> index;
	index_gen_1d(index, id_1);
	tensor_add(t, index, data);
}

void tensor_add_2d(
	tensor& t,
	int id_1, int id_2,
	float data)
{
	vector<int> index;
	index_gen_2d(index, id_1, id_2);
	tensor_add(t, index, data);
}

void tensor_add_3d(
	tensor& t,
	int id_1, int id_2, int id_3,
	float data)
{
	vector<int> index;
	index_gen_3d(index, id_1, id_2, id_3);
	tensor_add(t, index, data);
}

void tensor_add_4d(
	tensor& t,
	int id_1, int id_2, int id_3, int id_4,
	float data)
{
	vector<int> index;
	index_gen_4d(index, id_1, id_2, id_3, id_4);
	tensor_add(t, index, data);
}

void tensor_init(
	int dims,
	tensor& t,
	vector<int> index)
{
	t.dims = dims;
	t.tensor_dim_size = index;

	t.tensor_dim_size_mul.resize(dims, 0);
	t.tensor_dim_size_mul[dims - 1] = 1;

	for (int i = dims - 2; i >= 0; --i)
		t.tensor_dim_size_mul[i] = t.tensor_dim_size_mul[i + 1] * index[i + 1];
	t.data.resize(t.tensor_dim_size_mul[0] * index[0], 0);
}

void tensor_init_1d(
	tensor& t,
	int size_1)
{
	vector<int> index;
	index_gen_1d(index, size_1);
	tensor_init(1, t, index);
}

void tensor_init_2d(
	tensor& t,
	int size_1, int size_2)
{
	vector<int> index;
	index_gen_2d(index, size_1, size_2);
	tensor_init(2, t, index);
}

void tensor_init_3d(
	tensor& t,
	int size_1, int size_2, int size_3)
{
	vector<int> index;
	index_gen_3d(index, size_1, size_2, size_3);
	tensor_init(3, t, index);
}

void tensor_init_4d(
	tensor& t,
	int size_1, int size_2, int size_3, int size_4)
{
	vector<int> index;
	index_gen_4d(index, size_1, size_2, size_3, size_4);
	tensor_init(4, t, index);
}

void tensor_permute(
	tensor t,
	int dims,
	vector<int> index,
	tensor& output)
{
	tensor_init(dims, output, index);
	output.data = t.data;
}