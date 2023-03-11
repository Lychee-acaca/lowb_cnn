#ifndef  TENSOR_H
#define  TENSOR_H

#include <vector>

using namespace std;

struct tensor
{
	int dims; //张量的维度
	vector<int> tensor_dim_size; //张量的每个维度的大小
	vector<int> tensor_dim_size_mul; //张量的i+1维后所有维度大小乘积
	vector<float> data; //数据
};

void index_gen_1d(
	vector<int>& index,
	int id_1);

void index_gen_2d(
	vector<int>& index,
	int id_1, int id_2);

void index_gen_3d(
	vector<int>& index,
	int id_1, int id_2, int id_3);

void index_gen_4d(
	vector<int>& index,
	int id_1, int id_2, int id_3, int id_4);

#define tensor_index t_id
int tensor_index(
	tensor t,
	vector<int> index);

#define tensor_value t_v
float tensor_value(
	tensor t,
	vector<int> index);

#define tensor_value_1d t_v_1
float tensor_value_1d(
	tensor t,
	int id_1);

#define tensor_value_2d t_v_2
float tensor_value_2d(
	tensor t,
	int id_1, int id_2);

#define tensor_value_3d t_v_3
float tensor_value_3d(
	tensor t,
	int id_1, int id_2, int id_3);

#define tensor_value_4d t_v_4
float tensor_value_4d(
	tensor t,
	int id_1, int id_2, int id_3, int id_4);

#define tensor_edit t_e
void tensor_edit(
	tensor& t,
	vector<int> index,
	float data);

#define tensor_edit_1d t_e_1
void tensor_edit_1d(
	tensor& t,
	int id_1,
	float data);

#define tensor_edit_2d t_e_2
void tensor_edit_2d(
	tensor& t,
	int id_1, int id_2,
	float data);

#define tensor_edit_3d t_e_3
void tensor_edit_3d(
	tensor& t,
	int id_1, int id_2, int id_3,
	float data);

#define tensor_edit_4d t_e_4
void tensor_edit_4d(
	tensor& t,
	int id_1, int id_2, int id_3, int id_4,
	float data);

#define tensor_add t_a
void tensor_add(
	tensor& t,
	vector<int> index,
	float data);

#define tensor_add_1d t_a_1
void tensor_add_1d(
	tensor& t,
	int id_1,
	float data);

#define tensor_add_2d t_a_2
void tensor_add_2d(
	tensor& t,
	int id_1, int id_2,
	float data);

#define tensor_add_3d t_a_3
void tensor_add_3d(
	tensor& t,
	int id_1, int id_2, int id_3,
	float data);

#define tensor_add_4d t_a_4
void tensor_add_4d(
	tensor& t,
	int id_1, int id_2, int id_3, int id_4,
	float data);

void tensor_init(
	int dims,
	tensor& t,
	vector<int> index);

void tensor_init_1d(
	tensor& t,
	int size_1);

void tensor_init_2d(
	tensor& t,
	int size_1, int size_2);

void tensor_init_3d(
	tensor& t,
	int size_1, int size_2, int size_3);

void tensor_init_4d(
	tensor& t,
	int size_1, int size_2, int size_3, int size_4);

void tensor_permute(
	tensor t,
	int dims,
	vector<int> index,
	tensor& output);

#endif