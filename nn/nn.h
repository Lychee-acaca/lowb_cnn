#ifndef  NN_H
#define  NN_H

#include <vector>
#include "tensor.h"

using namespace std;

void conv2d(
	int out_f, int in_f,
	int kernelsize_h,
	int kernelsize_w,
	tensor img,
	tensor kernel_w, tensor kernel_b,
	tensor& output);

void relu(
	tensor img,
	tensor& output);

void avgpool(
	int kernelsize,
	tensor img,
	tensor& output);

void flatten(tensor img, tensor& output);

void fc(
	int out_f, int in_f,
	tensor x,
	tensor fc_w,
	tensor fc_b,
	tensor& output);

#endif
