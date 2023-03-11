#include "nn.h"

void conv2d(
	int out_f, int in_f,
	int kernelsize_h,
	int kernelsize_w,
	tensor img,
	tensor kernel_w, tensor kernel_b,
	tensor& output)
{
	int img_h = img.tensor_dim_size[1];
	int img_w = img.tensor_dim_size[2];
	int output_h = img_h - kernelsize_h + 1;
	int output_w = img_w - kernelsize_w + 1;
	int x_h = (kernelsize_h - 1) / 2;
	int x_w = (kernelsize_w - 1) / 2;
	int odd_h = 0, odd_w = 0;
	if (kernelsize_h % 2 == 0)
		odd_h = 1;
	if (kernelsize_w % 2 == 0)
		odd_w = 1;

	tensor_init_3d(output, out_f, output_h, output_w);

	for (int kernel_n = 0; kernel_n < out_f; kernel_n++)
		for (int img_i = x_h; img_i < img_h - x_h - odd_h; ++img_i)
			for (int img_j = x_w; img_j < img_w - x_w - odd_w; ++img_j)
			{
				int output_i = img_i - x_h;
				int output_j = img_j - x_w;

				t_e_3(output, kernel_n, output_i, output_j, t_v_1(kernel_b, kernel_n));
				for (int in_f_n = 0; in_f_n < in_f; ++in_f_n)
				{
					//C H W, Ãªµã(img_i, img_j)
					float o = 0;
					for (int kernel_i = 0; kernel_i < kernelsize_h; ++kernel_i)
						for (int kernel_j = 0; kernel_j < kernelsize_w; ++kernel_j)
						{
							int img_ii = img_i - x_h + kernel_i;
							int img_jj = img_j - x_w + kernel_j;
							o += t_v_3(img, in_f_n, img_ii, img_jj) * t_v_4(kernel_w, kernel_n, in_f_n, kernel_i, kernel_j);
						}
					t_a_3(output, kernel_n, output_i, output_j, o);
				}
			}
}

void relu(
	tensor img,
	tensor& output)
{
	output = img;
	int m = img.tensor_dim_size_mul[0] * img.tensor_dim_size[0];
	for (int i = 0; i < m; ++i)
		if (output.data[i] < 0)
			output.data[i] = 0;
}

void avgpool(
	int kernelsize,
	tensor img,
	tensor& output)
{
	int img_c = img.tensor_dim_size[0];
	int img_h = img.tensor_dim_size[1];
	int img_w = img.tensor_dim_size[2];

	int output_h = img_h / kernelsize;
	int output_w = img_w / kernelsize;

	tensor_init_3d(output, img_c, output_h, output_w);
	for (int cc = 0; cc < img_c; ++cc)
		for (int i = 0; i < img_h; i += kernelsize)
			for (int j = 0; j < img_w; j += kernelsize)
			{
				int output_i = i / 2;
				int output_j = j / 2;
				t_e_3(output, cc, output_i, output_j, 0);
				for (int a = 0; a < kernelsize; ++a)
					for (int b = 0; b < kernelsize; ++b)
						t_a_3(output, cc, output_i, output_j, t_v_3(img, cc, i + a, j + b));
				float t = t_v_3(output, cc, output_i, output_j) / 4;
				t_e_3(output, cc, output_i, output_j, t);
			}
}

void flatten(tensor img, tensor& output)
{
	int img_c = img.tensor_dim_size[0];
	int img_h = img.tensor_dim_size[1];
	int img_w = img.tensor_dim_size[2];
	vector<int> index;
	index_gen_1d(index, img_c * img_h * img_w);
	tensor_permute(img, 1, index, output);
}

void fc(
	int out_f, int in_f,
	tensor x, 
	tensor fc_w, 
	tensor fc_b,
	tensor& output)
{
	vector<int> index;
	index_gen_3d(index, 1, 1, x.tensor_dim_size[0]);
	tensor_permute(x, 3, index, x);

	index_gen_4d(index, fc_w.tensor_dim_size[0], 1, 1, fc_w.tensor_dim_size[1]);
	tensor_permute(fc_w, 4, index, fc_w);

	conv2d(out_f, 1, 1, in_f, x, fc_w, fc_b, output);
	index_gen_1d(index, output.tensor_dim_size[0]);
	tensor_permute(output, 1, index, output);
}