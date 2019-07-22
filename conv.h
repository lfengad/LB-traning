#ifndef CONV_H
#define CONV_H

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>
#include "ap_int.h"

typedef ap_int<8> char_;
typedef ap_int<32> int_;
typedef ap_int<64> long_;
typedef ap_uint<8> uchar_;
typedef ap_uint<1> ubit_;

#define img_num 128 
#define moments (int)4294967296 

char_ outdiff_qt( int_ outdiff, int s_outdif_pre, int s_out, ubit_ * larger, ubit_ *smaller, uchar_ *seed);

char_ input_qt( int_ input, int s_in_pre, int s_in, ubit_ * larger, ubit_ * smaller);

char_ bias_qt( int_ moment, char_ weight, int s_moment, int s_w_pre, int s_w, ubit_ * larger, ubit_ * smaller, int rate);

int_ momentb_qt( int_ moment_buf, int_ weight_buf, int s_moment, int s_moment_pre,  int s_weight, ubit_ * larger, ubit_ * smaller);

int_ moment_qt( int_ moment_buf, int_ weight_buf, int s_moment, int s_moment_pre,  int s_weight, ubit_ * larger, ubit_ * smaller);

char_ weight_qt( int_ moment, char_ weight, int s_moment, int s_w_pre, int s_w, ubit_ * larger, ubit_ * smaller, int rate, int decay );

ubit_ round_(ap_uint<8> value, ap_uint<8>* seed);

void memcpy_int(int_* to, int_* from, int size);

void memset_int(int_* to, int size);

void training(int_* data, char_* wg, float* fp, int* setting, int* sparams, int rate, int epoch, int decay);

void conv(int_ *input, char_ *weights,// char_ *bias,
    int_ *output, int *params, int* sparams);
void conv_back(int_ *outdiff, char_ *weights, int_ *indiff,
		int *params, int* sparams);

void weight_back(int_ *outdiff, char_ *weights, int_ *input,
		int_* moment, int *params, int* sparams,  int rate, int decay);
void bias_back(int_ *outdiff, char_ *bias,
		int_* moment, int *params, int* sparams, int rate);
void fc(int_ *input, char_ *weights, char_ *bias,
    int_ *output, int *params, int* sparams);
void back_fc(int_ *outdiff, char_ *weights,
    int_ *indiff, int *params, int* sparams);
void fc_weight(int_ *outdiff, char_ *weights,
	    int_ *input, int_* moment, int *params, int* sparams, int rate, int decay);
void bias_back_fc(int_ *outdiff, char_ *bias,
		int_* moment, int *params, int* sparams, int rate);
void pool(int_ *input, char_ *pos,
    int_ *output, int *params, int* sparams);
void back_pool(int_ *indiff, char_ *pos,
    int_ *outdiff0, int_ *outdiff1, int *params, int* sparams);
void relu(int_ *input,
    int_ *output, int *params, int* sparams);
void relu_back(int_ *input,
    int_ *outdiff0, int_* outdiff1, int_ *indiff, int *params, int* sparams);
void eltwise(int_ *input0, int_ *input1,
    int_ *output, int *params, int* sparams);
void eltwise_back(int_ *outdiff, int_ *indiff, int *params, int* sparams);

void batch(int_ *input, int_ *temp,
    int_ *output, int *params, int* sparams);
void batch_back(int_ *output, int_ *indiff, int_ *temp,
    int_ *outdiff,  int *params, int* sparams);
void scale(int_ *input, char_ *weights, char_ *bias,
    int_ *output, int *params, int* sparams) ;
void scale_back(int_ *indiff, char_ *weights,
	 int_ *outdiff, int *params, int* sparams);
void weight_scale(int_ *input, char_ *weights,
	 int_ *outdiff, int_* moment, int *params, int* sparams, int rate, int decay);
void bias_scale(char_ *bias, int_ *outdiff, int_* moment, int *params, int* sparams, int rate);
void softmax(int_ *input, char_ *output, int *params, int* sparams, int_* diff,
	float* tmp)  ;

#endif
