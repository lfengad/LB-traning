#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>
#include "ap_int.h"
#include "math.h"
#include "conv.h"


#define PARA 16
//#define img_num 32

void eltwise(int_ *input0, int_ *input1,
    int_ *output, int *params,  int* sparams)
{
	//char_ weight_buf[PARA][64];
	int_ in_buf0[PARA][img_num];
	int_ in_buf1[PARA][img_num];
  //  char_ bias_buf[PARA][2048/PARA];
    int_ out_buf[PARA][img_num];
    //int_ tmp[PARA*128];
    ubit_ larger = 0;
    ubit_ smaller = 1;

	int inChannel = params[0];
	int outChannel = params[1];
	int inCh_once = params[2];
	int outCh_once = params[3];
	int iteration_out = outChannel/PARA;
	int x_dim = params[4];
	int y_dim = params[5];
	int stride = params[6];
	int pad = params[7];
	int ksize = params[8];
	int maxpool = params[9];
	//int img_num = params[10];
/*	int s_in = params[11]; 
	int s_w = params[12];
	int s_out = params[13];
	int s_b = params[14];
	int s_in_pre = params[15];
  int s_w_pre = params[16];
  int s_out_pre = params[17];
  int s_b_pre = params[18]; 
  int s_indif_pre = params[19];
  int s_outdif_pre = params[20];
  int s_in_pre0 = params[21];*/
	//int rate = params[16];
  int s_in = sparams[0]; 
  int s_w = sparams[1];
  int s_out = sparams[2];
  int s_b = sparams[3];
  int s_in_pre = sparams[4];
  int s_w_pre = sparams[5];
  int s_out_pre = sparams[6];
  int s_b_pre = sparams[7]; 
  int s_indif_pre = sparams[8];
  int s_outdif_pre = sparams[9];
  int s_in_pre0 = sparams[10];



	 for(int o = 0; o<iteration_out; o++)
	    for(int y =0; y < y_dim; y++)
	    	for(int x = 0; x < x_dim;x++)
	       {
                   #pragma omp parallel for
	    	   for(int p=0;p<PARA; p++)
	    	   {
	    	    	 int in_trans_size = img_num;
	    	    	 int in_off = ((y*x_dim+x) * inChannel + o*PARA + p)*img_num;
	    	    	 memcpy(in_buf0[p], input0 + in_off, 4*in_trans_size);
	    	    	 memcpy(in_buf1[p], input1 + in_off, 4*in_trans_size);

	    	    	for(int im = 0; im<img_num;im++)
	    	    	{
                         if(s_in_pre>s_in_pre0)
	    	    	 out_buf[p][im] = in_buf0[p][im]+in_buf1[p][im]<<(s_in_pre-s_in_pre0);
	    	    	 else
                         out_buf[p][im] = in_buf0[p][im]+in_buf1[p][im]>>(s_in_pre0-s_in_pre);
	    	    	}

	    	   	 int out_trans_size = img_num;
		         int out_off = ((y*x_dim+x) * inChannel + o*PARA + p)*img_num;
		         memcpy( output + out_off, out_buf[p], 4*in_trans_size);
	    	     }
	    	 }

	    	 sparams[17-11] = s_in_pre;
}


void eltwise_back(int_ *outdiff, int_ *indiff, int *params, int*sparams)
{
	//char_ weight_buf[PARA][64];
	//int_ in_buf[PARA][img_num];
    //int_ tmp[PARA];
    //int_ out_buf[PARA];
    ubit_ larger = 0;
    ubit_ smaller = 1;

	int inChannel = params[0];
	int outChannel = params[1];
	int inCh_once = params[2];
	int outCh_once = params[3];
	int iteration_out = outChannel/PARA;
	int x_dim = params[4];
	int y_dim = params[5];
	int stride = params[6];
	int pad = params[7];
	int ksize = params[8];
	int maxpool = params[9];
//	int img_num = params[10];
	/*int s_in = params[11]; 
	int s_w = params[12];
	int s_out = params[13];
	int s_b = params[14];
	int s_in_pre = params[15];
	  int s_w_pre = params[16];
	  int s_out_pre = params[17];
	  int s_b_pre = params[18]; 
	  int s_indif_pre = params[19];
	  int s_outdif_pre = params[20];
	  int s_in_pre0 = params[21];*/

  int s_in = sparams[0]; 
  int s_w = sparams[1];
  int s_out = sparams[2];
  int s_b = sparams[3];
  int s_in_pre = sparams[4];
  int s_w_pre = sparams[5];
  int s_out_pre = sparams[6];
  int s_b_pre = sparams[7]; 
  int s_indif_pre = sparams[8];
  int s_outdif_pre = sparams[9];
  int s_in_pre0 = sparams[10];

//	int rate = params[16];


	     for(int o = 0; o<iteration_out; o++)
	    	for(int y =0; y < y_dim; y++)
	    	  for(int x = 0; x < x_dim;x++)
	    	 {
	    	    int out_trans_size = PARA;
	    	    int out_off = ((y*x_dim+x) * inChannel + o*PARA);
	    	    memcpy_int(indiff+out_off, outdiff + out_off, out_trans_size);
	         }
        
        sparams[19-11] = s_outdif_pre;

}
