#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>
#include "ap_int.h"
#include "math.h"
#include "conv.h"


#define PARA 16
//#define img_num 32

void relu(int_ *input,
    int_ *output, int *params, int* sparams )
{
	//char_ weight_buf[PARA][64];
	int_ in_buf[PARA][img_num];
  //  char_ bias_buf[PARA][2048/PARA];
    int_ out_buf[PARA][img_num];
  //  int_ tmp[PARA*128];
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
//	int rate = params[16];
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

         // printf("x_dim %d y_dim %d %d %d\n", x_dim, y_dim, params[4], params[5]);

	     for(int o = 0; o<iteration_out; o++)
	    	for(int y =0; y < y_dim; y++)
	    	  for(int x = 0; x < x_dim;x++)
	    	 {
                     #pragma omp parallel for  
	    	     for(int p=0;p<PARA; p++)
	    	     {
                        
	    	    	 int in_trans_size = img_num;
	    	    	 int in_off = ((y*x_dim+x) * inChannel + o*PARA + p)*img_num;
//                        printf("here 0\n"); 	    	    	
                         memcpy(in_buf[p], input + in_off, 4*in_trans_size);
  //                      printf("here 1\n"); 	    	    	

	    	    	for(int im = 0; im<img_num;im++)
	    	    	{
	    	    	  int_ ip = in_buf[p][im];
	    	    	  if(ip.range(31,31))
	    	    	  out_buf[p][im] = 0;
	    	    	  else
	    	    	  out_buf[p][im] = ip;
	    	    	}

	    	   	 int out_trans_size = img_num;
			 int out_off = ((y*x_dim+x) * inChannel + o*PARA + p)*img_num;
                       // printf("here 2 %d %d %d %d %d %d\n", y, x_dim, x, inChannel, o, p); 	    	    	
			 memcpy(output + out_off, out_buf[p], 4*in_trans_size);
 //                       printf("here 3\n"); 	    	    	
	    	     }
	    	 }
	    	 sparams[17-11] = s_in_pre;
}


void relu_back(int_ *input,
    int_ *outdiff0, int_* outdiff1, int_ *indiff, int *params, int* sparams)
{
	//char_ weight_buf[PARA][64];
	int_ in_buf[PARA][img_num];
    int_ tmp[PARA];
    int_ out_buf[PARA];
    int_ out0_buf[PARA];
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
	int dup = params[7];
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
  int s_outdif_pre0 = params[21];*/
//	int rate = params[16];
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
  int s_outdif_pre0 = sparams[10];



	     for(int o = 0; o<iteration_out; o++)
	    	for(int y =0; y < y_dim; y++)
	    	  for(int x = 0; x < x_dim;x++)
	    	 {
	    	    int out_trans_size = PARA;
	    	    int out_off = ((y*x_dim+x) * inChannel + o*PARA);
	    	    memcpy(out_buf, outdiff0 + out_off, 4*out_trans_size);
	    	    if(dup)
	    	    memcpy(out0_buf, outdiff1 + out_off, 4*out_trans_size);


	    		int in_trans_size = PARA*img_num;
	    		int in_off = ((y*x_dim+x) * inChannel + o*PARA)*img_num;
	    		memcpy(in_buf, input + in_off, 4*in_trans_size);
                   
                    #pragma omp parallel for
	    	    for(int p=0;p<PARA; p++)
	    	     {
	    			int_ tmpout;
					if(dup)
                                           {
                                               if(s_outdif_pre>s_outdif_pre0)  
						tmpout = out_buf[p] + out0_buf[p]<<(s_outdif_pre-s_outdif_pre0);
                                              else
						tmpout = out_buf[p] + out0_buf[p]<<(-(s_outdif_pre-s_outdif_pre0));
				           }
                                 	else
						tmpout = out_buf[p];
                    tmp[p] = 0;
	    	    	for(int im = 0; im<img_num;im++)
	    	    	{
	    	    	  int_ ip = in_buf[p][im];

	    	    	  if(ip>0)
	    	    	  tmp[p] += tmpout;
	    	    	}
	    	    	tmp[p]/=img_num;
	    	     }
	    	   	// int out_trans_size = PARA;
				// int out_off = ((y*x_dim+x) * inChannel + o*PARA);
				 memcpy(indiff + out_off, tmp, 4*out_trans_size);
	    	   }

	    	   sparams[19-11] = s_outdif_pre;
}



