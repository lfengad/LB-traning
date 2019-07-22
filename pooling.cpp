#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>
#include "ap_int.h"
#include "math.h"
#include "conv.h"


#define PARA 16
//#define img_num 32

void pool(int_ *input, char_ *pos,
    int_ *output, int *params, int * sparams)
{
	char_ weight_buf[PARA][64];
	int_ in_buf[64][PARA*img_num];
  //  char_ bias_buf[PARA][2048/PARA];
    int_ tmp[PARA*128];
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
	/*int s_in = params[11]; 
	int s_w = params[12];
	int s_out = params[13];
	int s_b = params[14];
	int s_in_pre = params[15];
	  int s_w_pre = params[16];
	  int s_out_pre = params[17];
	  int s_b_pre = params[18]; 
	  int s_indif_pre = params[19];
	  int s_outdif_pre = params[20];*/
 // int s_in_pre0 = params[21];
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



	int x_out = ((x_dim - ksize) / stride) + 1;
    int y_out = x_out;
     

	     for(int o = 0; o<iteration_out; o++)
	    	for(int y =0; y < y_out; y++)
	    	  for(int x = 0; x < x_out;x++)
	    	 {
	            for(int p=0;p<ksize;p++)
	            	for(int q=0;q<ksize;q++)
	            	{
	            	int x_in = x*stride+q;
					int y_in = y*stride+p;
					int inoff = (p*ksize+q);
				    int insize = PARA*img_num;
					if((stride + q < ksize) && x != 0)
					{
					   memcpy_int(in_buf[inoff], in_buf[p*ksize+stride + q], insize);
					}
					else
					{
				       int in_off = ((y_in*x_dim+x_in) * inChannel + o*PARA )*img_num;
					   memcpy_int(in_buf[inoff], input + in_off,  insize);
                                        //  for(int dd = 0 ;dd<insize; dd++)
                                         //  {printf(" %d ", *((int*)(input+in_off+dd)));}
                                           
     
					}
	            	}
                  
   
	          for(int p=0;p<PARA;p++)
                    #pragma omp parallel for
	    	    for(int im=0;im<img_num;im++)
	    	    {
	    	    int_ max = 0;
	    	    char_ idx = 0;
	    	    ap_int<64> sum = 0;
	            for(int m=0;m<ksize;m++)
	              for(int n=0;n<ksize;n++)
	             {
	               int_ ip = in_buf[m*ksize+n][p*img_num+im];
                      // printf(" %d ", (int)ip);
	               if(maxpool)
	               {if(ip>max)
	                 {
	            	   idx = m*ksize+n;
	            	   max = ip;
	                 }
	               }
	               else
	               {
	            	   sum += ip;
	               }
	             }
                    // printf("\n");
	             int out_off = ((y*x_out+x)*outChannel + o*PARA + p)*img_num+im;

	             if(maxpool)
	             {
	             output[out_off] = max;
	             pos[out_off] = idx;
	             }
	             else
                     {
	              output[out_off] = sum/ksize/ksize;
                     // printf(" %d ", (int)output[out_off]);

                     }
	    	   }

	         }
	         sparams[17-11] = s_in_pre;
}


void back_pool(int_ *indiff, char_ *pos,
    int_ *outdiff0, int_ *outdiff1, int *params, int* sparams)
{
//	char_ weight_buf[PARA][64];
	int_ out_buf[PARA];
	int_ out0_buf[PARA];
	char_ pos_buf[PARA][img_num];
   // char_ bias_buf[PARA][2048/PARA];
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
  int s_outdif_pre0 = params[21];*/


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

//	int rate = params[16];
	int x_out = ((x_dim - ksize) / stride) + 1;
    int y_out = x_out;


	     for(int o = 0; o<iteration_out; o++)
	    	for(int y =0; y < y_out; y++)
	    	  for(int x = 0; x < x_out;x++)
	    	 {
	    		int out_trans_size = PARA;
	    		int out_offset = ((y*x_out+x)*outChannel + o*PARA);
	    		memcpy(out_buf, outdiff0+out_offset, 4*out_trans_size);
	    		if(dup)
	    		{
	    		 memcpy(out0_buf, outdiff1+out_offset, 4*out_trans_size);
	    		}

	    		//if(maxpool)
	    		memcpy(pos_buf, pos+out_offset, out_trans_size*img_num);
                        #pragma omp parallel for
	    		for(int p =0;p<PARA;p++)
	    		{
	    			int_ tmpout;
					if(dup)
                                             { if( s_outdif_pre > s_outdif_pre0)
						tmpout = out_buf[p] + out0_buf[p]<<(s_outdif_pre-s_outdif_pre0);
						else
                                                tmpout = out_buf[p] + out0_buf[p]>>(s_outdif_pre0-s_outdif_pre);
					     }
                                        else
						tmpout = out_buf[p];

	    		 if(maxpool)
	    		 { 	int_ tmp[49];
	    		        memset_int(tmp,  49);
	    			for(int im=0;im<img_num;im++)
	    			{
	    			int idx = pos_buf[p][im];
	    			tmp[idx] += tmpout;
	    			}
                    for(int m=0; m<ksize; m++)
                       for(int n=0; n<ksize; n++)
                        {
       	    			int x_in = x*stride + n;
       	    			int y_in = y*stride + m;
    	    			int in_off = (y_in*x_dim+x_in) * inChannel + o*PARA+p;
    	    		        if (((stride+ n) <ksize && x!= 0)  || ((stride +m) <ksize && y!=0) )
				indiff[in_off] += tmp[m*ksize+n]/img_num;
    	    			else
    	    			indiff[in_off] = tmp[m*ksize+n]/img_num;
                        }
	    		 }
	    		 else
	    			 for(int m=0;m<ksize;m++)
	    				for(int n=0;n<ksize;n++)
	    				{
	    					int x_in = x*stride + n;
	    					int y_in = y*stride + m;
	    					int in_off = (y_in*x_dim+x_in) * inChannel + o*PARA+p;
	    					if (((stride+ n) <ksize && x!= 0)  || ((stride +m) <ksize && y!=0) )
	    					indiff[in_off] += tmpout /ksize/ksize;
	    					else
	    					indiff[in_off] = tmpout /ksize/ksize;
	    				}
	    		}
	    	 }


    sparams[19-11] = s_outdif_pre;
}
