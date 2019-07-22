#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>
#include "ap_int.h"
#include "math.h"
#include "conv.h"

#define PARA 16
//#define img_num 32

ap_uint<8> seed_sc = 255;

void scale(int_ *input, char_ *weights, char_ *bias,
    int_ *output, int *params, int* sparams) {

	char_ weight_buf[PARA];
	char_ bias_buf[PARA];
	int_ in_buf[PARA][img_num];
	int_ out_buf[PARA][img_num];
    int_ factor[PARA];
    long_ sum[PARA];
    int_ mean[PARA];
    int_ var[PARA];
    int_ out_tmp[PARA][img_num];
    ubit_ larger = 0;
    ubit_ smaller = 1;

	int inChannel = params[0];
	int outChannel = params[1];
	int inCh_once = params[2];
	int outCh_once = params[3];
	int iteration_in = inChannel/inCh_once;
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


     for(int i = 0; i<iteration_in; i++)
    {
     int wg_trans_size= PARA;
     int wg_offset = i*PARA;
     memcpy(weight_buf, weights + wg_offset, wg_trans_size);
     memcpy(bias_buf, bias + wg_offset, wg_trans_size);
    #pragma omp parallel for
    for(int p=0;p<PARA;p++)
    {
    	for(int y =0; y < y_dim; y++)
    	  for(int x = 0; x < x_dim;x++)
    	 {
    	    	int in_trans_size = img_num;
    		    int in_offset = ((y*x_dim+x)*inChannel+i*PARA+p)*img_num;
                    for(int im=0; im<img_num;im++)
    		    {
				in_buf[p][im] = input_qt(input[in_offset+im], s_in_pre, s_in, &larger, &smaller);
				input[in_offset+im] = in_buf[p][im];
    		    }

    		  //  memcpy(in_buf[p], input+in_offset, in_trans_size);

    	        for(int im=0;im<img_num;im++)
    	        {
                 if(s_in + s_w_pre > s_b_pre)                    
    	           out_buf[p][im] = in_buf[p][im]*weight_buf[p] + (bias_buf[p]<<(s_in+s_w_pre-s_b_pre));
                 else 
    	           out_buf[p][im] = in_buf[p][im]*weight_buf[p] + (bias_buf[p]>>(-s_in-s_w_pre+s_b_pre));
    	        }
    	        memcpy(output + in_offset, out_buf[p], 4*in_trans_size);
    	 }

    }
}

    sparams[17-11] = s_in + s_w_pre;

     if(larger)
     	s_in = s_in - 1;
     if(smaller)
     	s_in = s_in + 1;

     sparams[11-11] = s_in;

}

 void scale_back(int_ *indiff, char_ *weights,
	 int_ *outdiff, int *params, int* sparams) {

	char_ weight_buf[PARA];
	char_ bias_buf[PARA];
	int_ in_buf[PARA];
	int_ out_buf[PARA];
	 int_ factor[PARA];
	 long_ sum[PARA];
	 int_ mean[PARA];
	 int_ var[PARA];
	 int_ out_tmp[PARA][img_num];
	 ubit_ larger = 0;
	 ubit_ smaller = 1;

	int inChannel = params[0];
	int outChannel = params[1];
	int inCh_once = params[2];
	int outCh_once = params[3];
	int iteration_in = inChannel/inCh_once;
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
  int s_outdif_pre = params[20];*/
  //int s_in_pre0 = params[21];
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


          printf("***************************************scale %d %d %d\n", x_dim, y_dim, iteration_in);
	  for(int i = 0; i<iteration_in; i++)
	 {
	  int wg_trans_size= PARA;
	  int wg_offset = i*PARA;
	  memcpy(weight_buf, weights + wg_offset, wg_trans_size);
          #pragma omp parallel for
	  for(int p=0;p<PARA;p++)
		for(int y =0; y < y_dim; y++)
		  for(int x = 0; x < x_dim;x++)
		 {
				int in_trans_size = 1;
				int in_offset = ((y*x_dim+x)*inChannel+i*PARA+p);
				  out_buf[p] = outdiff_qt(outdiff[in_offset], s_outdif_pre, s_out, &larger, &smaller, & seed_sc);
                                 // printf(" %d-scale:%d-%d-%lld-%lld-%lld ", (int)out_buf[p],(int)weight_buf[p],(int)outdiff[in_offset], (long long)(outdiff[in_offset]*(s_out)*512/(s_outdif_pre)), s_out, s_outdif_pre);
				  outdiff[in_offset] = out_buf[p];

			  //  memcpy(in_buf[p], input+in_offset, in_trans_size);
				indiff[in_offset] = out_buf[p]*weight_buf[p] ;

			//	memcpy(indiff + in_offset, in_buf[p], in_trans_size);
		 }
 //             printf("\n");    
	 }
     sparams[19-11] = s_w_pre + s_out;

	   if(larger)
	   	s_out = s_out - 1;
	   if(smaller)
	   	s_out = s_out + 1;


		sparams[13-11] = s_out;

}


 void weight_scale(int_ *input, char_ *weights, 
	 int_ *outdiff, int_ * moment, int *params, int *sparams, int rate, int decay) {

	char_ weight_buf[PARA];
	char_ bias_buf[PARA];
	int_ in_buf[PARA][img_num];
	int_ moment_buf[PARA];
	int_ out_buf[PARA];
	int_ factor[PARA];
	int_ sum[PARA];
	int_ mean[PARA];
	int_ var[PARA];
	int_ out_tmp[PARA][img_num];
	ubit_ larger = 0;
	ubit_ smaller = 1;
	ubit_ larger_m = 0;
	ubit_ smaller_m = 1;

	int inChannel = params[0];
	int outChannel = params[1];
	int inCh_once = params[2];
	int outCh_once = params[3];
	int iteration_in = inChannel/inCh_once;
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
  int s_wmoment = sparams[11]; 
  int s_wmoment_pre = sparams[12]; 
  int s_bmoment = sparams[13]; 
  int s_bmoment_pre = sparams[14]; 


	  for(int i = 0; i<iteration_in; i++)
	 {
	  int wg_trans_size= PARA;
	  int wg_offset = i*PARA;
	  memcpy(weight_buf, weights + wg_offset, wg_trans_size);
	  memcpy(moment_buf, moment + wg_offset, wg_trans_size*4);

         #pragma omp parallel for
	 for(int p=0;p<PARA;p++)
	 {
		for(int y =0; y < y_dim; y++)
		  for(int x = 0; x < x_dim;x++)
		 {
				int in_trans_size = img_num;
				int in_offset = ((y*x_dim+x)*inChannel+i*PARA+p)*img_num;
			    memcpy(in_buf[p], input+in_offset, in_trans_size*4);
			    out_buf[p] = outdiff[(y*x_dim+x)*inChannel+i*PARA+p];

				for(int im=0;im<img_num;im++)
				{
				   if(im == 0 && x == 0 && y == 0)
				   sum[p] = out_buf[p]*in_buf[p][im];
				   else	
				   sum[p]+= out_buf[p]*in_buf[p][im];
				}
		 }
         

          moment_buf[p] = moment_qt( moment_buf[p], sum[p], s_wmoment, s_wmoment_pre, s_in_pre+ s_outdif_pre, &larger_m, &smaller_m );

          weights[i*PARA+p] = weight_qt(moment_buf[p], weight_buf[p],s_wmoment, s_w_pre, s_w, &larger, &smaller, rate, decay);

	 }

	 memcpy(moment + wg_offset, moment_buf, wg_trans_size*4);
 }

          sparams[16-11]= s_w;
          sparams[12] = s_wmoment;
	   if(larger)
	   	s_w = s_w - 1;
	   if(smaller)
	   	s_w = s_w + 1;

   if(larger_m)
   	s_wmoment = s_wmoment - 1;
   if(smaller_m)
   	s_wmoment = s_wmoment + 1;

	sparams[12-11]= s_w;
        sparams[11] = s_wmoment;

}

 void bias_scale(char_ *bias, int_ *outdiff, int_ * moment, int *params, int* sparams, int rate) {

 	char_ bias_buf[PARA];
 	int_ moment_buf[PARA];
 	int_ in_buf[PARA][img_num];
 	int_ out_buf[PARA];
 	int_ factor[PARA];
 	int_ sum[PARA];
 	int_ mean[PARA];
 	int_ var[PARA];
 	int_ out_tmp[PARA][img_num];
 	ubit_ larger = 0;
 	ubit_ smaller = 1;
 	ubit_ larger_m = 0;
 	ubit_ smaller_m = 1;

 	int inChannel = params[0];
 	int outChannel = params[1];
 	int inCh_once = params[2];
 	int outCh_once = params[3];
 	int iteration_in = inChannel/inCh_once;
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
  int s_outdif_pre = params[20];*/
  //int s_in_pre0 = params[21];
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
  int s_wmoment = sparams[11]; 
  int s_wmoment_pre = sparams[12]; 
  int s_bmoment = sparams[13]; 
  int s_bmoment_pre = sparams[14]; 

 	  for(int i = 0; i<iteration_in; i++)
 	 {
 	  int bias_trans_size= PARA;
 	  int bias_offset = i*PARA;
 	  memcpy(bias_buf, bias + bias_offset, bias_trans_size);
 	  memcpy(moment_buf , moment + bias_offset, bias_trans_size*4);
 	
    	 #pragma omp parallel for
 	 for(int p=0;p<PARA;p++)
 	 {
 		for(int y =0; y < y_dim; y++)
 		  for(int x = 0; x < x_dim;x++)
 		 {
 			    out_buf[p] = outdiff[(y*x_dim+x)*inChannel+i*PARA+p];
                
                if(x==0 && y==0)
                sum[p] = out_buf[p];
                else	
                sum[p] += out_buf[p];
 		 }

          moment_buf[p] = momentb_qt(moment_buf[p],sum[p], s_bmoment, s_bmoment_pre, s_outdif_pre, &larger_m, & smaller_m);

          bias[i*PARA+p] = bias_qt(moment_buf[p], bias_buf[p], s_bmoment, s_b_pre, s_b, & larger, & smaller, rate);


 	 }

 	 memcpy(moment + bias_offset, moment_buf, bias_trans_size*4);
 	
  }
     
      sparams[18-11]=s_b;
      sparams[14] = s_bmoment;

 	   if(larger)
 	   	s_b = s_b-1;
 	   if(smaller)
 	   	s_b = s_b+1;
 	   if(larger_m)
 	   	s_bmoment = s_bmoment-1;
 	   if(smaller)
 	   	s_bmoment = s_bmoment+1;

       sparams[14-11]=s_b;
       sparams[13] = s_bmoment;

 }




