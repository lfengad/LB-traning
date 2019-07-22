#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>
#include "ap_int.h"
#include "math.h"
#include "conv.h"

#define PARA 16

//#define img_num 32

void batch(int_ *input, int_ *temp,
    int_ *output, int *params, int *sparams) {

	char_ weight_buf[PARA][1024];
	int_ in_buf[PARA][img_num];
    char_ bias_buf[PARA][2048/PARA];
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
 /* int s_in = params[11]; 
  int s_w = params[12];
  int s_out = params[13];
  int s_b = params[14];
  int s_in_pre = params[15];
  int s_w_pre = params[16];
  int s_out_pre = params[17];
  int s_b_pre = params[18]; 
  int s_indif_pre = params[19];
  int s_outdif_pre = params[20];*/

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


//	int rate = params[16];



     for(int i = 0; i<iteration_in; i++)
    {
    #pragma omp parallel for
    for(int p=0;p<PARA;p++)
    {
    	for(int y =0; y < y_dim; y++)
    	  for(int x = 0; x < x_dim;x++)
    	 {
    	    	int in_trans_size = img_num;
    		    int in_offset = ((y*x_dim+x)*inChannel+i*PARA+p)*img_num;
    		    memcpy(in_buf[p], input+in_offset, in_trans_size*4);
    	        for(int im=0;im<img_num;im++)
    	        {
    	           if(x==0 && y==0 && im==0)
    	           sum[p] = in_buf[p][im];
    	           else	
    	           sum[p]+= in_buf[p][im];
    	        }
    	  }
    	  mean[p] = sum[p]/x_dim/y_dim/img_num;
      	for(int y =0; y < y_dim; y++)
      	  for(int x = 0; x < x_dim;x++)
      	 {
      	    	int in_trans_size = img_num;
      		    int in_offset = ((y*x_dim+x)*inChannel+i*PARA+p)*img_num;
      		    memcpy(in_buf[p], input+in_offset, in_trans_size*4);
      	        for(int im=0;im<img_num;im++)
      	        {

      	           int_ tmp = (in_buf[p][im]-mean[p]);
      	           
      	           if(x == 0 && y==0 && im ==0)
                   sum[p] = tmp*tmp;
                   else 
      	           sum[p] += tmp*tmp;
      	        }

      	  }
         
          if(s_in_pre>0)
      	  var[p] = (sum[p]/x_dim/y_dim/img_num) >> s_in_pre;
          else
      	  var[p] = (sum[p]<< -s_in_pre)/x_dim/y_dim/img_num;
  
            float e = 1e-5;
            factor[p] = (int)(sqrt((float)var[p]/(exp2((float)s_in_pre)) + e)* (exp2((float)s_in_pre)));
 //           printf(" %d-%f ", (int)factor[p],(sqrt((float)var[p]/s_in_pre+e)*s_in_pre));
            for(int y =0; y < y_dim; y++)
		    for(int x = 0; x < x_dim;x++)
		     {
				int in_trans_size = img_num;
				int in_offset = ((y*x_dim+x)*inChannel+i*PARA+p)*img_num;
				memcpy(in_buf[p], input+in_offset, in_trans_size*4);
				for(int im=0;im<img_num;im++)
				{
                                 if(s_in_pre > 0 ) 
				 out_tmp[p][im] = ((in_buf[p][im]-mean[p])<<s_in_pre)/factor[p];
				 else
                                 out_tmp[p][im] = ((in_buf[p][im]-mean[p])>>-s_in_pre)/factor[p];
			    }
				memcpy(output+in_offset, out_tmp[p], in_trans_size*4);
		     }
    }
  //  printf("\n");
    memcpy(temp+i*PARA, factor, PARA*4);
//   printf("temp: "); 
//   for(int p = 0; p<PARA; p++)
//     printf(" %d ", (int)(*(temp+i*PARA+p)));
//   printf("\n");
 }

   sparams[17-11] = s_in_pre;
}

void batch_back(int_ *output, int_ *indiff, int_ *temp,
    int_ *outdiff,  int *params, int *sparams) {

	char_ weight_buf[PARA][1024];
	int_ out_buf[PARA];
	int_ dif_buf[PARA];
    char_ bias_buf[PARA][2048/PARA];
    long_ sum0[PARA];
    long_ sum1[PARA];
    int_ mean0[PARA];
    int_ mean1[PARA];
    int_ tmp_buf[PARA];
    int_ out_tmp[PARA];
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
/*  int s_in = params[11]; 
  int s_w = params[12];
  int s_out = params[13];
  int s_b = params[14];
  int s_in_pre = params[15];
  int s_w_pre = params[16];
  int s_out_pre = params[17];
  int s_b_pre = params[18]; 
  int s_indif_pre = params[19];
  int s_outdif_pre = params[20];*/
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


     for(int i = 0; i<iteration_in; i++)
    {
    	for(int y =0; y < y_dim; y++)
    	  for(int x = 0; x < x_dim;x++)
    	 {
    	    	int in_trans_size = PARA;
    		    int in_offset = ((y*x_dim+x)*inChannel+i*PARA);
    		    memcpy(out_buf, output+in_offset, in_trans_size*4);
    		    memcpy(dif_buf, outdiff+in_offset, in_trans_size*4);
                    #pragma omp parallel for
    		    for(int p=0;p<PARA;p++)
    		    {  
    		       if(x==0 && y==0)
    		       sum0[p] = dif_buf[p];	
    		       else	
                   sum0[p]+= dif_buf[p];

                   if(x==0 && y==0)
                   sum1[p] = dif_buf[p]*out_buf[p];
                   else
                   sum1[p]+= dif_buf[p]*out_buf[p];
    	        }
    	  }
         #pragma omp parallel for
    	 for(int p=0;p<PARA;p++)
    	 {
        //  printf("%d %d %d\n", x_dim, y_dim, s_out_pre);
    	  mean0[p] = sum0[p]/x_dim/y_dim;
    	  if(s_out_pre>0)
          mean1[p] = (sum1[p]/x_dim/y_dim) >> s_out_pre;
          else
          mean1[p] = (sum1[p]<<-s_out_pre)/x_dim/y_dim;
   
    	 }
    	 memcpy(tmp_buf, temp + i*PARA, PARA*4);
     //    for(int p =0;p<PARA;p++)
     //    {
     //      printf(" %d-%d ",(int)tmp_buf[p],(int)(*(temp+i*PARA+p)) );
    //       }
    //    printf("\n");

     	for(int y =0; y < y_dim; y++)
     	  for(int x = 0; x < x_dim;x++)
     	 {
     	    	int in_trans_size = PARA;
     		    int in_offset = ((y*x_dim+x)*inChannel+i*PARA);
     		    memcpy(out_buf, output+in_offset, in_trans_size*4);
     		    memcpy(dif_buf, outdiff+in_offset, in_trans_size*4);

                    #pragma omp parallel for
     		    for(int p=0;p<PARA;p++)
     		    {
     //               printf("%d\n", s_out_pre);
                    int_ tm0;
                    if(s_out_pre>0)   
                    tm0 = (out_buf[p]*mean1[p])>>s_out_pre;
                    else
                    tm0 = (out_buf[p]*mean1[p])<<-s_out_pre;
                    int_ tm1 = dif_buf[p] - mean0[p] - tm0;
       //             printf("teher %d %d\n",(int)tm1,(int)tmp_buf[p]); 
                    out_tmp[p] = tm1/tmp_buf[p];
       //             printf("finish %d\n", (int)out_tmp[p]); 
     	          }
     		    memcpy(indiff+in_offset, out_tmp,in_trans_size*4);
     	  }
    }

   sparams[19-11] = s_outdif_pre;
}

