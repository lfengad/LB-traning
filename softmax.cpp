#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>
#include "ap_int.h"
#include "math.h"
#include "conv.h"

#define PARA 10
//#define img_num 32


ap_uint<8> seed_soft = 255;

void softmax(int_ *input, char_ *output, int *params, int* sparams, int_* diff, float* tmp)
{
	char_ weight_buf[PARA];
	float sum[img_num];
	float chn[PARA][img_num];
	float tm[img_num];
	char_ bias_buf[PARA];
	int_ in_buf[PARA][img_num];
	float in_diffbuf[PARA];
	float resbuf[PARA][img_num];
	int_ diffbuf[PARA];
	char_ out_buf[PARA][img_num];
    int_ factor[PARA];
    int_ mean[PARA];
    int_ var[PARA];
    int_ out_tmp[PARA][img_num];
    ap_uint<1> larger = 0;
    ap_uint<1> smaller = 1;



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

	    int x_out = x_dim;  //((x_dim - ksize + 2 * pad) / stride) + 1;
	    int y_out = x_out;
   
               //  printf("start softmax\n");
             
		 for(int i = 0; i<iteration_in; i++)
		    {

		    for(int p=0;p<PARA;p++)
		    {
               //  printf("start softmax 0\n");
		    int in_trans_size= img_num;
		    int in_offset = (i*PARA+p)*img_num;
		    memcpy(in_buf[p], input + in_offset, 4*in_trans_size);
               //  printf("start softmax 1\n");
                    #pragma omp parallel for
                    for(int im =0; im<img_num;im++)
                     	{
                     // printf("at %d and %d: %d and %d sinpre is %d\n", p, im , (int)in_buf[p][im],(int)*(input + in_offset+im),  s_in_pre);
              		tm[im] = exp((float)in_buf[p][im]/exp2((float)s_in_pre));
              //printf("tm is %f\n", tm[im]);
              		if(i==0 && p==0)
               		sum[im] = tm[im];
              		else
               		sum[im] += tm[im];
             		}
                 //   printf("at %d: ", p);
                 //    for(int im=0;im<img_num;im++) 
                 //     printf(" %d ",(int)in_buf[p][im]);
                 //    printf("\n");  
            		memcpy(tmp+in_offset, tm, 4*in_trans_size);
		    }
		    }
                // printf("start softmax 2\n");

			for(int i = 0; i<iteration_in; i++)
			{
                        #pragma omp parallel for
			for(int p=0;p<PARA;p++)
			{
	                 int in_trans_size= img_num;
			 int in_offset = (i*PARA+p)*img_num;
              //           printf("0:%d\n", p); 
			 memcpy(chn[p], tmp + in_offset, 4*in_trans_size);
                //         printf("1:%d\n", p); 
                         memcpy(out_buf[p], output+in_offset, in_trans_size);
                //         printf("2:%d\n", p); 
			 in_diffbuf[p] = 0;
			  for(int im=0;im<img_num;im++)
			   {
                          //    printf("sum is %f\n", sum[im]);
			      in_diffbuf[p]+= chn[p][im]/sum[im];
			      resbuf[p][im] = chn[p][im]/sum[im];
	                      in_diffbuf[p]-= out_buf[p][im];
			   }
			    in_diffbuf[p]/= img_num;
			    int_ tmp_int = (int)(in_diffbuf[p]*(exp2((float)s_out))*512);
			    int_ tmp_int0 = tmp_int.range(19,8) + round_(tmp_int.range(7,0), &seed_soft);
			    int_ tmp_int1 = tmp_int.range(19,9) + round_(tmp_int.range(8,1), &seed_soft);
			    if(tmp_int1>127 || tmp_int1<-128)
				 {
					 larger = 1;
				 }
				 if(tmp_int0>127 || tmp_int0<-128)
				 {
					 smaller = 0;
				 }
				 
				 if(tmp_int1>127)
				 	tmp_int1 = 127; 
				  else if (tmp_int1<-128)
				    tmp_int1 = -128;
			    // diffbuf[p] = tmp_int1.range(7,0);
			     diff[i*PARA+p] = tmp_int1.range(7,0);
			}
      //                 printf("back loss:");
      //                 for(int p = 0;p<PARA;p++)
      //                    printf(" %d ",(int)diff[i*PARA+p]);    
     //                  printf("\n");   
             
                     //    printf("3:%d\n", i); 
                        memcpy( tmp+ i*PARA*img_num , resbuf, PARA*img_num*4 );
                     //    printf("4:%d\n", i); 
			}
			
                    sparams[19-11] = s_out;

			   if(larger)
			   	s_out = s_out -1;
			   if(smaller)
			   	s_out = s_out +1;



		   	sparams[13-11] = s_out;




}


