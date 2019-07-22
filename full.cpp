#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>
#include "ap_int.h"
#include "math.h"
#include "conv.h"

#define PARA 10
//#define img_num 32


ap_uint<8> seed_fc = 255;

void fc(int_ *input, char_ *weights, char_ *bias,
    int_ *output, int *params, int*sparams) {

	char_ weight_buf[PARA][64];
	int_ in_buf[64][img_num];
    char_ bias_buf[PARA];
    int_ tmp[PARA][img_num];
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

     for(int o = 0; o<iteration_out; o++)
     {   
         int bias_offset = o*PARA;
         memcpy(bias_buf, bias+ bias_offset , PARA);

	    for(int i = 0; i<iteration_in; i++)
	     for(int y =0; y < y_dim; y++)
    	  for(int x = 0; x < x_dim;x++)
    	 {
    	     int in_trans_size = inCh_once*img_num;
    	     int in_offset = ((y*x_dim+x)*inChannel + i*inCh_once)*img_num;

    	    if(o==0)
    	     for(int in=0; in<inCh_once;in++)
    		    #pragma omp parallel for
    	     for(int im=0; im<img_num;im++)
    	       	   {
   

              in_buf[in][im] = input_qt(input[in_offset+in*img_num+im], s_in_pre, s_in, &larger, &smaller);

    	       input[in_offset+in*img_num+im] = in_buf[in][im];
    	       	   }
    	    else
   	           memcpy(in_buf, input+in_offset, 4*in_trans_size);



           #pragma omp parallel for
    	   for(int p =0; p<PARA;p++)
    	    {
    	     int wg_trans_size = inCh_once;
             int wg_offset = ((y*x_dim+x)*outChannel + o*PARA+p)*inChannel + i*inCh_once;
             memcpy(weight_buf[p], weights+wg_offset, wg_trans_size);
    	    }

    	   #pragma omp parallel for
    	   for(int p =0; p<PARA; p++)
    	   {

    		for(int in = 0; in< inCh_once; in++)
    		{ 
              char_ wg = weight_buf[p][in];
    		
            for(int im = 0; im<img_num; im++)
    		{
    		  if( x==0 && y==0 && i == 0 && in==0 )
    		 { 
                   if(s_w_pre+s_in>s_b_pre)
                    tmp[p][im] = bias_buf[p]<<(s_w_pre+s_in-s_b_pre) + wg*in_buf[in][im];
                   else 
                   tmp[p][im] = bias_buf[p]>>(-(s_w_pre+s_in-s_b_pre)) + wg*in_buf[in][im];
                  // printf(" %d-%d-%d-%d ", (int)tmp[p][im], (int)in_buf[in][im], (int)wg, (int)bias_buf[p]);
                   }
    		  else
       		 { tmp[p][im] += wg*in_buf[in][im];
                 //  printf(" %d-%d-%d ", (int)tmp[p][im], (int)in_buf[in][im], (int)wg );
    	         }	
              }
    	   }
    	   }
           printf("\n");
    	 }


    	  int out_trans_size = PARA*img_num;
       	  int out_offset = o*PARA*img_num;
       	  memcpy(output+out_offset, tmp, 4*out_trans_size);
       
 }
    sparams[17-11] = s_w_pre + s_in;


    if(larger)
    	s_in = s_in-1;
    if(smaller)
    	s_in = s_in+1;

    sparams[11-11] = s_in;
}
#define PARA 16
void back_fc(int_ *outdiff, char_ *weights,
    int_ *indiff, int *params, int* sparams) {

	char_ weight_buf[PARA][64];
	int_ out_buf[64];
   // char_ bias_buf[PARA][2048/PARA];
    int_ in_buf[PARA];
    ap_uint<1> larger = 0;
    ap_uint<1> smaller = 1;

	int inChannel = params[0];
	int outChannel = params[1];
	int inCh_once = params[2];
	int outCh_once = params[3];
	int iteration_in = inChannel/PARA;
	int iteration_out = outChannel/outCh_once;
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


         // printf("1st %d\n", iteration_in);

     for(int o = 0; o<iteration_out; o++)
     {   int out_trans_size = outCh_once;
	     int out_offset = o*outCh_once;
	     //memcpy(out_buf, outdiff+out_offset, out_trans_size);
    		    #pragma omp parallel for
	       
         	 for(int on=0;on<outCh_once;on++)
	     {
             //      printf("try 0 \n"); 
		    out_buf[on] = outdiff_qt(outdiff[out_offset+on], s_outdif_pre, s_out, &larger, &smaller, & seed_fc);
       
		    outdiff[out_offset+on] = out_buf[on];
              //      printf(" %d:%d:%lld:%lld:%lld ", (int)outdiff[out_offset+on], (int)tmp_int1, s_out, s_outdif_pre, (int)(outdiff[out_offset+on]*512*s_out/s_outdif_pre));
           //        printf("try 1 \n"); 
		  }
 //             printf("\n");
//            printf("2nd %d\n", iteration_in);

	    for(int i = 0; i<iteration_in; i++)
	     for(int y =0; y < y_dim; y++)
    	  for(int x = 0; x < x_dim;x++)
    	 {
    	     {
    		 int in_trans_size = PARA;
    	     int in_offset = ((y*x_dim+x)*inChannel + i*PARA);
          //  printf("befroe %d %d\n", x, y); 
    	     memcpy(in_buf, indiff+in_offset, 4*in_trans_size);
          //   printf("after %d %d\n", x, y); 
    	     }
    	   #pragma omp parallel for
    	   for(int on =0; on<outCh_once;on++)
    	    {
             int wg_offset = ((y*x_dim+x)*outChannel + o*outCh_once+on)*inChannel + i*PARA;
             for(int p = 0; p<PARA;p++)
             weight_buf[p][on] = weights[wg_offset+p];
    	    }

    	   #pragma omp parallel for
    	   for(int p =0; p<PARA; p++)
    	   {

    		for(int on = 0; on< outCh_once; on++)
    		{  char_ wg = weight_buf[p][on];
    		{
    		  if(on == 0 && o == 0)
    		  { 
                   in_buf[p] = wg*out_buf[on];
          //         printf(" %d-%d-%d ", (int)in_buf[p], (int)wg, (int)out_buf[on]);
    		  }
                  else
    		  { 
                   in_buf[p]+= wg*out_buf[on];
            //       printf(" %d-%d-%d ", (int)in_buf[p], (int)wg, (int)out_buf[on]);
    		  }
                }
    	        }
    	  }
         // printf("\n");

  		 int in_trans_size = PARA;
  	     int in_offset = ((y*x_dim+x)*inChannel + i*PARA);
        //     printf("befroe %d %d %d\n", x, y, i); 
  	     memcpy(indiff+in_offset, in_buf, 4*in_trans_size);
         //    printf("after %d %d %d\n", x, y, i); 
    	}


      }

     sparams[19-11] = s_w_pre + s_out;

    if(larger)
    	s_out = s_out-1;
    if(smaller)
    	s_out = s_out+1;


	  sparams[13-11] = s_out;

 //   printf("finsih\n");
}

#define PARA 10

void fc_weight(int_ *outdiff, char_ *weights,  int_ *input,
   int_ * moment,  int *params, int*sparams, int rate, int decay)
{

	int_ weight_buf[PARA][64];
    int_ moment_buf[PARA][64];
	int_ in_buf[64][img_num];
   // char_ bias_buf[PARA][2048/PARA];
    int_ out_buf[PARA];
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


     for(int o = 0; o<iteration_out; o++)
     {   int out_trans_size = PARA;
	     int out_offset = o*PARA;
          //   printf("befroe %d\n", o);
	     memcpy(out_buf, outdiff+out_offset, 4*out_trans_size);
          //   printf("after %d\n", o);
	    for(int i = 0; i<iteration_in; i++)
	     for(int y =0; y < y_dim; y++)
    	  for(int x = 0; x < x_dim;x++)
    	 {
    	     int in_trans_size = inCh_once*img_num;
    	     int in_offset = ((y*x_dim+x)*inChannel + i*inCh_once)*img_num;
             
         //    printf("befroe %d %d %d %d\n", o, i, x, y);
    	     memcpy(in_buf, input+in_offset, 4*in_trans_size);
         //    printf("after %d %d %d %d\n", o, i, x, y);

            #pragma omp parallel for
    	   for(int p =0; p<PARA;p++)
    	    {
             int wg_offset = ((y*x_dim+x)*outChannel + o*PARA+p)*inChannel + i*inCh_once;
         //    printf("%d %d %d %d %d %d %d %d %d", y, x_dim, x , outChannel, o ,p, inChannel, i, inCh_once );
         //    printf("befroe %d %d %d %d %d\n", o, i, x, y, p);
             //printf("%d and ", moment_buf[p])
             memcpy(moment_buf[p], moment + wg_offset, inCh_once*4);
          //   memcpy(moment_buf[p], moment_buf + wg_offset, inCh_once*4);
          //   memcpy(moment_buf[p], moment_buf + wg_offset, inCh_once*4);
          //   printf("after %d %d %d %d %d\n", o, i, x, y, p);
    	    }

         //  printf("try 0\n");
    		    #pragma omp parallel for
    	   for(int p =0; p<PARA; p++)
    	   {
    		char_ op = out_buf[p];
    		for(int in = 0; in< inCh_once; in++)
    		{
    	    int wg_offset = ((y*x_dim+x)*outChannel + o*PARA+p)*inChannel + i*inCh_once+in;
    	    weight_buf[p][in] = 0;
    		for(int im=0; im<img_num; im++)
    		{
    	     weight_buf[p][in]+= in_buf[in*img_num+im]*op;
    		}
          moment_buf[p][in] = moment_qt( moment_buf[p][in], weight_buf[p][in], s_wmoment, s_wmoment_pre, s_in_pre+ s_outdif_pre, &larger_m, &smaller_m );

          weights[wg_offset] = weight_qt(moment_buf[p][in], weights[wg_offset],s_wmoment, s_w_pre, s_w, &larger, &smaller, rate, decay);

		    }

    	  }

         //  printf("try 1\n");
    		    #pragma omp parallel for
           for(int p =0; p<PARA;p++)
            {
             int wg_offset = ((y*x_dim+x)*outChannel + o*PARA+p)*inChannel + i*inCh_once;
             memcpy( moment + wg_offset, moment_buf[p], inCh_once*4);
            }

    	   }
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



         //  printf("finish\n");
}


void bias_back_fc(int_ *outdiff, char_ *bias, int_ * moment,
		int *params, int* sparams, int rate)
{
	int_ bias_buf[PARA];
	//int_ out_buf[256];
    //char_ in_buf[PARA*4096];
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
 /* int s_in = params[11]; 
  int s_w = params[12];
  int s_out = params[13];
  int s_b = params[14];
  int s_in_pre = params[15];
  int s_w_pre = params[16];
  int s_out_pre = params[17];
  int s_b_pre = params[18]; 
  int s_indif_pre = params[19];
  int s_outdif_pre = params[20];
*/

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
 // int s_in_pre0 = params[21];
//	int rate = params[16];
	int x_out = ((x_dim - ksize + 2 * pad) / stride) + 1;
    int y_out = x_out;

    for(int i = 0;i<iteration_out;i++)
    {
    		    #pragma omp parallel for
    	for(int o = 0; o< PARA; o++)
    	 {
    		int bias_offset = i*PARA+o;
            int_ moment_buf = moment[bias_offset];
               

          moment_buf = momentb_qt(moment_buf, outdiff[bias_offset], s_bmoment, s_bmoment_pre, s_outdif_pre, &larger_m, & smaller_m);

          bias[bias_offset] = bias_qt(moment_buf, bias[bias_offset], s_bmoment, s_b_pre, s_b, & larger, & smaller, rate);


    
          moment[bias_offset] = moment_buf;
         }
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


