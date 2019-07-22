#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>
#include "ap_int.h"
#include "math.h"
#include "conv.h"

#define PARA 16

//#define img_num 

uchar_ seed_conv = 255;




void conv(int_ *input, char_ *weights, /*char_ *bias,*/
    int_ *output, int *params, int *sparams) {

	  char_ weight_buf[PARA][1024];
	  int_ in_buf[49][PARA*img_num];
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
	int y_dim = x_dim;
	//int moment = params[5];
	//params[5];
	int stride = params[6];
	int pad = params[7];
	int ksize = params[8];
	int bias_en = params[9];
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
     //   printf("stride %d\n", stride);  
	int x_out = ((x_dim - ksize + 2 * pad) / stride) + 1;
        int y_out = x_out;

     //                                               printf("s_in_pre %lld\n", s_in_pre);  

     //  for(int i = 0; i<100; i++)
     //  printf(" %d-%p ",(int)input[i], input+i);
     //  printf("\n");


	/*for(int i = 0;i<iteration_out;i++)
	 for(int o = 0; o< PARA; o++)
	 {
		 int bias_trans_size = 1;
		 int bias_offset = (i*PARA + o);
		 memcpy(bias_buf[o]+ i, bias+ bias_offset , 1);
	 }*/

 //     printf("%d %d %d %d\n", iteration_in, iteration_out, y_out, x_out);

   for(int i = 0; i<iteration_in; i++)
     for(int o = 0; o<iteration_out; o++)
    {
       #pragma omp parallel for
       for(int p =0; p<PARA;p++)
       {
    	int weight_trans_size = inCh_once*ksize*ksize;
    	int weight_offset = (((o*PARA+p))*inChannel + i*inCh_once)*ksize*ksize;
    	memcpy(weight_buf[p], weights + weight_offset, weight_trans_size);
       }
     //  if(bias_en)
    //  { int bias_offset = (o*PARA);
     //   memcpy(bias_buf, bias + bias_offset, PARA);
   //    }
    	for(int y =0; y < y_out; y++)
    	  for(int x = 0; x < x_out;x++)
    	 {
    		//if(i!=0)
           // printf("kszie %d\n", ksize);
            for(int p=0;p<ksize;p++)
            	for(int q=0;q<ksize;q++)
            	{     // printf("hellll\n");
            	      int x_in = x*stride-pad+q;
				int y_in = y*stride-pad+p;
				int inoff = (p*ksize+q);
			    int insize = inCh_once*img_num;
				if(x_in<0 || y_in<0 || x_in>x_dim-1 || y_in>y_dim-1)
				{
         //                           printf("hello2\n");
				    memset_int(in_buf[inoff], insize);
				}
				else if((stride + q < ksize) && x != 0)
				{
        //                            printf("hello3\n");
				   memcpy_int(in_buf[inoff], in_buf[p*ksize+stride + q], insize);
				}
				else
				{
					int in_off = ((y_in*x_dim+x_in) * inChannel + i*inCh_once )*img_num;

          //                               printf("hello0\n");  
					if(o==0 && (stride + p >= ksize || y==0))
					{
            //                             printf("hello1\n");  
					 for(int in=0;in<inCh_once;in++)
    		                                #pragma omp parallel for
						for(int im = 0;im<img_num;im++)
						{
	         in_buf[inoff][in*img_num+im] = input_qt(input[in_off + in*img_num+im], s_in_pre, s_in, &larger, &smaller);
		input[in_off+in*img_num+im] = in_buf[inoff][in*img_num+im];
					        //        printf(" %d-%d-%d ", (int)input[in_off+in*img_num+im],(int)tmp_int1, in_off+in*img_num+im);	
                                               }
                                        //  printf("\n");
					}
					else
					memcpy(in_buf[inoff], input + in_off, 4*insize);
				}
            	}


      #pragma omp parallel for
       for(int p=0;p<PARA;p++)
       {

    	 int out_trans_size = img_num;
    	 int out_offset = ((y*x_out+x)*outChannel + o*PARA + p)*img_num;
    	 memcpy(tmp[p], output+out_offset, 4*out_trans_size);

        for(int in = 0;in<inCh_once;in++)
         for(int m=0;m<ksize;m++)
          for(int n=0;n<ksize;n++)
          {
           char_ wg = weight_buf[p][in*ksize*ksize+m*ksize+n];
    	   for(int im=0;im<img_num;im++)
    	  {
    	      // if(i==0 && in==0 && bias_en)
    	    //	tmp[p][im] = bias_buf[p]*s_in*s_w/s_b + in_buf[m*ksize+n][in*img_num + im]*wg;//0;//bias[p][i]*s_w*s_in/s_b;
    	       if (i ==0 && in == 0)
               { tmp[p][im] = in_buf[m*ksize+n][in*img_num + im]*wg;//0;//bias[p][i]*s_w*s_in/s_b;
           //      printf(" %d-%d-%d ", (int)in_buf[m*ksize+n][in*img_num + im], (int)wg, (int)tmp[p][im]);//0;//bias[p][i]*s_w*s_in/s_b;
               }
               else
    	       {
    	    	tmp[p][im] += in_buf[m*ksize+n][in*img_num + im]*wg;
             //    printf(" %d-%d-%d ", (int)in_buf[m*ksize+n][in*img_num + im], (int)wg, (int)tmp[p][im]);//0;//bias[p][i]*s_w*s_in/s_b;
    	       }
    	  }
         }


  //  	  int out_trans_size = img_num;
   //    	  int out_offset = ((y*x_out+x)*outChannel + o*PARA + p)*img_num;
       	  memcpy(output+out_offset, tmp[p], 4*out_trans_size);
        // printf("\n");
        }
      }

    }

  sparams[17-11] = s_in + s_w_pre;

    if(larger)
    	s_in = s_in-1;
    if(smaller)
    	s_in = s_in+1;
  
    sparams[11-11] = s_in;

}


void conv_back(int_ *outdiff, char_ *weights, int_ *indiff,
		int *params, int * sparams)
{
	char_ weight_buf[PARA][1024];
	int_ out_buf[64];
        int_ in_buf[49][PARA];
        ubit_ larger = 0;
        ubit_ smaller = 1;

	int inChannel = params[0];
	int outChannel = params[1];
	int inCh_once = params[2];
	int outCh_once = params[3];
	int iteration_in = inChannel/PARA;
	int iteration_out = outChannel/outCh_once;
	int x_dim = params[4];
	int y_dim = x_dim;//params[5];
	//int moment = params[5];
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
	int x_out = ((x_dim - ksize + 2 * pad) / stride) + 1;
        int y_out = x_out;




   for(int o = 0; o<iteration_out; o++)
    for(int i = 0; i<iteration_in; i++)
    {
      #pragma omp parallel for
       for(int p =0; p<outCh_once;p++)
       {
    	int weight_trans_size = ksize*ksize;
    	int weight_offset = ((o*outCh_once+p)*inChannel + i*PARA)*ksize*ksize;
    	for(int m=0; m<PARA; m++)
                {
      //          printf("before %d and %d\n", m, p);       
		memcpy( weight_buf[m]+ksize*ksize*p, weights + weight_offset + m*ksize*ksize, weight_trans_size);
       //         printf("after %d and %d\n", m, p);       
        
                 } 
      }


    	for(int y =0; y < y_out; y++)
    	  for(int x = 0; x < x_out;x++)
    	 {
    		int out_trans_size = outCh_once;
    		int out_offset = (y*x_out + x)*outChannel + o*outCh_once;

     	   if(i==0)
    		    #pragma omp parallel for
     		 for(int on =0;on<outCh_once; on++)
     	   {

              out_buf[on] = outdiff_qt(outdiff[out_offset+on], s_outdif_pre, s_out, &larger, &smaller, &seed_conv);
              outdiff[out_offset+on] = out_buf[on];
     	   }
     	   else
    		memcpy(out_buf, outdiff + out_offset ,out_trans_size*4);



            for(int p=0;p<ksize;p++)
              for(int q=0;q<ksize;q++)
            	{
            	int x_in = x*stride-pad+q;
				int y_in = y*stride-pad+p;
				int inoff = (p*ksize+q);
			    int insize = PARA;
				if(x_in<0 || y_in<0 || x_in>x_dim-1 || y_in>y_dim-1)
				{
				}
				else if((stride + q < ksize) && x != 0)
				{
				   memcpy_int(in_buf[inoff], in_buf[p*ksize+stride + q], insize);
				}
				else
				{
				   int in_off = ((y_in*x_dim+x_in) * inChannel + i*PARA);
				   memcpy_int(in_buf[inoff], indiff + in_off, PARA);
				}
        }


         #pragma omp parallel for
         for(int p=0;p<PARA;p++)
         {
          for(int on = 0;on<outCh_once;on++)
           for(int m=0;m<ksize;m++)
            for(int n=0;n<ksize;n++)
           {
        	char_ wg = weight_buf[p][on*ksize*ksize+m*ksize+n];
    	       if(o==0 && on==0  &&  !((stride + n < ksize) && (x != 0)) &&  !((stride + m < ksize) && (y != 0)) )
    	    	in_buf[m*ksize+n][p] = wg*out_buf[on];
    	       else
    	       {
    	    	in_buf[m*ksize+n][p] += wg*out_buf[on];
    	       }
    	   }
          }


         for(int m=0;m<ksize;m++)
            for(int n=0;n<ksize;n++)
          {
              int x_in = x*stride-pad+n;
              int y_in = y*stride-pad+m;

          	 if(x_in<0 || y_in<0 || x_in>x_dim-1 || y_in>y_dim-1)
          			continue;
    	    int in_trans_size = PARA;
       	  int in_offset = (y_in * x_dim+x_in)*inChannel + i*PARA;
       	  memcpy(indiff+in_offset, in_buf[m*ksize+n], out_trans_size*4);
          }

        }
     }
    
  sparams[19-11] = s_w_pre + s_out;  

   if(larger)
   	s_out = s_out-1;
   if(smaller)
   	s_out = s_out+1;

   sparams[13-11] = s_out;

  }

void weight_back(int_ *outdiff, char_ *weights, int_ *input, int_ * moment,
		int *params, int *sparams, int rate , int decay)
{
	int_ weight_buf[PARA][1024];
	int_ moment_buf[PARA][1024];
	int_ out_buf[PARA];
  int_ in_buf[PARA][4096];
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
	int y_dim = x_dim;//params[5];
  //int s_w = params[5];//aim 
	//int moment = params[5];
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
  int s_wmoment = sparams[11]; 
  int s_wmoment_pre = sparams[12]; 
  int s_bmoment = sparams[13]; 
  int s_bmoment_pre = sparams[14]; 
//	int rate = pa
  //  printf("%d %d %d %d\n", x_dim, ksize, pad, stride);
    int x_out = ((x_dim - ksize + 2 * pad) / stride) + 1;
    int y_out = x_out;



   for(int i = 0; i<iteration_in; i++)
    for(int o = 0; o<iteration_out; o++)
    {

    //   printf("%d %d\n", x_out, y_out);

    	for(int y =0; y < y_out; y++)
    	  for(int x = 0; x < x_out;x++)
    	 {
    		int out_trans_size = PARA;
    		int out_offset = (y*x_out + x)*outChannel + o*PARA;
    		memcpy(out_buf, outdiff + out_offset ,out_trans_size*4);

    		for(int p=0;p<ksize;p++)
			for(int q=0;q<ksize;q++)
			{
			int x_in = x*stride-pad+q;
			int y_in = y*stride-pad+p;
			int inoff = (p*ksize+q);//*inCh_once*img_num;
			int insize = inCh_once*img_num;
			if(x_in<0 || y_in<0 || x_in>x_dim-1 || y_in>y_dim-1)
			{
			   memset_int(in_buf[inoff], insize);
			}
			else if((stride + q < ksize) && x != 0)
			{
			   memcpy_int(in_buf[inoff], in_buf[p*ksize+stride + q], insize);
			}
			else
			{
			   int in_off = ((y_in*x_dim+x_in) * inChannel + i*inCh_once )*img_num;
			   memcpy_int(in_buf[inoff], input + in_off, insize);
			}
			}
      //  printf("here int1\n");

         #pragma omp parallel for
         for(int p=0;p<PARA;p++)
         {
          char_ op = out_buf[p];
          for(int in = 0;in<inCh_once;in++)
           for(int m=0;m<ksize;m++)
            for(int n=0;n<ksize;n++)
            {
        	  for(int im=0;im<img_num;im++)
        	  {
        		if(x==0 && y==0 && im==0)
        		weight_buf[p][in*ksize*ksize+m*ksize+n] = op*in_buf[m*ksize+n][in*img_num+im];
        		else
        		weight_buf[p][in*ksize*ksize+m*ksize+n]+= op*in_buf[m*ksize+n][in*img_num+im];
    	      }
        	}
    	  }
        }    	 

//      printf("here inter\n");
       #pragma omp parallel for
        for(int p =0; p<PARA;p++)
        {
     	  int weight_trans_size = inCh_once*ksize*ksize;
     	  int weight_offset = ((o*PARA+p)*inChannel + i*inCh_once)*ksize*ksize;
     	  memcpy(moment_buf[p], moment+weight_offset, weight_trans_size*4);

      	  for (int l=0;l<weight_trans_size;l++)
      	  {
          
          moment_buf[p][l] = moment_qt( moment_buf[p][l], weight_buf[p][l], s_wmoment, s_wmoment_pre, s_in_pre+ s_outdif_pre, &larger_m, &smaller_m );

          weights[weight_offset+l] = weight_qt( moment_buf[p][l], weights[weight_offset + l], s_wmoment,  s_w_pre, s_w, &larger, & smaller, rate, decay); 
      	  }

        memcpy(moment+weight_offset, moment_buf[p], weight_trans_size*4);
        }

    }
    
  //      printf("here finish\n");

  sparams[16-11] = s_w;
  sparams[12] = s_wmoment;

   if(larger)
   	s_w = s_w - 1;
   if(smaller)
   	s_w = s_w + 1;

   if(larger_m)
   	s_wmoment = s_wmoment - 1;
   if(smaller_m)
   	s_wmoment = s_wmoment + 1;

    sparams[12-11] = s_w;
    sparams[11] = s_wmoment;
 }


/*
void bias_back(int_ *outdiff, char_ *bias, int_ * moment
		int *params, long long * sparams, int rate)
{
	int_ bias_buf[PARA];
  int_ moment_buf[PARA];
	int_ out_buf[PARA];
  //  char_ in_buf[PARA*4096];
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
  int s_outdif_pre = params[20];*/


/*
  long long s_in = sparams[0]; 
  long long s_w = sparams[1];
  long long s_out = sparams[2];
  long long s_b = sparams[3];
  long long s_in_pre = sparams[4];
  long long s_w_pre = sparams[5];
  long long s_out_pre = sparams[6];
  long long s_b_pre = sparams[7]; 
  long long s_indif_pre = sparams[8];
  long long s_outdif_pre = sparams[9];
//	int rate = params[16];
	int x_out = ((x_dim - ksize + 2 * pad) / stride) + 1;
  int y_out = x_out;



    for(int o = 0; o<iteration_out; o++)
    { 
      int moment_offset = o*PARA;
      memcpy(moment_buf, moment + moment_offset, PARA*4);

    	for(int y =0; y < y_out; y++)
    	  for(int x = 0; x < x_out;x++)
    	 {
    		int out_trans_size = PARA;
    		int out_offset = (y*x_out + x)*outChannel + o*PARA;
        memcpy(out_buf, outdiff+out_offset, PARA*4);

            for(int p=0;p<PARA;p++)
            {
            if(x==0 && y==0)
        		bias_buf[p] = out_buf[p];
        		else
        		bias_buf[p] += out_buf[p];
          	}
    	 }

         for(int p =0; p<PARA;p++)
         {
      	  int bias_offset = o*PARA+p;
      	  {
          moment_buf[p] = (moment_buf[p]*9 + bias_buf[p]*moments/s_outdif_pre)/10;   
          ap_int<12> tmp_int = (bias[bias_offset]*s_b*moments*4 - (moment_buf[p])*s_b_pre*s_b*4/rate)/moments/s_b_pre;        
       		ap_int<12> tmp_int0 = tmp_int.range(11,1) + tmp_int.range(0,0);
			    ap_int<12> tmp_int1 = tmp_int.range(11,2) + tmp_int.range(1,1);

            if(tmp_int1>127 || tmp_int1<-128)
            {
             larger = 1;
             }
            if(tmp_int0>127 || tmp_int0<-128)
            {
           	 smaller = 0;
            }
            bias[bias_offset] = tmp_int1.range(7,0);
      	  }
         }

        memcpy(moment + moment_offset, moment_buf, PARA*4);


    }

  sparams[18-11] = s_b;

   if(larger)
   	s_b = s_b/2;
   if(smaller)
   	s_b = s_b*2;

  sparams[14-11] = s_b;

  }
*/





