#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>
#include <cmath>
#include <random>
#include <fstream>
#include "ap_int.h"
#include "math.h"
#include "conv.h"
#include "table.h"

#define PI 3.14159265
#define img_num 128
#define layer_num 94

#include <cstdlib>
#include <cmath>
#include <limits>


double normdis(double mu, double sigma)
{
	static const double epsilon = std::numeric_limits<double>::min();
	static const double two_pi = 2.0*3.14159265358979323846;

        static double z1;
	static bool generate;
	generate = !generate;

	if (!generate)
	   return z1 * sigma + mu;

	double u1, u2;
	do
	 {
	   u1 = rand() * (1.0 / RAND_MAX);
	   u2 = rand() * (1.0 / RAND_MAX);
	 }
	while ( u1 <= epsilon );

	double z0;
	z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
	z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
	return z0 * sigma + mu;
}


int main(int argc, char const *argv[])
{   
    int_ * data =  (int_*)malloc(113766836 * 4);
    char_ * wg = (char_*)malloc(281946);
    float fp[2];
    char * label = (char*)&wg[280666];
    float * res = (float*)&data[112873578];    
    FILE* outloss;
    outloss = fopen("outputloss.txt","w");
    FILE* outs;
    outs = fopen("outputs.txt","w");
 
    int epoch = 100;
    int decay = 2000;
    float lr = 0.1;
    int rates[layer_num];

    for(int i=0; i<epoch; i++)
    {
       float lrt =  0.1 * cos(i*PI/2/100);
       rates[i] = 1/lrt;
    }

    int params[layer_num][16];
    int offset[layer_num][8];
    int connect[layer_num][4];
    int sparams[layer_num][16];
 //   int outfactor[256][2];

    for(int i=0; i<layer_num; i++)
	{
     // printf("first %d\n", i);
     // printf("content is ", *(ins+28*i))
      memcpy(params[i], (int*)ins+28*i, 16*4);
     // printf("second %d\n", i);
      memcpy(offset[i], (int*)ins+28*i+16, 8*4);
     // printf("third %d\n", i);
      memcpy(connect[i], (int*)ins+28*i+24, 4*4);
      for(int j = 0; j< 11; j++)
        sparams[i][j] = 16;
      for(int j = 11; j< 16; j++)
        sparams[i][j] = 18;
      if(params[i][10] == 2) 
        sparams[i][5] = 6;
      if(i == 0)
        sparams[i][4] = 0;
	}
     printf("here finish cpy\n");
   // outfactor[0][0] = params[0][15];


  
    float sample;
    int w_num, innode;
   // std::default_random_engine gen;
   // std::normal_distribution<float> d(0,1);


    for(int i=0;i<layer_num;i++)
    {
     // printf("here initilize weight at %d\n", i);
      switch(ins[i][10])
      {
       case 0:
           w_num =  params[i][0]*params[i][1]*params[i][8]*params[i][8];
           innode = params[i][0]*params[i][8]*params[i][8]; 
         //  printf("wg: \n");
           for(int j = 0; j < w_num; j++)
           {
           //  printf("conv %d\n",j);  
            // sample = d(gen); 
             sample = normdis(0,1);
           //  printf("conv finish %d\n",j);  
             wg[offset[i][1]+j] = std::sqrt(2/(float)innode)*sample*65536;  
           //  printf(" %f-%d-%f-%f-%f ", sample, (int)wg[offset[i][1]+j], std::sqrt(2/(float)innode)*sample*50000, 2/(float)innode, std::sqrt(2/(float)innode));  
         //   if(i == 0)
         //   { printf(" %d ", (int)wg[offset[i][1]+j]); 
         //      if(j%(64*3) == 0)
         //      printf("\n");         
         //    }
           }
       //    printf("wg:finish \n");
       //    printf("\n");
           break;
       case 2:
           w_num =  params[i][1];
           for(int j = 0; j < w_num; j++)
           {
          //  printf("scale %d\n",j);  
             wg[offset[i][1]+j] = 1*64;  
             wg[offset[i][3]+j] = 0;
           }
           break;
       case 6:
           w_num =  params[i][0]*params[i][1]*params[i][4]*params[i][4];
          // printf("full: \n");
           for(int j = 0; j < w_num; j++)
           {
          //  printf("fc %d\n",j);  
             sample = normdis(0,1); 
           //  printf(" %f-%f-%d ", sample, 0.001*sample*50000, wg[offset[i][1]+j]);  
             wg[offset[i][1]+j] = (int)(0.001*sample*65536);
             wg[offset[i][3]+j] = 0; 
           //  printf(" %f-%f-%d ", sample, 0.001*sample*50000, (int)wg[offset[i][1]+j]);  
           }
           printf("\nfull: finish \n");
           break;    
      }  
    }

    std::ifstream file;
   for(int ep = 0; ep<100; ep++)
  { 
    int idx = 0;
   while(idx <(50000/img_num)*img_num)
   { 
    if(idx == 0) 
    file.open("/home/ust.hk/lfengad/Downloads/cifar-10-batches-bin/data_batch_1.bin", std::ios::binary);
    else if(idx == 10000)
    {file.close();
     file.open("/home/ust.hk/lfengad/Downloads/cifar-10-batches-bin/data_batch_2.bin", std::ios::binary);
    }else if(idx == 20000)
    {file.close();
     file.open("/home/ust.hk/lfengad/Downloads/cifar-10-batches-bin/data_batch_3.bin", std::ios::binary);
    }else if(idx == 30000)
    {file.close();
     file.open("/home/ust.hk/lfengad/Downloads/cifar-10-batches-bin/data_batch_4.bin", std::ios::binary);
    }else if(idx == 40000)
    {file.close();
     file.open("/home/ust.hk/lfengad/Downloads/cifar-10-batches-bin/data_batch_5.bin", std::ios::binary);
    }
    if(file.is_open())
    {
   //  printf("idx here is %d\n", idx); 
     int im = idx%img_num;
     unsigned char labelno;
     file.read((char*)&labelno, 1);
     for(int i = 0; i<10 ;i++)
     {
      if(labelno == i)
      label[i*img_num+im] = 1;
      else
      label[i*img_num+im] = 0;
     }     
     for(int i = 0; i< 3; i++)
      for(int y = 0; y<32; y++)
       for(int x = 0 ;x<32; x++)
         {
           unsigned char tmp;
           file.read((char*)&tmp, 1); 
           data[(y*32+x)*3*img_num+i*img_num+im] = tmp;      
         }      
     idx++;
     if(idx%img_num == 0)
     {
     printf("start trainingg idx:%d\n", idx);
     training( data, wg, fp, (int*)ins, (int*)sparams, rates[ep], epoch, decay);
     float loss = 0;     

     for(int i = 0; i<img_num;i++)
      {
       for(int j = 0;j < 10; j++) 
         { if(j == 1)
             loss+= -log(res[j*img_num+i]);
         }
      }
     printf("************************batch:%d loss:%f\n", idx/img_num, loss); 
     fprintf(outloss,"batch:%d loss:%f\n", idx/img_num, loss); 
     for(int i=0; i<layer_num; i++ ) 
     {
       for(int j = 0 ;j<12; j++)
         fprintf(outs, " %lld ", sparams[i][j]);
       fprintf(outs, "\n");
     }
       fprintf(outs, "*****************************************\n");
     }

    }
    else
      printf("error: no file open\n");     
   }
   
   }





    return 0;
}

