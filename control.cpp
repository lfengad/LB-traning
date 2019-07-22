#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>
#include "ap_int.h"
#include "math.h"
#include "conv.h"

#define PARA 16

#define layer_num 94

void training(int_* data, char_* wg, float* fp, int* setting, int * sp, int rate, int epoch, int decay)
{
    int params[layer_num][16];
    int offset[layer_num][8];
    int connect[layer_num][4];
 //   long long sparams[layer_num][12];
 //   int outfactor[256][2];
      int (*sparams)[16] = (int (*)[16])  sp; 
	for(int i=0; i<layer_num; i++)
	{
        memcpy(params[i], setting+28*i, 16*4);
  //      printf("%d : %d %d\n", i, params[i][4], params[i][5]);
        memcpy(offset[i], setting+28*i+16, 8*4);
        memcpy(connect[i], setting+28*i+24, 4*4);
    /*  for(int j = 0; j< 12; j++)
        sparams[i][j] = 50000;
      if(params[i][10] == 2) 
        sparams[i][5] = 100;*/
	}

   // outfactor[0][0] = params[0][15];

   //  for(int i = 0;i<layer_num;i++)
   //     printf(" -- %d : %d %d --", i, params[i][4], params[i][5]);
   //  printf("\n");

 /*     int s_in = params[11]; 
  int s_w = params[12];
  int s_out = params[13];
  int s_b = params[14];
  int s_in_pre = params[15];
  int s_w_pre = params[16];
  int s_out_pre = params[17];
  int s_b_pre = params[18]; 
  int s_indif_pre = params[19];
  int s_outdif_pre = params[20];*/

 // int rate;

  // for(int i =0; i<epoch; i++)
  // {
  //  rate = rates[i];
  //    printf("forward: ");
	 for(int j=0; j<layer_num; j++ )
	 {
       printf("forward %d-%d\n", j, params[j][10]);
    //    printf("inside training: layer %d: opcode: %d\n", j, params[j][10]);
      //  printf("debug0 %d\n", params[92][0]);
     /* printf("params: ");
        for(int m=0; m<16;m++ )
         printf(" %d ", params[j][m]);
        printf("\n");
      printf("sparams: ");
        for(int m=0; m<12;m++ )
         printf(" %lld ", sparams[j][m]);
        printf("\n");
       printf("%d\n", params[86][8]);*/
  /*      printf("offset: ");
        for(int m=0; m<8;m++ )
         printf(" %d ", offset[j][m]);
        printf("\n");
        printf("connect: ");
        for(int m=0; m<4;m++ )
         printf(" %d ", connect[j][m]);
        printf("\n");
*/
  //   for(int i = 0;i<layer_num;i++)
  //      printf(" -- %d : %d %d --", i, params[i][4], params[i][5]);
  //   printf("\n");
        /*       printf("before %d and %d %d %d %d %d:\ninput: ", j, params[j][10], offset[j][0], offset[j][4], connect[j][0], connect[j][1]);
               for(int i = 0; i< 50; i++ ) 
               printf(" %d ", (int)(*(data+offset[j][0]+i)));
               printf("\n"); 
               printf("output: ");
               for(int i = 0; i< 50; i++ ) 
               printf(" %d ", (int)(*(data+offset[j][4]+i)));
               printf("\n"); */
       switch(params[j][10])
       {
       case 0: 
              /* printf("%d and %d:\ninput: ", j, params[j][10]);
               for(int i = 0; i< 100; i++ ) 
               printf(" %d-%p ", (int)(*(data+offset[j][0]+i)), data+offset[j][0]+i);
               printf("\n"); */
               sparams[j][15-11] = sparams[connect[j][0]][17-11];
    	         conv( data+offset[j][0], wg+offset[j][1],
    		       data+offset[j][4], params[j], sparams[j]);
               //outfactor[j][0] = params[j][11]*params[j][12];
               //
             /*  printf("wg: ");
               for(int i = 0; i< 100; i++ ) 
               printf(" %d ", (int)(*(wg+offset[j][1]+i)));
               printf("\n");*/
               break;
       case 1:  //params[j][11] = outfactor[connect[j][0]][0];
                sparams[j][15-11] = sparams[connect[j][0]][17-11];
    	          batch(data+offset[j][0], data+offset[j][1]/*eps+q factor*/,
    		       data+offset[j][4], params[j], sparams[j]);
    	       // outfactor[j][0] = params[j][11];
    	        break;
       case 2:  sparams[j][15-11] = sparams[connect[j][0]][17-11];
    	        scale(data+offset[j][0], wg+offset[j][1], wg+offset[j][3],
    		    data+offset[j][4], params[j], sparams[j]) ;
    	        //outfactor[j][0] = params[j][11]*params[j][12];
    	        break;
       case 3:  sparams[j][15-11] = sparams[connect[j][0]][17-11];
    	        relu(data+offset[j][0],
    		    data+offset[j][4], params[j], sparams[j]);
    	        //outfactor[j][0] = params[j][15];
                break;
       case 4:  sparams[j][15-11] = sparams[connect[j][0]][17-11];
    	        pool(data+offset[j][0], wg+offset[j][1],
    		    data+offset[j][4], params[j], sparams[j]);
    	       // outfactor[j][0] = params[j][15];
                break;
       case 5:  
    	        sparams[j][21-11] = sparams[connect[j][1]][17-11];
    	        sparams[j][15-11] = sparams[connect[j][0]][17-11];
    	        eltwise(data+offset[j][0], data+offset[j][7],
    		    data+offset[j][4], params[j], sparams[j]);
    	        //outfactor[j][0] = params[j][11];
                break;
       case 6:  sparams[j][15-11] = sparams[connect[j][0]][17-11];
    	        fc(data+offset[j][0], wg+offset[j][1],wg+offset[j][3],
    		    data+offset[j][4], params[j], sparams[j]);
    	       // outfactor[j][0] = params[j][11]*params[j][12];
    	        break;
       case 7:  sparams[j][15-11] = sparams[connect[j][0]][17-11];
    	        softmax(data+offset[j][0], wg+offset[j][4], params[j], sparams[j],
   		    data+offset[j][2], (float*)(data+offset[j][1]));
    	      //  outfactor[j][1] = params[j][13];
       }
  /*             printf("after %d and %d %d %d %d %d:\ninput: ", j, params[j][10], offset[j][0], offset[j][4], connect[j][0], connect[j][1]);
               for(int i = 0; i< 50; i++ ) 
               printf(" %d-%f ", (int)(*(data+offset[j][0]+i)), (float)(*(data+offset[j][0]+i))/(float)sparams[j][15-11]);
               printf("\n"); 
               printf("output: ");
               for(int i = 0; i< 50; i++ ) 
               {
                 if(params[j][10]==7)
                 printf(" %d-%f-%f ", (int)(*(data+offset[j][2]+i)) , *((float*)(data+offset[j][1]+i)), 
                                    (float)(*(data+offset[j][2]+i))/(float)sparams[j][19-11]);
                 else 
                 printf(" %d-%f ", (int)(*(data+offset[j][4]+i)), (float)(*(data+offset[j][4]+i))/(float)sparams[j][17-11]);
               }printf("\n"); 
*/
  //     printf("debug %d\n", params[92][0]);
//     for(int i = 0;i<layer_num;i++)
 //       printf(" -- %d : %d %d --", i, params[i][4], params[i][5]);
 //    printf("\n");

 }
     //  printf("\n");	
     
       
    //  printf("backward: ");
	 for(int j=layer_num-2; j>=0; j--)
	 {
       printf("backward %d-%d\n", j, params[j][10]);
    //    printf("inside training back: layer %d: opcode: %d\n", j, params[j][10]);
    /*   printf("params: ");
        for(int m=0; m<16;m++ )
         printf(" %d ", params[j][m]);
        printf("\n");
       printf("%d\n", params[86][8]);*/
 /*       printf("offset: ");
        for(int m=0; m<8;m++ )
         printf(" %d ", offset[j][m]);
        printf("\n");
        printf("connect: ");
        for(int m=0; m<4;m++ )
         printf(" %d ", connect[j][m]);
        printf("\n");
*/
      // switch(params[j][10])
              
            /*  printf("before %d and %d:\ninput: ", j, params[j][10]);
              for(int i = 0; i< 100; i++ ) 
               printf(" %d-%f ", (int)(*(data+offset[j][5]+i)), (float)(*(data+offset[j][5]+i))/(float)sparams[j][20-11]);
               printf("\n"); */
             /*  printf("output: ");
               for(int i = 0; i< 100; i++ ) 
               printf(" %d ", (int)(*(data+offset[j][2]+i)), (float)(*(data+offset[j][2]+i))/(float)sparams[j][19-11]);
               printf("\n"); */
		switch(params[j][10])
		{
		case 0: sparams[j][20-11] = sparams[connect[j][2]][19-11];
			      conv_back(data + offset[j][5], wg+offset[j][1], data+offset[j][2],
				params[j], sparams[j]);
		        weight_back(data + offset[j][5], wg+offset[j][1], data+offset[j][0], 
            data+offset[j][6], params[j], sparams[j], rate, decay);



            //    outfactor[j][1] = params[j][12]*params[j][13];
                break;
		case 1: sparams[j][20-11] = sparams[connect[j][2]][19-11];
			    batch_back(data+ offset[j][4], data+ offset[j][2], data+offset[j][1],
			    data+ offset[j][5], params[j], sparams[j]);
			    //outfactor[j][1] = params[j][15];
                break;
		case 2:  sparams[j][20-11] = sparams[connect[j][2]][19-11];
			     scale_back(data+offset[j][2], wg+offset[j][1],
				 data+offset[j][5], params[j], sparams[j]);
			     weight_scale(data+offset[j][0], wg+offset[j][1],
				 data+offset[j][5],  data+offset[j][6], params[j], sparams[j], rate, decay);
			     bias_scale(wg+offset[j][3], data+offset[j][5], data+offset[j][7],
			    		 params[j], sparams[j], rate);
			//     outfactor[j][1] = params[j][12]*params[j][13];
                 break;
		case 3: sparams[j][20-11] = sparams[connect[j][2]][19-11];
		        sparams[j][21-11] = sparams[connect[j][3]][19-11];
			      relu_back(data+offset[j][0],
			       data+offset[j][5], data+offset[j][6], data+offset[j][2], params[j], sparams[j]);
			   // outfactor[j][1] = params[j][15];
		        break;
		case 4: sparams[j][20-11] = sparams[connect[j][2]][19-11];
            params[j][21] = params[connect[j][3]][19];
			    back_pool(data+offset[j][2], wg+offset[j][1],
			    data+offset[j][5], data+offset[j][6], params[j], sparams[j]);
			  //  outfactor[j][1] = params[j][15];
			    break;
		case 5: sparams[j][20-11] = sparams[connect[j][2]][19-11];
			    eltwise_back(data+offset[j][5], data+offset[j][2],
				params[j], sparams[j]);
			   // outfactor[j][1] = params[j][15];
		        break;
		case 6: sparams[j][20-11] = sparams[connect[j][2]][19-11];
        //    params[j][21] = params[connect[j][3]][19];
			    back_fc(data+offset[j][5], wg+offset[j][1],
			    data+offset[j][2], params[j], sparams[j]);
			    fc_weight(data+offset[j][5], wg+offset[j][1],
				    data+offset[j][0], data+offset[j][6], params[j], sparams[j], rate, decay);
			    bias_back_fc(data+offset[j][5], wg+offset[j][3], data+offset[j][7],
					params[j], sparams[j], rate);
             /*  printf("%d and %d:\ninput: ", j, params[j][10]);
               for(int i = 0; i< 100; i++ ) 
               printf(" %d ", (int)(*(data+offset[j][5]+i)));
               printf("\n"); 
               printf("output: ");
               for(int i = 0; i< 100; i++ ) 
               printf(" %d ", (int)(*(data+offset[j][2]+i)));
               printf("\n"); 
               for(int i = 0; i<100; i++)
               printf(" %d ", (int)(*(data+offset[j][1]+i)));
               printf("\n"); 
               for(int i = 0; i<100; i++)
               printf(" %d ", (int)(*(data+offset[j][3]+i)));
               printf("\n"); 
               */
			    //outfactor[j][1] = params[j][12]*params[j][13];
			    break;

		}
              /* printf("after %d and %d:\ninput: ", j, params[j][10]);
               for(int i = 0; i< 100; i++ ) 
               printf(" %d-%f ", (int)(*(data+offset[j][5]+i)), (float)(*(data+offset[j][5]+i))/(float)sparams[j][20-11]);
               printf("\n"); */
              // if(params[j][10] == 6)
             /*  printf("output: ");
               for(int i = 0; i< 10000; i++ ) 
               printf(" %d-%f ", (int)(*(data+offset[j][2]+i)), (float)(*(data+offset[j][2]+i))/(float)sparams[j][19-11]);
               printf("\n"); */
              
	 }
  //     printf("\n");	
//}
 }

















