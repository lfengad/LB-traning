#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>
#include "ap_int.h"
#include "math.h"
#include "conv.h"

#define PARA 16


void rand_(uchar_* lfsr) {
  ubit_ b_7 = (*lfsr).get_bit(7);
  ubit_ b_5 = (*lfsr).get_bit(5);
  ubit_ b_4 = (*lfsr).get_bit(4);
  ubit_ b_3 = (*lfsr).get_bit(3);
  bool new_bit = b_7 ^ b_5 ^ b_4 ^ b_3;
  (*lfsr) = (*lfsr) << 1;
  (*lfsr).set_bit(0, new_bit);
}

void memcpy_int(int_* to, int_* from, int size)
{
  for(int i=0;i<size;i++)
    to[i] = from[i];
}

void memset_int(int_* to, int size)
{
  for(int i=0;i<size;i++)
    to[i] = 0;
}


ubit_ round_(uchar_ value, uchar_* lfsr)
{
	 rand_(lfsr);
	 return value>(*lfsr);
}

char_ input_qt( int_ input, int s_in_pre, int s_in, ubit_ * larger, ubit_ * smaller)
{
	 int_ tmp_int;
         if(s_in + 2 > s_in_pre)     
            tmp_int = input << (s_in + 2 -s_in_pre);
         else 
           tmp_int = input >> (-(s_in + 2 -s_in_pre));
         
       int_ tmp_int0 = tmp_int.range(31,1) + tmp_int.range(0,0);
       int_ tmp_int1 = tmp_int.range(31,2) + tmp_int.range(1,1);
				
                if(tmp_int1>127 || tmp_int1<-128)
			*larger = 1;
		if(tmp_int0>127 || tmp_int0<-128)
			*smaller = 0;

                      if(tmp_int1 > 127)
                         tmp_int1 = 127;
                      else if (tmp_int1 < -128)
                         tmp_int1 = -128;

                 return  tmp_int1.range(7,0);
}


char_ outdiff_qt( int_ outdiff, int s_outdif_pre, int s_out, ubit_ * larger, ubit_ *smaller, uchar_ *seed)
{

     		 int_ tmp_int;
     		 if(s_out+9>s_outdif_pre)
                 tmp_int = outdiff << (s_out+9-s_outdif_pre);
     		 else
                 tmp_int = outdiff >> (-(s_out+9-s_outdif_pre));

     		 int_ tmp_int0 = tmp_int.range(31,8) + round_(tmp_int.range(7,0), seed);
                 int_ tmp_int1 = tmp_int.range(31,9) + round_(tmp_int.range(8,1), seed);
              if(tmp_int1>127 || tmp_int1<-128)
              {
                  *larger = 1;
              }
              if(tmp_int0>127 || tmp_int0<-128)
              {
             	 *smaller = 0;
              }
              if(tmp_int1>127)
                 tmp_int1 = 127;
              else if (tmp_int1<-128)
                 tmp_int1 = -128;

              return tmp_int1.range(7,0);
}


int_ momentb_qt( int_ moment_buf, int_ weight_buf, int s_moment, int s_moment_pre,  int s_weight, ubit_ * larger, ubit_ * smaller)
{
     long_ tmp0, tmp00;  
     
     if(s_moment > (s_weight))
             tmp0 = (((long_)weight_buf) << (s_moment - s_weight));             
      else
             tmp0 = (weight_buf) >> (s_weight - s_moment);  

      if(s_moment > s_moment_pre)
           tmp00 = moment_buf << (s_moment - s_moment_pre);
      else   
           tmp00 = moment_buf >> (s_moment_pre - s_moment);
  
           long_ tmp1 = ((tmp00*9 + tmp0)<<2)/10;
           
       	   long_ tmp2 = tmp1.range(63,1) + tmp1.range(0,0);
	   long_ tmp3 = tmp1.range(63,2) + tmp1.range(1,1);
    
            if(tmp3> 2e32-1 || tmp3<-2e32)
            {
	        *larger = 1;
            }
      	    if(tmp2>2e32-1 || tmp2<-2e32)
      	    {
      	        *smaller = 0;
      	    }
            if(tmp3 > 2e32-1)
               tmp3 = 2e32-1;
            else if(tmp3 < -2e32)
               tmp3 = -2e32; 
           
          return tmp3.range(31, 0);           
}








int_ moment_qt( int_ moment_buf, int_ weight_buf, int s_moment, int s_moment_pre,  int s_weight, ubit_ * larger, ubit_ * smaller)
{
     long_ tmp0, tmp00;  
     
     if(s_moment > (s_weight))
             tmp0 = (((long_)weight_buf) << (s_moment - s_weight))/img_num;             
      else
             tmp0 = (weight_buf/img_num) >> (s_weight - s_moment);  

      if(s_moment > s_moment_pre)
           tmp00 = moment_buf << (s_moment - s_moment_pre);
      else   
           tmp00 = moment_buf >> (s_moment_pre - s_moment);
  
           long_ tmp1 = ((tmp00*9 + tmp0)<<2)/10;
           
       	   long_ tmp2 = tmp1.range(63,1) + tmp1.range(0,0);
	   long_ tmp3 = tmp1.range(63,2) + tmp1.range(1,1);
    
            if(tmp3> 2e32-1 || tmp3<-2e32)
            {
	        *larger = 1;
            }
      	    if(tmp2>2e32-1 || tmp2<-2e32)
      	    {
      	        *smaller = 0;
      	    }
            if(tmp3 > 2e32-1)
               tmp3 = 2e32-1;
            else if(tmp3 < -2e32)
               tmp3 = -2e32; 
           
          return tmp3.range(31, 0);           
}

char_ weight_qt( int_ moment, char_ weight, int s_moment, int s_w_pre, int s_w, ubit_ * larger, ubit_ * smaller, int rate, int decay )
{

          int w_decay = decay - 1;  
          long_ tmp_int, tmpw, tmpg;
          if(s_w+2 > s_w_pre)
             tmpw = (long_)weight << (s_w + 2 - s_w_pre);
          else
             tmpw = weight >> (s_w_pre - s_w -2);    
          
          tmpw = tmpw * w_decay/decay;
           
          if(s_w+2 > s_moment)             
              tmpg = (((long_)moment) << (s_w + 2 - s_moment))/rate;
          else      
              tmpg = (moment/rate) >> (s_moment - 2 - s_w);

           tmp_int = tmpw - tmpg;
      
       	   int_ tmp_int0 = tmp_int.range(31,1) + tmp_int.range(0,0);
	   int_ tmp_int1 = tmp_int.range(31,2) + tmp_int.range(1,1);

            if(tmp_int1>127 || tmp_int1<-128)
            {
	        *larger = 1;
             }
      	    if(tmp_int0>127 || tmp_int0<-128)
      	     {
      		 *smaller = 0;
      	      }
            if(tmp_int1>127)
                  tmp_int1 = 127;
            else if(tmp_int1<-128)
                  tmp_int1 = -128;

            return tmp_int1.range(7,0);

}

char_ bias_qt( int_ moment, char_ weight, int s_moment, int s_w_pre, int s_w, ubit_ * larger, ubit_ * smaller, int rate)
{
          long_ tmp_int, tmpw, tmpg;
          if(s_w+2 > s_w_pre)
             tmpw = (long_)weight << (s_w + 2 - s_w_pre);
          else
             tmpw = weight >> (s_w_pre - s_w -2);    
          
           
          if(s_w+2 > s_moment)             
              tmpg = (((long_)moment) << (s_w + 2 - s_moment))/rate;
          else      
              tmpg = (moment/rate) >> (s_moment - 2 - s_w);

           tmp_int = tmpw - tmpg;
      
       	   int_ tmp_int0 = tmp_int.range(31,1) + tmp_int.range(0,0);
	   int_ tmp_int1 = tmp_int.range(31,2) + tmp_int.range(1,1);

            if(tmp_int1>127 || tmp_int1<-128)
            {
	        *larger = 1;
             }
      	    if(tmp_int0>127 || tmp_int0<-128)
      	     {
      		 *smaller = 0;
      	      }
            if(tmp_int1>127)
                  tmp_int1 = 127;
            else if(tmp_int1<-128)
                  tmp_int1 = -128;

            return tmp_int1.range(7,0);

}
