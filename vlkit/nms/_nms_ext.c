/*
 * Licensed under The MIT License
 * Written by KAI-ZHAO and Shanghua-Gao
 */
#include "nms.h"
#include <math.h>
#include<stdio.h>
#define SUB2IND(x,y) y*w+x
float interp( float *I, int h, int w, float x, float y ) {
  x = x<0 ? 0 : (x>w-1.001 ? w-1.001 : x);
  y = y<0 ? 0 : (y>h-1.001 ? h-1.001 : y);
  int x0=(int)x;
  int  y0=(int)y;
  int x1=x0+1;
  int y1=y0+1;
  float dx0=x-x0, dy0=y-y0;
  float dx1=1-dx0, dy1=1-dy0;
  float ret = I[SUB2IND(x0, y0)]*dx1*dy1 + I[SUB2IND(x1, y0)]*dx0*dy1 +
         I[SUB2IND(x0, y1)]*dx1*dy0 + I[SUB2IND(x1, y1)]*dx0*dy0;
  return ret;
}

void nms(float * in_array1, float * in_array2, float * out_array, int h, int w){
    int r=1,s=5;
    float m=1.01;
    for( int x=0; x<w; x++ ) for( int y=0; y<h; y++ ) {
    float e=out_array[SUB2IND(x,y)]=in_array1[SUB2IND(x,y)]; if(!e) continue; e*=m;
    float coso=cos(in_array2[SUB2IND(x,y)]), sino=sin(in_array2[SUB2IND(x,y)]);
    for( int d=-r; d<=r; d++ ) if( d ) {
      float e0 = interp(in_array1,h,w,x+d*coso,y+d*sino);
      if(e < e0) { out_array[SUB2IND(x,y)]=0; break; }
    }
  }

  // suppress noisy edge estimates near boundaries
  s=s>w/2?w/2:s; s=s>h/2? h/2:s;
  for( int x=0; x<s; x++ ) for( int y=0; y<h; y++ ) {
    out_array[SUB2IND(x,y)]*=x/(float)s; out_array[SUB2IND((w-1-x),y)]*=x/(float)s; }
  for( int x=0; x<w; x++ ) for( int y=0; y<s; y++ ) {
    out_array[SUB2IND(x,y)]*=y/(float)s; out_array[SUB2IND(x, (h-1-y))]*=y/(float)s;}
}
