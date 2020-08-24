#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 16:52:20 2020

@author: veraisice
"""
import numpy as np
def find_num(x,y,z):
    count = 0
    n=list(range(y,z+1))
    
    for i in n:
        if  set(int(m)for m in str(i*x)).isdisjoint(int(j) for j in str(i))== True:
            count +=1
            
        else :
            count +=0
    return count

def money(l):
  x = np.zeros(len(l))
  for i in range(len(l)):
      x[i] = sum(l[:i+1])
  if min(x)>0:
     x0 =0
  else:
      x0 = max(x*(-1))+1
    
 
  return x0




def returnx(b):
    for i in b:
        if i ==0:
           print("YES")
        else:
            print("NO")


