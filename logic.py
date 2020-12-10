# -*- coding: utf-8 -*-
"""Untitled8.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qZjfnvKJ1MNDIOSBajI2BkI0cLV-DEDo
"""

import numpy as np
from PIL import Image

'''
테스트를 위해서 드라이브에서 이미지를 받아옴
실제 어플리케이션에선 segmentation된 이미지를 받아올것
'''

imgpath = '/content/drive/MyDrive/Samsung_Photo6.png'
im = Image.open(imgpath).convert('RGBA').convert('RGB')
imnp = np.array(im)
h,w = imnp.shape[:2]




colours, counts = np.unique(imnp.reshape(-1,3), axis=0, return_counts=1)
for ind, x in enumerate(colours):
  counts[ind] = 0


#100~900사이의 값은 중앙 그 외는 외곽으로 판정
for height in range(h):
  for width in range(w):
    if height >= 100 and height <= 900 and width >= 100 and width <= 900:
      i = np.where((colours == imnp[height][width]).all(axis=1))
      counts[i[0][0]] += 1
    else:
      i = np.where((colours == imnp[height][width]).all(axis=1))
      counts[i[0][0]] += 0.7


env_ret = 0.0
conv_ret = 0.0

syn = 0


for index, colour in enumerate(colours):
    count = counts[index]
    
    proportion = (100 * count) / ((h-30) * (w-30)) #외곽 픽셀 0.7 반영
    print(f"   Colour: {colour}, count: {count}, proportion: {proportion:.2f}%")
    if(np.array_equal(colour,np.array([0,125,0]))): #녹지
      env_ret += proportion
      syn += 1
    elif(np.array_equal(colour,np.array([100,100,100]))): #공장
      env_ret -= proportion * 1.25 / 2
    elif(np.array_equal(colour,np.array([150,150,250]))): #대형건물
      if(proportion <= 20):
        conv_ret += proportion
      else:
        conv_ret += 20
    elif(np.array_equal(colour,np.array([0,0,0]))): #도로
      env_ret -= proportion * 1.25 / 10
    elif(np.array_equal(colour, np.array([0,0,150]))): #강
      env_ret += proportion
      syn += 1

if(syn == 2): #강 + 녹지 시너지
  env_ret += 5
  
print("쾌적도 : ", env_ret+conv_ret)