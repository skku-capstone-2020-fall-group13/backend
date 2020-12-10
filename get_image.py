#-*- coding:utf-8 -*-
import urllib3
import json
import requests
import numpy as np

def get_image(x, y):

    openApiURL = "http://api.vworld.kr/req/image" 

    request="getmap"
    service="image"
    key = "2CCCA2EC-0108-3305-94F7-16446DB16433"
    zoom="16"
    size="1024,1024"
    crs="EPSG:4326"
    basemap="PHOTO"
    Param={"service":service,"request":request,"key":key,"zoom":zoom,"size":size,"basemap":basemap,"crs":crs}
    Param["center"]=str(x)+","+str(y)
    print(Param)
    # GET
    response=requests.get(openApiURL,Param)
    return response.content