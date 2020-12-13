#-*- coding:utf-8 -*-
import urllib3
import json
import requests
import numpy as np

def get_image(x, y):

    openApiURL = "http://api.vworld.kr/req/image" 

    request="getmap"
    service="image"
    key = "38E7C820-0357-3E0B-83F2-731A5D06E6A6"
    zoom="18"
    size="1024,1024"
    crs="EPSG:4326"
    basemap="PHOTO"
    Param={"service":service,"request":request,"key":key,"zoom":zoom,"size":size,"basemap":basemap,"crs":crs}
    Param["center"]=str(x)+","+str(y)
    # GET
    response=requests.get(openApiURL,Param)
    return response.content
