# coding: utf-8

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D
from keras.optimizers import SGD
from keras.utils import to_categorical
import pandas as pd
import keras
import cv2

import os
def eachFile(filepath):                 #将目录内的文件名放入列表中
    pathDir =  os.listdir(filepath)
    out = []
    for allDir in pathDir:
        #print(allDir)
        #child = allDir.decode('gbk')    # .decode('gbk')是解决中文显示乱码问题
        out.append(allDir)
    return out




import numpy as np
import matplotlib.image as mpimg

sv = []
res = []
up = []
bottom = []
up_right = []
up_left = []
bottom_right = []
bottom_left = []
location = ['up', 'buttom', 'up_right', 'up_left', 'bottom_right', 'bottom_left']

def predict_data(pic_dir_data,data_format=None):   #从文件夹中获取图像数据
    pre = []
    true = []
    des = []

    index = 0
    #bottom_right = up = bottom = up_right = up_left = bottom_left = 0
	## load model
    model = keras.models.load_model('./all_drop.h5')
    pic_dir_set = eachFile(pic_dir_data)  
    #print(pic_dir_set)
	## 依照片位置做影像處理
    for pic_dir in pic_dir_set:
        #print(pic_dir_data+pic_dir)
        if not os.path.isdir(os.path.join(pic_dir_data,pic_dir)):
            continue    
        pic_set = eachFile(os.path.join(pic_dir_data,pic_dir))
        
        for pic_name in pic_set:
            pic = pic_name.split(".")[1]
            if pic not in sv:
                sv.append(pic)
            if(index == 0):
                true.append(1)
            else:
                true.append(0)
            print(pic_name)
            
            if not os.path.isfile(os.path.join(pic_dir_data,pic_dir,pic_name)):
                continue
            if('Bottom_Right.jpg' in pic_name):
                img = cv2.imread(os.path.join(pic_dir_data,pic_dir,pic_name))
                img = img[20:60,0:50]
                ret,img = cv2.threshold(img,170,255,cv2.THRESH_BINARY)
                img = cv2.resize(img, (64, 64), interpolation = cv2.INTER_AREA)
                img = img / 255.0
                        #print(img.shape)
                img = np.expand_dims(img,axis = 0)
                        
                prediction = np.argmax(model.predict(img),axis = 1)
                if(prediction[0] == 1):
                    des.append({pic_name:'Bottom_Right fail'})
                    bottom_right.append('fail')
                else:
                    des.append({pic_name:'Bottom_Right pass'})
                    bottom_right.append('pass')
                        #print(prediction)
                pre.extend(prediction)
                        
                       
            if('Bottom_Left.jpg' in pic_name):
                img = cv2.imread(os.path.join(pic_dir_data,pic_dir,pic_name))
                img = img[32:55,10:42]
                  
                ret,img = cv2.threshold(img,170,255,cv2.THRESH_BINARY)
                img = cv2.resize(img, (64, 64), interpolation = cv2.INTER_AREA)
                img = img / 255.0
                        #print(img.shape)
                img = np.expand_dims(img,axis = 0)
                        
                prediction = np.argmax(model.predict(img),axis = 1)
                if(prediction[0] == 1):
                    des.append({pic_name:'Bottom_Left fail'})
                    bottom_left.append('fail')
                else:
                    des.append({pic_name:'Bottom_Left pass'})
                    bottom_left.append('pass')
                        #print(prediction)
                pre.extend(prediction)
                
            if('Up_Right' in pic_name):
                img = cv2.imread(os.path.join(pic_dir_data,pic_dir,pic_name))
                img = img[30:60,10:50]
                  
                ret,img = cv2.threshold(img,170,255,cv2.THRESH_BINARY)
                img = cv2.resize(img, (64, 64), interpolation = cv2.INTER_AREA)
                img = img / 255.0
                        #print(img.shape)
                img = np.expand_dims(img,axis = 0)
                if model.predict(img)[0][1] > 0.6:
                    prediction[0] == 1
                else:
                    prediction[0] == 0
                #prediction = np.argmax(model.predict(img),axis = 1)
                if(prediction[0] == 1):
                    des.append({pic_name:'Up_Right fail'})
                    up_right.append('fail')
                else:
                    des.append({pic_name:'Up_Right pass'})
                    up_right.append('pass')
                        #print(prediction)
                pre.extend(prediction)
            
            if('Up_Left' in pic_name):
                img = cv2.imread(os.path.join(pic_dir_data,pic_dir,pic_name))
                img = img[30:56,0:50]
                  
                ret,img = cv2.threshold(img,170,255,cv2.THRESH_BINARY)
                img = cv2.resize(img, (64, 64), interpolation = cv2.INTER_AREA)
                img = img / 255.0
                        #print(img.shape)
                img = np.expand_dims(img,axis = 0)

                if model.predict(img)[0][1] > 0.6:
                    prediction[0] == 1
                else:
                    prediction[0] == 0
                #prediction = np.argmax(model.predict(img),axis = 1)
                if(prediction[0] == 1):
                    des.append({pic_name:'Up_Left fail'})
                    up_left.append('fail')
                else:
                    des.append({pic_name:'Up_Left pass'})
                    up_left.append('pass')
                        #print(prediction)
                pre.extend(prediction)
                
            if('XRAY_Up.jpg' in pic_name):
                img = cv2.imread(os.path.join(pic_dir_data,pic_dir,pic_name))
                img = img[0:40,20:50]
                  
                img = cv2.resize(img, (64, 64), interpolation = cv2.INTER_AREA)
                img = img / 255.0
                        #print(img.shape)
                img = np.expand_dims(img,axis = 0)
                        
                prediction = np.argmax(model.predict(img),axis = 1)
                if(prediction[0] == 1):
                    des.append({pic_name:'Up fail'})
                    up.append('fail')
                else:
                    des.append({pic_name:'Up pass'})
                    up.append('pass')
                        #print(prediction)
                pre.extend(prediction)
            
            if('XRAY_Bottom.jpg' in pic_name):
                img = cv2.imread(os.path.join(pic_dir_data,pic_dir,pic_name))
                img = img[20:50,0:40]
                  
                img = cv2.resize(img, (64, 64), interpolation = cv2.INTER_AREA)
                img = img / 255.0
                        #print(img.shape)
                img = np.expand_dims(img,axis = 0)
                        
                prediction = np.argmax(model.predict(img),axis = 1)
                if(prediction[0] == 1):
                    des.append({pic_name:'Bottom fail'})
                    bottom.append('fail')
                else:
                    des.append({pic_name:'Bottom pass'})
                    bottom.append('pass')
                        #print(prediction)
                pre.extend(prediction)
        index = index + 1
    return pre,des,true

pre,des,true = predict_data('D:/test_data_all')
pre = np.array(pre)
len(np.where((pre == 0))[0])
from sklearn.metrics import classification_report
print(classification_report(true, pre))
#model = keras.models.load_model('./all.h5')

# 輸出csv檔
dic = {"a_sv":sv,
        "up":up,
         "bottom":bottom,
         "up_left":up_left,
         "up_right":up_right,
         "bottom_left":bottom_left,
         "bottom_right":bottom_right}
try:
    data = pd.DataFrame(dic)
    data.to_csv("./res.csv")
except IOError:
    print("Error: 各個方向圖片數量不同，請確保各個方向都有圖片")
else:
    print("結果生成文件成功")



