
# coding: utf-8
import os
def eachFile(filepath):                 #将目录内的文件名放入列表中
    pathDir =  os.listdir(filepath)
    out = []
    for allDir in pathDir:
        print(allDir)
        #child = allDir.decode('gbk')    # .decode('gbk')是解决中文显示乱码问题
        out.append(allDir)
    return out

from keras.utils import np_utils, conv_utils

# input 
#原圖大小為64*64
Width = 64
Height = 64
num_classes = 2  # 分兩類
pic_dir_data = r'D:\Data'  #data path


import cv2
import numpy as np
import matplotlib.image as mpimg

def get_data(train_percentage=1.0,resize=True,data_format=None):   #从文件夹中获取图像数据
    #file_name = os.path.join(pic_dir_out,data_name +".pkl")   
    #if os.path.exists(file_name):           #判断之前是否有存到文件中
        #(X_train, y_train), (X_test, y_test) = cPickle.load(open(file_name,"rb"))
        #return (X_train, y_train), (X_test, y_test)  
    data_format = conv_utils.normalize_data_format(data_format)
    print(data_format)
    pic_dir_set = eachFile(pic_dir_data)  
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    label = 0
    for pic_dir in pic_dir_set:
        print(pic_dir_data+pic_dir)
        if not os.path.isdir(os.path.join(pic_dir_data,pic_dir)):
            continue    
        pic_set = eachFile(os.path.join(pic_dir_data,pic_dir))
        pic_index = 0
        train_count = int(len(pic_set)*train_percentage)
        for pic_name in pic_set:
            if not os.path.isfile(os.path.join(pic_dir_data,pic_dir,pic_name)):
                continue
            #img = cv2.imread(os.path.join(pic_dir_data,pic_dir,pic_name))
            img = mpimg.imread(os.path.join(pic_dir_data,pic_dir,pic_name))

            #img = cv2.imread(os.path.join(pic_dir_data,pic_dir,pic_name),0)
            if img is None:
                continue
            #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
            #if (resize):
                #img = cv2.resize(img,(Width,Height))
            #if (data_format == 'channels_last'):
                #img = img.reshape(-1,Width,Height,1)
           # elif (data_format == 'channels_first'):
               # img = img.reshape(-1,1,Width,Height)
			   
			## 依照片位置做影像處理
            if('Bottom_Right.jpg' in pic_name):
                img = cv2.imread(os.path.join(pic_dir_data,pic_dir,pic_name)) #影像讀取
                img = img[20:60,0:50] #讀取某一範圍
                ret,img = cv2.threshold(img,170,255,cv2.THRESH_BINARY) #二值化
                img = cv2.resize(img, (64, 64), interpolation = cv2.INTER_AREA) ## 圖像resize 至64*64,並使用像素区域关系进行重采样
                img = img / 255.0  #正規化
                                       
                       
            if('Bottom_Left.jpg' in pic_name):
                img = cv2.imread(os.path.join(pic_dir_data,pic_dir,pic_name))
                img = img[32:55,10:42]
                  
                ret,img = cv2.threshold(img,170,255,cv2.THRESH_BINARY)
                img = cv2.resize(img, (64, 64), interpolation = cv2.INTER_AREA)
                img = img / 255.0
             
                
            if('Up_Right' in pic_name):
                img = cv2.imread(os.path.join(pic_dir_data,pic_dir,pic_name))
                img = img[30:60,10:50]
                  
                ret,img = cv2.threshold(img,170,255,cv2.THRESH_BINARY)
                img = cv2.resize(img, (64, 64), interpolation = cv2.INTER_AREA)
                img = img / 255.0
               
            if('Up_Left' in pic_name):
                img = cv2.imread(os.path.join(pic_dir_data,pic_dir,pic_name))
                img = img[30:56,0:50]
                  
                ret,img = cv2.threshold(img,170,255,cv2.THRESH_BINARY)
                img = cv2.resize(img, (64, 64), interpolation = cv2.INTER_AREA)
                img = img / 255.0
               
                
            if('XRAY_Up.jpg' in pic_name):
                img = cv2.imread(os.path.join(pic_dir_data,pic_dir,pic_name))
                img = img[0:40,20:50]
                img = cv2.resize(img, (64, 64), interpolation = cv2.INTER_AREA)
                img = img / 255.0
                
            
            if('XRAY_Bottom.jpg' in pic_name):
                img = cv2.imread(os.path.join(pic_dir_data,pic_dir,pic_name))
                img = img[20:50,0:40]
                img = cv2.resize(img, (64, 64), interpolation = cv2.INTER_AREA)
                img = img / 255.0
                        #print(img.shape)

            if "pass" in os.path.join(pic_dir_data,pic_dir,pic_name).lower():
                label = 0
            else:
                label = 1
                
                
            if (pic_index < train_count):
                X_train.append(img)
                y_train.append(label)          
            else:
                X_test.append(img)
                y_test.append(label)
            pic_index += 1
            print(os.path.join(pic_dir_data,pic_dir,pic_name), label)
        # if len(pic_set) > 0:
        #     label += 1
    X_train = np.array(X_train)        
    X_test = np.array(X_test)    
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    #cPickle.dump([(X_train, y_train), (X_test, y_test)],open(file_name,"wb")) 
    return X_train, y_train, X_test, y_test 


X_train, y_train,X_test, y_test = get_data(0.9,data_format='channels_last')


# import keras packages
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D
from keras.optimizers import SGD
from keras.utils import to_categorical
import keras


# batch size for gradient descent
batch_size = 32
# number of classes
num_classes = 2
# number of epochs (1 epoch = amount of iterations that covers the whole training set)
epochs = 100 # try a larger number of epochs here (for example 10 or larger)

# input image dimensions
nmb_samples, img_rows, img_cols = X_train.shape[0],X_train.shape[1], X_train.shape[2]
nmb_test_samples = X_test.shape[0]

## 將類向量(整數)轉成二進制類矩陣
y_train = keras.utils.to_categorical(np.squeeze(y_train), num_classes)
y_test = keras.utils.to_categorical(np.squeeze(y_test), num_classes)


from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization
from keras.optimizers import Adam
from keras import regularizers

# 建立簡單的線性執行的模型
model = Sequential()

model.add(Conv2D(128, (5, 5), strides = 2,  #第一次卷積128*30*30.卷積層 input 128個神經元, strides = 2 
                     padding='valid',  #採用丟棄的方式
                     activation = 'relu', #激活函數
                     input_shape=[img_rows, img_cols, 3], name = 'conv1'))

model.add(Conv2D(128, (5, 5), strides = 2,  #第二次卷積#128*13*13
                     padding='valid', 
                     activation = 'relu',
                     name = 'conv2'))


model.add(MaxPooling2D(pool_size=(13, 13), strides= 13 )) #Pooling 層 結果輸出為 128*1*1

model.add(Flatten()) #輸入"壓平"，即把多维的输入一维化。把Max pooling的結果, 拉直成vector, 往下塞給下一階段

model.add(Dropout(.5)) #隨機丟掉神經元，避免過擬和
model.add(Dense(2))  # 輸出結果是2個類別，所以維度是2
model.add(Dropout(.3)) #随機扔掉當前層一些weight
model.add(Activation('softmax')) # 最後一層用softmax作為激活函數，做分類用
model.summary() # 輸出模型摘要資訊

#以compile函數定義損失函數(loss)、優化函數(optimizer)及成效衡量指標(mertrics)
#優化器為Adam，loss function選用了categorical_crossentropy
model.compile(loss=keras.losses.categorical_crossentropy, 
             optimizer=Adam(),
              metrics=['accuracy'])


# training 進行訓練, 訓練過程會存在history的變數中
history = model.fit(X_train, y_train,
              batch_size=batch_size, #批大小，指定進行梯度下降時每個batch包含的樣本數
              epochs=100,#訓練模型迭代輪次
              verbose=1, #輸出進度條記錄
              validation_data=(X_test, y_test)) #用来評估損失，以及在每輪結束時的任何模型度量指標



# save model 
model.save('./all_drop.h5')