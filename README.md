# Is_iphone_Clean
keras，yolo，cnn，classification

项目描述
========
>
1 手机天线处经常会有杂质，或者污染，通过aoi拍摄照片，运用深度学习自动判断是否有污染、杂质。
>
2 手机天线共有6个方向处需要判断，但也是二分类问题，pass和fail。
>
3 图片数据为6个方向裁剪的数据，并且需要对图片进行预处理。
>
4 使用自己搭建的cnn，vgg，all-cnn，resnet等模型进行建模比较。
>
5 使用labelimg标记图片，用yolo3进行训练，找出污染或者杂质所在的位置。

原图范例以及更详细的描述
=========
>
![github](https://github.com/Juary88/Is_iphone_Clean/blob/master/pic/question.png)
>

模型训练结果
=========
![github](https://github.com/Juary88/Is_iphone_Clean/blob/master/pic/lenet.png)
>
![github](https://github.com/Juary88/Is_iphone_Clean/blob/master/pic/confision.png)
>

yolo模型结果（目前只标注了100多张，定位位置有偏差）
=========
![github](https://github.com/Juary88/Is_iphone_Clean/blob/master/pic/yolo.png)
