import cv2
from cv2 import imread,cvtColor,imshow,waitKey,pyrDown,pyrUp
import numpy as np

class Cv():
    def convertGray(self,img_rgb):
        #灰度转换的自定义实现
        img_gray = img_rgb[:,:,0] * 0.11 + img_rgb[:,:,1] * 0.59 + img_rgb[:,:,2] * 0.3
        img_gray = img_gray.astype(np.uint8)##只有当数组类型为uint8时，opencv才会认为这是图片
        return img_gray


if __name__=='__main__':
    im1=imread('1.jpg')
    im2=imread('2.jpg')
    gray1=cvtColor(im1,cv2.COLOR_BGR2GRAY)
    gray2=cvtColor(im2,cv2.COLOR_BGR2GRAY)

    g=gray1.copy()
    gp1=[g]#高斯金字塔1
    for i in range(6):
        g=pyrDown(g)
        gp1.append(g)
    
    g=gray2.copy()
    gp2=[g]#高斯金字塔1
    for i in range(6):
        g=pyrDown(g)
        gp2.append(g)





    imshow('1',gray1)
    imshow('2',gray2)
    waitKey(1000)

    