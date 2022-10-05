import cv2
from cv2 import imread,cvtColor,imshow,waitKey,pyrDown,pyrUp,subtract,add
import numpy as np

class Cv():
    def convertGray(self,img_rgb):
        #灰度转换的自定义实现
        img_gray = img_rgb[:,:,0] * 0.11 + img_rgb[:,:,1] * 0.59 + img_rgb[:,:,2] * 0.3
        img_gray = img_gray.astype(np.uint8)##只有当数组类型为uint8时，opencv才会认为这是图片
        return img_gray

def min(x,y):
    if x>y:
        return y
    return x

def sameSize(img1, img2):#受奇偶性影响,上采样可能shape有1的差异
    row1, col1 = img1.shape
    row2, col2 = img2.shape
    rows,cols=min(row1,row2),min(col1,col2)
    dst1 = img1[:rows,:cols]
    dst2 = img2[:rows,:cols]  
    return dst1,dst2  

def gengp(im,layer):
    '''给定图像和层数,生成高斯'''
    g=im.copy()
    gp=[g]
    for i in range(layer-1):
        g=pyrDown(g)
        gp.append(g)
    return gp

def genlp(gp):
    layer=len(gp)
    lp=[gp[-1]]
    for i in range(layer-1,0,-1):
        up=pyrUp(gp[i])
        src1,src2=sameSize(gp[i-1],up)
        l=subtract(src1,src2)
        lp.append(l)
    return lp

def lp2gp(lp):
    '''由拉普拉斯金字塔得到高斯金字塔'''
    gp=[lp[0]]
    for i in range(len(lp)-1):
        up=pyrUp(gp[i])
        #print(pyrUp(lp[i]).shape,lp[i+1].shape)
        src1,src2=sameSize(up,lp[i+1])
        g=add(src1,src2)
        gp.append(g)
    return gp

if __name__=='__main__':
    im1=imread('1.jpg')
    im2=imread('2.png')
    gray1=cvtColor(im1,cv2.COLOR_BGR2GRAY)
    gray2=cvtColor(im2,cv2.COLOR_BGR2GRAY)
    layer=4 #高斯金字塔层数
    gp1=gengp(gray1,4)
    gp2=gengp(gray2,4)

    lp1=genlp(gp1)
    lp2=genlp(gp2)
    
    newgp1=lp2gp(lp1)
    newgp2=lp2gp(lp2)
    print('金字塔生成成功')
    
    for i in range(layer):
        imshow('gp',gp1[len(gp1)-1-i])
        imshow('lp',newgp1[i])
        imshow('new',lp1[i])
        waitKey(2000)
        cv2.destroyAllWindows()
    