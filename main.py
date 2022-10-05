import cv2
from cv2 import imread,cvtColor,imshow,waitKey,pyrDown,pyrUp,subtract,add,bitwise_not,bitwise_and
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
def masked(src1,src2):
    '''图像金字塔与掩模金字塔相乘'''
    layer=len(src1)
    dst=[]
    for i in range(layer):
        r=bitwise_and(src1[i],src2[i])
        dst.append(r)
    return dst
def addimgae(src1,src2):
    '''图像金字塔相加'''
    layer=len(src1)
    dst=[]
    for i in range(layer):
        r=add(src1[i],src2[i])
        dst.append(r)
    return dst
if __name__=='__main__':
    layer=4 #高斯金字塔层数
    
    '''输入图像'''
    im1=imread('1.jpg')#掩模
    im2=imread('2.png')
    mask=imread('3.jpg')
   
    '''将RGB转化为灰度图像'''
    gray1=cvtColor(im1,cv2.COLOR_BGR2GRAY)
    gray2=cvtColor(im2,cv2.COLOR_BGR2GRAY)
    mask=cvtColor(mask,cv2.COLOR_BGR2GRAY)
    opmask=bitwise_not(mask)

    '''生成高斯金字塔'''
    gp1=gengp(gray1,4)
    gp2=gengp(gray2,4)
    gpmask=gengp(mask,4)
    gpopmask=gengp(opmask,4)

    '''生成图像的拉普拉斯金字塔'''
    lp1=genlp(gp1)
    lp2=genlp(gp2)
    
    '''拉普拉斯金字塔与掩模高斯金字塔相乘'''
    m1=masked(gpopmask[::-1],lp1)
    m2=masked(gpmask[::-1],lp2)
    
    '''图像的拉普拉斯金字塔相加后还原高斯金字塔'''
    combine=addimgae(m1,m2)
    newgp=lp2gp(combine)
    
    print('金字塔生成成功')
    
    for i in range(layer):
        # imshow('gp',gp1[len(gp1)-1-i])
        # imshow('lp',lp1[i])
        imshow('new',newgp[i])
        waitKey(2000)
        cv2.destroyAllWindows()
    