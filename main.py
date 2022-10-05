import cv2
import numpy as np

class Cv():
    def convertGray(self,img_rgb):
        img_gray = img_rgb[:,:,0] * 0.11 + img_rgb[:,:,1] * 0.59 + img_rgb[:,:,2] * 0.3
        img_gray = img_gray.astype(np.uint8)##只有当数组类型为uint8时，opencv才会认为这是图片
        return img_gray


if __name__=='__main__':
    im1=cv2.imread('1.jpg')
    im2=cv2.imread('2.jpg')