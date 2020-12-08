import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import greycomatrix, greycoprops
scale_percent=50

Area=[]
Peri=[]
def Average(lst): 
    return sum(lst) / len(lst) 

        
while(1):
    print("Starting")
    name=input("enter image name");
    x= cv2.imread(name)
    #x=cv2.imread('image018.png')
    w=int(x.shape[1]*scale_percent/100)
    h=int(x.shape[0]*scale_percent/100)
    dim=(w,h)
    img=cv2.resize(x,dim)
    
    gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    # Set up the detector with default parameters.
    #convert into binary
    greenchannel=img[:,:,1]
    ret,binary = cv2.threshold(greenchannel,160,255,cv2.THRESH_BINARY)# 160 - threshold, 255 - value to assign, THRESH_BINARY_INV - Inverse binary

    #averaging filter
    kernel = np.ones((5,5),np.float32)/9
    dst = cv2.filter2D(binary,-1,kernel)# -1 : depth of the destination image


    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

    #erosion
    erosion = cv2.erode(dst,kernel2,iterations = 1)

    #dilation 
    dilation = cv2.dilate(erosion,kernel2,iterations = 1)

    #edge detection
    edges = cv2.Canny(dilation,100,200)

##    ### Size detection
    contours,hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
##    print("No. of rice grains=",len(contours))
##    total_ar=0
    ##### Feature Extraction
    # compute some GLCM properties each patch
    xs =0
    
    glcm = greycomatrix(gray, [5], [0], 256, symmetric=True, normed=True)
    xs=greycoprops(glcm, 'dissimilarity')[0, 0]
    
    #print(xs)
    if(xs>3.1):
        print('Fundus is abnormol')
    else:
        print('Fundus is normal')
    for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)
          
           

    cv2.imshow('disp',img)
    #plot the images
    imgs_row=2
    imgs_col=3
    plt.subplot(imgs_row,imgs_col,1),plt.imshow(img,'gray')
    plt.title("Original image")

    plt.subplot(imgs_row,imgs_col,2),plt.imshow(binary,'gray')
    plt.title("Binary image")

    plt.subplot(imgs_row,imgs_col,3),plt.imshow(dst,'gray')
    plt.title("Filtered image")

    plt.subplot(imgs_row,imgs_col,4),plt.imshow(erosion,'gray')
    plt.title("Eroded image")

    plt.subplot(imgs_row,imgs_col,5),plt.imshow(dilation,'gray')
    plt.title("Dialated image")

    plt.subplot(imgs_row,imgs_col,6),plt.imshow(edges,'gray')
    plt.title("Edge detect")

    plt.show()

