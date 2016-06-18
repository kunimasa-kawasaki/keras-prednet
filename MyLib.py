# -*- coding: utf-8 -*-
from Import import *

KEY_ESC = 27 # use opencv wait key

#---Matlab function---
def showImagePLT(im):
    im_list = np.asarray(im)
    plt.imshow(im_list)
    plt.show()
    
def showGraphData(data1,size,legend,title,save_name=0):
    plt.figure(figsize=size)
    plt.plot(range(len(data1)), data1)
    plt.legend(legend)
    plt.title(title)
    plt.plot()
    if not save_name == 0:
        plt.savefig(save_name) 
    else:
        plt.show()
        
def showGraph2Data(data1,data2,size,legend,title,save_name=0):
    plt.figure(figsize=size)
    plt.plot(range(len(data1)), data1)
    plt.plot(range(len(data2)), data2)
    plt.legend(legend)
    plt.title(title)
    plt.plot()
    if not save_name == 0:
        plt.savefig(save_name) 
    else:
        plt.show()

#---OS function---
def makeDir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def ListDir(path):
    file_list = os.listdir(path)
    file_list = sorted(file_list, key=str.lower)
    file_num = len(file_list)
    return file_list,file_num
