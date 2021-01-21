from functions import quanti, viewImage, del_bad_frames
import cv2
import numpy as np
import os
from numpy import linalg as LA
#image=cv2.imread('proj1/frames3/1.png')
#frames = [600,720,1280]

in_path='proj1/frames3/'
out_path="support1/"
n_frames=600
teta=20000

del_bad_frames(in_path,out_path,n_frames,teta)
