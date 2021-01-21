from sklearn.cluster import MiniBatchKMeans
import numpy as np
from numpy import linalg as LA
import cv2
#view window with an input image and name
def viewImage(image, name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_FREERATIO)
    cv2.resizeWindow(name_of_window, 1280, 720)
    cv2.imshow(name_of_window, image)
    key=cv2.waitKey(1)
    #113 - 'q'
    # if(key==113):
    #     cv2.destroyAllWindows()
#input:image in BGR and number of clusters
#output:clusterizated image in BGR
def quanti(image,n_clust):
    (h, w) = image.shape[0:2]
    imageb = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).reshape((image.shape[0] * image.shape[1], 3))
    clt = MiniBatchKMeans(n_clusters = n_clust)
    labels = clt.fit_predict(imageb)
    quant = clt.cluster_centers_.astype("uint8")[labels]
    quant = quant.reshape((h, w, 3))
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    return quant
#this func ignore "bad" frames, such, for example, frames with tooltips, windows
#read from in and write files to out path, 1.png, 2.png...
def del_bad_frames(in_path,out_path,n_frames,teta):
    #720p - teta=35000
    #360*480 - teta=20000
    j=0
    name_number=1
    for i in range(n_frames-2):
        support_image=cv2.imread(in_path + str(j+1) + '.png')
        image=cv2.imread(in_path + str(i+2) + '.png')
        print(j+1, '->', i+2, ' : ', LA.norm(support_image.astype(int) - image.astype(int)))
        if(LA.norm(support_image.astype(int) - image.astype(int)) <teta):
            cv2.imwrite(out_path + str(name_number) + ".png", support_image)
            name_number+=1
            j=i+1
#optical flow from videofile, just give path
def OF_dense_video(path):
    cap = cv2.VideoCapture(path)
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    i=0
    while(1):
        i+=1
        ret, frame2 = cap.read()
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        viewImage(cv2.hconcat([frame2,bgr]),'Window1')
        prvs = next
#optical flow from many numerated frames (1.png,2.png,...)
def OF_dense_frames(path,n_frames):
    i=0
    image=cv2.imread(path+'/1.png')
    prvs=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(image)
    hsv[...,1] = 255
    for i in range(n_frames-2):
        i+=1
        tmp=(cv2.imread(path + str(i+1) + '.png'))
        next = cv2.cvtColor(tmp,cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        viewImage(cv2.hconcat([tmp,bgr]),'Window1')
        prvs = next
        print(i)
        key=cv2.waitKey(1)
        if(key==113):
            cv2.destroyAllWindows()
            break
def median_filter(path,n_frames):
    #not finished, just ignore that func
    #frames = [160,720,1280]
    im1=cv2.imread(path + '1.png')
    frames=[n_frames,np.shape(im1)[0],np.shape(im1)[1]]
    mas = np.zeros(frames, int)

    # img_gray_mode = cv2.imread('frames/1.png', cv2.IMREAD_GRAYSCALE)
    for i in range(frames[0]-1):
        tmp = (cv2.imread(path + str(i + 1) + '.png')).astype(int)
        mas[i] = tmp[..., 0] + (2 ** 8) * tmp[..., 1] + (2 ** 16) * tmp[..., 2]
        print(i)
        bg = np.quantile(mas,0.5, axis=0,overwrite_input = True)
        result = np.zeros((frames[1], frames[2], 3), np.uint8)
        # r
        result[..., 2] = bg / (2 ** 16)
        # g
        result[..., 1] = (bg / (2 ** 8)) % (2 ** 8)
        # b
        result[..., 0] = bg % (2 ** 8)
    return result
