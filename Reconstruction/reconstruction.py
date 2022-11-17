import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import sparse
if __name__ == '__main__':
    #read source image
    img_path = "target.jpg"
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
    ##implement reconstruction
    h, w = img.shape
    K = h * w
    A = np.zeros((K, K))
    b = np.zeros(K)
    
    #four corners
    A[0,0], A[w-1, w-1] = 1, 1 # tl, tr
    A[w*(h-1),w*(h-1)], A[h*w-1, h*w-1] = 1, 1 #bl, br
    b[0], b[w-1] = 2, 2
    b[w*(h-1)], b[h*w-1] = 1, 1
    
    #1
    for i in range(1, w-1):
        pmap = np.zeros((h,w))
        pmap[0,i] = 2
        pmap[0,i-1] = -1
        pmap[0,i+1] = -1
        A[i] = pmap.flatten()
        b[i] = 2*int(img[0,i]) - int(img[0,i-1]) - int(img[0,i+1])
    #2
    for i in range(w, w*(h-1) ,w):
        pmap = np.zeros((h,w))
        pmap[i//w,0] = 2
        pmap[(i//w)-1, 0] = -1
        pmap[(i//w)+1, 0] = -1
        A[i] = pmap.flatten()
        b[i] = 2*int(img[i//w,0]) - int(img[(i//w)-1, 0]) - int(img[(i//w)+1, 0])
        
    #3
    for i in range(2*w-1, (w-1)*h, w):
        pmap = np.zeros((h,w))
        pmap[(i-w+1)//w, w-1] = 2
        pmap[((i-w+1)//w)-1, w-1] = -1
        pmap[((i-w+1)//w)+1, w-1] = -1
        A[i] = pmap.flatten()
        b[i] = 2*int(img[(i-w+1)//w, w-1]) - int(img[((i-w+1)//w)-1, w-1]) - int(img[((i-w+1)//w)+1, w-1])
        
    #4
    for i in range(1, w-1):
        pmap = np.zeros((h,w))
        pmap[h-1, i] = 2
        pmap[h-1, i-1] = -1
        pmap[h-1, i+1] = -1
        A[w*(h-1) + i] = pmap.flatten()
        b[w*(h-1) + i] = 2*int(img[h-1, i]) - int(img[h-1, i-1]) - int(img[h-1,i+1])
    
    #m
    for i in range(1, h-1):
        for j in range(1, w-1):
            pmap = np.zeros((h,w))
            pmap[i,j] = 4
            pmap[i+1,j],pmap[i-1,j] = -1, -1
            pmap[i,j+1],pmap[i,j-1] = -1, -1
            A[i*w + j] = pmap.flatten()
            b[i*w + j] = 4*int(img[i,j]) - int(img[i+1,j]) - int(img[i-1,j]) - int(img[i,j+1]) - int(img[i,j-1])
    
    
    
    
    A = sparse.csr_matrix(A)
    img_hat = sparse.linalg.spsolve(A, b)
    img_hat = img_hat.reshape((h,w))
    plt.axis('off')
    plt.imshow(img_hat, cmap = 'gray')
    plt.show()
    
    cv2.imwrite('output3.png',img_hat)
    
    #lse
    error = np.linalg.norm(A @ img_hat.flatten() - b)
    print("LSE = ", error)
    
    
    