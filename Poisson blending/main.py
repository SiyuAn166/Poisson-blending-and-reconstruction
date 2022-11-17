import cv2
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from align_target import align_target
import numpy as np

def poisson_blend(source_image, target_image, target_mask):
    #source_image: image to be cloned
    #target_image: image to be cloned into
    #target_mask: mask of the target image
    blended_image = target_image.copy()
    ind = np.where(target_mask==1)
    n = len(ind[0])
    A = np.zeros((n,n))
    b = np.zeros(n)
    l = list(zip(ind[0], ind[1]))
    errors = []
    for channel in range(source_image.shape[2]):
        for c,(i,j) in enumerate(l):
            if target_mask[i,j] == 1:
                if target_mask[i-1,j]==0 or target_mask[i+1,j]==0 or target_mask[i,j-1]==0 or target_mask[i,j+1]==0:
                    A[c,c] = 1
                    b[c] = target_image[i,j,channel]
                else:
                    A[c,c] = 4
                    A[c,l.index((i-1,j))],A[c,l.index((i+1,j))],A[c,l.index((i,j-1))],A[c,l.index((i,j+1))] = -1,-1,-1,-1
                    b[c] = 4*source_image[i,j,channel]-source_image[i-1,j,channel]-source_image[i+1,j,channel]-source_image[i,j-1,channel]-source_image[i,j+1,channel]
        
        A = sparse.csr_matrix(A)
        
        img = spsolve(A, b)
        error = np.linalg.norm(A @ img - b)
        errors.append(error)
        # print(np.max(img), np.min(img))
        img[img<0] = 0
        img[img>255] = 255
        for c, (i,j) in enumerate(l):
            blended_image[i,j,channel] = img[c]
    print(f"LSE={sum(errors)}")
    return blended_image
    
    
if __name__ == '__main__':
    #read source and target images
    source_path = 'source2.jpg'
    target_path = 'target.jpg'
    source_image = cv2.imread(source_path)
    target_image = cv2.imread(target_path)

    #align target image
    im_source, mask = align_target(source_image, target_image)
    ##poisson blend
    blended_image = poisson_blend(im_source, target_image, mask)
    plt.figure()
    plt.imshow(blended_image)
    cv2.imwrite("b2.png",blended_image)
    
    