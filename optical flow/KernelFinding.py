### Imports ###
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
import scipy
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from matplotlib import colors

### Import data ###
dirIn = "toy_problem/"
imgs = []
for i in range(1, 65):
    if i < 10:
        imgs.append(rgb2gray(plt.imread(dirIn + f"frame_0{i}.png")))
    elif not(i == 28 or i == 55):
        imgs.append(rgb2gray(plt.imread(dirIn + f"frame_{i}.png"))) # Do not consider the duplicate frames
imgs = np.array(imgs)
imgs.shape

### Low level ###
Vx = []
Vy = []
Vt = []
for t in range(imgs.shape[0]):
    Ix = []
    Iy = []
    It = []
    for y in range(1, imgs.shape[1]):
        Ux = []
        Uy = []
        Ut = []
        for x in range(1, imgs.shape[2]):
            Ux.append(imgs[t, y, x] - imgs[t, y, x - 1])
            Uy.append(imgs[t, y, x] - imgs[t, y - 1, x])
            if t > 0:
                Ut.append(imgs[t, y, x] - imgs[t - 1, y, x])
            else:
                Ut.append(0)
        Ix.append(Ux)
        Iy.append(Uy)
        It.append(Ut)
    Vx.append(Ix)
    Vy.append(Iy)
    Vt.append(It)
Vx = np.array(Vx)
Vy = np.array(Vy)
Vt = np.array(Vt)

### Sobel kernel ### 
I = imgs.astype(np.float64)

d = np.array([-1.0, 0.0, 1.0])
s = np.array([1.0, 2.0, 1.0])

Kt = np.einsum("t,w,h->twh", d, s, s) / 16
Kx = np.einsum("t,w,h->twh", s, s, d) / 16
Ky = np.einsum("t,w,h->twh", s, d, s) / 16

It = scipy.ndimage.convolve(I, Kt, mode="reflect")
Ix = scipy.ndimage.convolve(I, Kx, mode="reflect")
Iy = scipy.ndimage.convolve(I, Ky, mode="reflect")

### Gaussian kernel ###
sig = 1
Ux = scipy.ndimage.gaussian_filter(imgs, sigma=sig, order=(0, 0, 1), mode="reflect")
Uy = scipy.ndimage.gaussian_filter(imgs, sigma=sig, order=(0, 1, 0), mode="reflect")
Ut = scipy.ndimage.gaussian_filter(imgs, sigma=sig, order=(1, 0, 0), mode="reflect")

### Plotting kernels ###
fontSize = 25
imgNr = 10
plt.figure(figsize=(20, 20))
plt.subplot(3, 3, 1)
plt.imshow(Vx[imgNr], cmap="gray")
plt.yticks([])
plt.xticks([])
plt.title("Vx", fontsize=fontSize)
plt.ylabel("Low level", fontsize=fontSize)
plt.colorbar()
plt.subplot(3, 3, 2)
plt.imshow(Vy[imgNr], cmap="gray")
plt.yticks([])
plt.xticks([])
plt.colorbar()
plt.title("Vy", fontsize=fontSize)
plt.subplot(3, 3, 3)
plt.imshow(Vt[imgNr], cmap="gray")
plt.yticks([])
plt.xticks([])
plt.colorbar()
plt.title("Vt", fontsize=fontSize)

plt.subplot(3, 3, 4)
plt.imshow(-Ix[imgNr], cmap="gray")
plt.yticks([])
plt.xticks([])
plt.colorbar()
plt.ylabel("Sobel", fontsize=fontSize)
plt.subplot(3, 3, 5)
plt.imshow(-Iy[imgNr], cmap="gray")
plt.yticks([])
plt.xticks([])
plt.colorbar()
plt.subplot(3, 3, 6)
plt.imshow(-It[imgNr], cmap="gray")
plt.yticks([])
plt.xticks([])
plt.colorbar()

plt.subplot(3, 3, 7)
plt.imshow(Ux[imgNr], cmap="gray")
plt.yticks([])
plt.xticks([])
plt.colorbar()
plt.ylabel("Gaussian", fontsize=fontSize)
plt.subplot(3, 3, 8)
plt.imshow(Uy[imgNr], cmap="gray")
plt.yticks([])
plt.xticks([])
plt.colorbar()
plt.subplot(3, 3, 9)
plt.imshow(Ut[imgNr], cmap="gray")
plt.yticks([])
plt.xticks([])
plt.colorbar()
plt.tight_layout()
plt.show()