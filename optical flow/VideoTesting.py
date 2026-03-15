### Imports ###
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
import scipy
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from matplotlib import colors

### Import data ###
# Video 1 folder and values
dirIn = "Video1/"
vidLength = 39 # Amount of frames
offset = 1 # First frame index

# Video 2 folder and values
# dirIn = "Video2/"
# vidLength = 40 # Amount of frames
# offset = 60 # First frame index

imgs = []
for i in range(offset, vidLength+offset):
    if i < 10:
        imgs.append(rgb2gray(plt.imread(dirIn + f"000{i}.jpg")))
    else:
        imgs.append(rgb2gray(plt.imread(dirIn + f"00{i}.jpg")))
imgs = np.array(imgs)
imgs.shape

### Filtering ###
I = imgs.astype(np.float64)

d = np.array([-1.0, 0.0, 1.0])
s = np.array([1.0, 2.0, 1.0])

Kt = np.einsum("t,w,h->twh", d, s, s)
Kx = np.einsum("t,w,h->twh", s, s, d)
Ky = np.einsum("t,w,h->twh", s, d, s)

Vt = scipy.ndimage.convolve(I, Kt, mode="reflect")
Vx = scipy.ndimage.convolve(I, Kx, mode="reflect")
Vy = scipy.ndimage.convolve(I, Ky, mode="reflect")

### Lucas-Kanade ###
N = 15

allXs = []
allYs = []
allVs = []
for imgNr in range(imgs.shape[0]):
    Vs = []
    Xs = []
    Ys = []
    for x in range(0, imgs.shape[2]-N, N):
        for y in range(0, imgs.shape[1]-N, N):
            p = (imgNr, y, x)
            A = []
            b = []
            for i in range(-N//2, N//2 + 1):
                for j in range(-N//2, N//2 + 1):
                    A.append([Vx[p[0], p[1] + i, p[2] + j], Vy[p[0], p[1] + i, p[2] + j]])
                    b.append([-Vt[p[0], p[1] + i, p[2] + j]])
            A = np.array(A)
            b = np.array(b)
            v = np.linalg.lstsq(A, b, rcond=None)[0].flatten()
            dist = np.sqrt(v[0]**2 + v[1]**2)
            Xs.append(x)
            Ys.append(y)
            Vs.append(v)
    Xs = np.array(Xs)
    Ys = np.array(Ys)
    Vs = np.array(Vs)
    allXs.append(Xs)
    allYs.append(Ys)
    allVs.append(Vs)
allXs = np.array(allXs)
allYs = np.array(allYs)
allVs = np.array(allVs)


### Plotting ###
all_speed = np.linalg.norm(allVs, axis=2)
d_min = 20
d_max = 60
norm = colors.Normalize(vmin=d_min, vmax=d_max)

vmax = np.linalg.norm(allVs, axis=2).max()
norm = colors.Normalize(vmin=d_min, vmax=d_max)

fig, ax = plt.subplots(figsize=(8, 8))

def update(frame_idx):
    ax.clear()
    ax.imshow(imgs[frame_idx], cmap="gray")

    Xs = allXs[frame_idx]
    Ys = allYs[frame_idx]
    Vs = allVs[frame_idx]
    speed = np.linalg.norm(Vs, axis=1)

    valid = (speed >= d_min) & (speed <= d_max)

    q = ax.quiver(
        Xs[valid], Ys[valid], Vs[valid, 0], Vs[valid, 1], speed[valid],
        angles="xy", scale_units="xy", scale=0.33,
        cmap="jet", norm=norm, width=0.004, headwidth=3, headlength=5
    )
    ax.set_title(f"Frame {frame_idx+1}")
    ax.axis("off")
    return q,

anim = FuncAnimation(fig, update, frames=imgs.shape[0], interval=250, blit=False)
plt.show()
plt.close(fig)
HTML(anim.to_jshtml())