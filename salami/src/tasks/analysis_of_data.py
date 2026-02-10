
import numpy as np
from scipy import io
import matplotlib.pyplot as plt
from db.data_paths import DataPaths
from utils import showHistograms

def plot_multispectral_image(day: int):
    dp = DataPaths() 

    anno_path, color_path, multispectral_path = dp.get_data_path_by_day(day) 
    multiIm = io.loadmat(multispectral_path)["immulti"]
    plt.imshow(multiIm[:,:,0], cmap="gray") 
    plt.title(f"Multispectral image - day {day}")
    plt.show() 




def plot_band_histograms_for_annotations(day: int, band: int = 0):
    dp = DataPaths()

    anno_path, _, multispectral_path = dp.get_data_path_by_day(day)
    multiIm = io.loadmat(multispectral_path)["immulti"]
    annoIm = plt.imread(anno_path)

    if annoIm.ndim == 2:
        annoIm = annoIm[:, :, None]

    pixId = (annoIm > 0).astype(int)
    showHistograms(multiIm, pixId, band, 1)



if __name__ == "__main__":
    #plot_multispectral_image(1)
    plot_band_histograms_for_annotations(1, band=18)

