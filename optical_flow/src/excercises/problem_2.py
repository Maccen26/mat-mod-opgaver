import numpy as np 
from warm_up import load_jepgs 
import matplotlib.pyplot as plt



def low_level_gradient_vt(images):
    # Compute the low-level gradient for each image in the sequence
    gradients = []
    for i in range(1, len(images)):
        gradients.append((images[i] - images[i-1]))  #We compute gtadient over time
    return gradients

def low_level_gradient_vx(images):
    # Compute the low-level gradient for each image in the sequence
    gradients = []
    for i in range(len(images)):
        gradients.append((images[i][:, 1:] - images[i][:, :-1]))  #We compute gtadient over x-axis
    return gradients

def low_level_gradient_vy(images):
    # Compute the low-level gradient for each image in the sequence
    gradients = []
    for i in range(len(images)):
        gradients.append((images[i][1:, :] - images[i][:-1, :]))  #We compute gtadient over y-axis
    return gradients

if __name__ == "__main__":
    path = "optical_flow/data"
    images = load_jepgs(path)
    print(f"Loaded {len(images)} images from {path}")
    gradients_vt = low_level_gradient_vt(images)
    gradients_vx = low_level_gradient_vx(images) 
    gradients_vy = low_level_gradient_vy(images)
    print(f"Computed low-level gradients for {len(gradients_vt)} image pairs.")
    # Visualize the first few gradients
    for i in range(2):
        plt.imshow(gradients_vt[i], cmap="gray")
        plt.title(f"Low-Level Gradient Vt between Frame {i} and Frame {i+1}")
        plt.axis("off")
        plt.show() 
    for i in range(2):
        plt.imshow(gradients_vx[i], cmap="gray")
        plt.title(f"Low-Level Gradient Vx between Frame {i} and Frame {i+1}")
        plt.axis("off")
        plt.show() 
    for i in range(2):
        plt.imshow(gradients_vy[i], cmap="gray")
        plt.title(f"Low-Level Gradient Vy between Frame {i} and Frame {i+1}")
        plt.axis("off")
        plt.show() 