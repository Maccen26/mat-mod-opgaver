from skimage import io 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def build_paths(path):
    paths = []
    for i in range(1, 10):
        paths.append(f"{path}/frame_0{i}.png") 
    for i in range(10, 65):
        paths.append(f"{path}/frame_{i}.png")
    return paths

def load_jepgs(path): 
    image_paths = build_paths(path) 
    images = []
    for image_path in image_paths:
        images.append(io.imread(image_path, as_gray=True)) #grey scale 
    return images


def play_video(images, fps=12):
    fig, ax = plt.subplots()
    frame_artist = ax.imshow(images[0], cmap="gray")
    ax.set_axis_off()

    def update(frame_index):
        frame_artist.set_array(images[frame_index])
        return (frame_artist,)

    animation = FuncAnimation(
        fig,
        update,
        frames=len(images),
        interval=1000 / fps,
        blit=False,
        repeat=True,
    )
    fig.animation = animation
    plt.show()



if __name__ == "__main__":
    path = "optical_flow/data"
    images = load_jepgs(path)
    print(f"Loaded {len(images)} images from {path}")
    play_video(images)


