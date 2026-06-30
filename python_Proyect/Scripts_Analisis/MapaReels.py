import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import imageio_ffmpeg

lon = np.linspace(-120, -30, 500)
lat = np.linspace(-60, 30, 500)

X, Y = np.meshgrid(lon, lat)

fig, ax = plt.subplots(figsize=(9,16))

ax.set_facecolor("black")
ax.set_xlim(-120, -30)
ax.set_ylim(-60, 30)
ax.axis("off")

ax.text(-115,25,"Expansión de la especie",
        color="white",fontsize=28,weight="bold")

img = ax.imshow(
    np.zeros_like(X),
    extent=[-120,-30,-60,30],
    cmap="Greens",
    vmin=0,
    vmax=1,
    origin="lower"
)

def update(frame):

    center = 25 - frame*1.2
    spread = np.exp(-((Y-center)**2)/120)

    img.set_array(spread)

    return [img]

ani = FuncAnimation(fig, update, frames=80, interval=60)

writer = FFMpegWriter(
    fps=30,
    codec="libx264",
    bitrate=4000
)

ani.save(
    "expansion_latam.mp4",
    writer=writer,
    dpi=200
)

print("Video generado: expansion_latam.mp4")