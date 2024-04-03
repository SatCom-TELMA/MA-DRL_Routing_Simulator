import os
path = './pictures'
files = os.listdir(path)


for index, file in enumerate(files):
    os.rename(os.path.join(path, file), os.path.join(path, ''.join([str(index).zfill(4), '.png'])))

# ffmpeg -framerate 20 -i %04d.png video.avi
# ffmpeg -framerate 20 -i %04d.png -qscale:v 0 video.avi