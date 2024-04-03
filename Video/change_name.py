import os

path = './pictures/'
files = os.listdir(path)

# Function to extract key information from filename
def extract_info(filename):
    if filename[0].isdigit():
        parts = filename.split('_')
        # print(parts)
        source_destination = parts[0] + '_' + parts[1]  # Source and destination
        position = int(parts[2])  # Position in the route
        return source_destination, position, filename

# Extract info from each file and sort
file_info = [extract_info(file) for file in files]
sorted_files = sorted(file_info, key=lambda x: (x[0], x[1]))

# Rename files in sorted order
for index, (_, _, filename) in enumerate(sorted_files):
    new_name = f'{index}.png'
    os.rename(os.path.join(path, filename), os.path.join(path, new_name))

print("Files have been renamed in order.")


# import os
# path = './'#'./pictures'
# files = os.listdir(path)

# for index, file in enumerate(files):
#     # os.rename(os.path.join(path, file), os.path.join(path, f'{index}.png'))
#     os.rename(os.path.join(path, file), os.path.join(path, ''.join([str(index).zfill(4), '.png'])))


# ffmpeg -framerate 20 -i %04d.png video.avi
# ffmpeg -framerate 20 -i %04d.png -qscale:v 0 video.avi
# ffmpeg -framerate 10 -i ./%d.png -c:v libx264 -r 30 -pix_fmt yuv420p ./video.mp4
