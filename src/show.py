import numpy as np
import matplotlib.pyplot as plt

file = open("image.txt", "r")
array = []
for line in file:
    array.append(float(line))


w = 731
h = 473
c = 3

image = np.array(array)
img = np.reshape(image, (w, h, c))

# print(img.shape)

plt.imshow(img)
plt.show()

# container = np.zeros((h, w, c))
# for channel in range(c):
    # for y in range(h):
        # for x in range(w):
            # container[y][x][channel] = image[x + y * w  + channel * w * h]

# plt.imshow(container)
# plt.show()
