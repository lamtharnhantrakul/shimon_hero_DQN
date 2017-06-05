import matplotlib.pyplot as plt
import matplotlib.image as mpimg
image = mpimg.imread("../UnitTests/snail.jpg")
print (image.shape)
plt.imshow(image)
plt.show()