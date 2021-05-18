import matplotlib.pyplot as plt
from skimage import io
from sklearn.cluster import MiniBatchKMeans

tiger = io.imread('tiger.jpg')
ax = plt.axes(xticks=[], yticks=[])
ax.imshow(tiger)
print(tiger.shape)

data = tiger / 255.0  # use 0...1 scale
data = data.reshape(600 * 1066, 3)

kmeans = MiniBatchKMeans(16)
kmeans.fit(data)
new_colors = kmeans.cluster_centers_[kmeans.predict(data)]

china_recolored = new_colors.reshape(tiger.shape)

fig, ax = plt.subplots(1, 2, figsize=(16, 6),
                       subplot_kw=dict(xticks=[], yticks=[]))
fig.subplots_adjust(wspace=0.05)
ax[0].imshow(tiger)
ax[0].set_title('Original Image', size=16)
ax[1].imshow(china_recolored)
ax[1].set_title('16-color Image', size=16)
plt.show()
