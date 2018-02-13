from sklearn.datasets import fetch_mldata
import matplotlib.cm
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import os

# Loading the Dataset
mnist = fetch_mldata('MNIST original')
X, y = mnist["data"], mnist["target"]

#Training the dataset using randodm forest
rnd_clf = RandomForestClassifier(random_state=42)
rnd_clf.fit(mnist["data"], mnist["target"])

#Plotting the each pixel's importance
def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = matplotlib.cm.hot,
               interpolation="nearest")
    plt.axis("off")
plot_digit(rnd_clf.feature_importances_)
cbar = plt.colorbar(ticks=[rnd_clf.feature_importances_.min(), rnd_clf.feature_importances_.max()])
cbar.ax.set_yticklabels(['Not important', 'Very important'])
plt.show()