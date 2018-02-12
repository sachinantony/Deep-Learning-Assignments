'''
Created on Jan 16, 2018
@author: Sachin Antony
'''
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
import numpy as np
from numpy import size

# Loading the Dataset
mnist = fetch_mldata('MNIST original')
X, y = mnist["data"], mnist["target"]

#Splitting into training and test sets
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000],y[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

#Using PCA for Dimensionality reduction and reconstructing the image
pca = PCA(n_components = 154)
X_reduced = pca.fit_transform(X_train)
X_recovered = pca.inverse_transform(X_reduced)

#Displaying the original image
some_digit = X_train[333]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary,interpolation="nearest")
plt.axis("off")
plt.show()

#Displaying the reconstructed image
some_digit = X_recovered[333]
some_digit_image2 = some_digit.reshape(28, 28)
plt.imshow(some_digit_image2, cmap = matplotlib.cm.binary,interpolation="nearest")
plt.axis("off")
plt.show()

#Displaying the original image
some_digit = X_train[666]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary,interpolation="nearest")
plt.axis("off")
plt.show()

#Displaying the reconstructed image
some_digit = X_recovered[666]
some_digit_image2 = some_digit.reshape(28, 28)
plt.imshow(some_digit_image2, cmap = matplotlib.cm.binary,interpolation="nearest")
plt.axis("off")
plt.show()

#Displaying the original image
some_digit = X_train[3333]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary,interpolation="nearest")
plt.axis("off")
plt.show()

#Displaying the reconstructed image
some_digit = X_recovered[3333]
some_digit_image2 = some_digit.reshape(28, 28)
plt.imshow(some_digit_image2, cmap = matplotlib.cm.binary,interpolation="nearest")
plt.axis("off")
plt.show()

some_digit = pca.components_[1]
some_digit_image2 = some_digit.reshape(28, 28)
plt.imshow(some_digit_image2, cmap = matplotlib.cm.binary,interpolation="nearest")
plt.axis("off")
plt.show()

some_digit = pca.components_[2]
some_digit_image2 = some_digit.reshape(28, 28)
plt.imshow(some_digit_image2, cmap = matplotlib.cm.binary,interpolation="nearest")
plt.axis("off")
plt.show()

some_digit = pca.components_[3]
some_digit_image2 = some_digit.reshape(28, 28)
plt.imshow(some_digit_image2, cmap = matplotlib.cm.binary,interpolation="nearest")
plt.axis("off")
plt.show()

code = pca.transform(X_train)
code = [np.round(a*4,0)/4.0 for a in code ]

X_recovered = pca.inverse_transform(code);

#Displaying the reconstructed image
some_digit = X_recovered[3333]
some_digit_image2 = some_digit.reshape(28, 28)
plt.imshow(some_digit_image2, cmap = matplotlib.cm.binary,interpolation="nearest")
plt.axis("off")
plt.show()

#Displaying the reconstructed image
some_digit = X_recovered[333]
some_digit_image2 = some_digit.reshape(28, 28)
plt.imshow(some_digit_image2, cmap = matplotlib.cm.binary,interpolation="nearest")
plt.axis("off")
plt.show()