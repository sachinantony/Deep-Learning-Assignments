'''
Created on Feb 5, 2018

@author: sachi
'''
from sklearn.datasets import fetch_mldata
from sklearn.manifold import Isomap,LocallyLinearEmbedding,MDS,TSNE
import matplotlib.pyplot as plt

# Loading the Dataset
mnist = fetch_mldata('MNIST original')
data = mnist.data[::30]
target = mnist.target[::30]

#Isomap implementation
model = Isomap(n_components=2,n_neighbors=10)
proj = model.fit_transform(data)
plt.scatter(proj[:,0],proj[:,1],c=target,cmap=plt.get_cmap('jet',10))
plt.colorbar(ticks=range(10))
plt.clim(-0.5,9.5)
plt.show()

#Locally Linear Embedding implementation
model = LocallyLinearEmbedding(n_components=2,n_neighbors=10)
proj = model.fit_transform(data)
plt.scatter(proj[:,0],proj[:,1],c=target,cmap=plt.get_cmap('jet',10))
plt.colorbar(ticks=range(10))
plt.clim(-0.5,9.5)
plt.show()

#T-SNE implementation
model = TSNE(n_components=2)
proj = model.fit_transform(data)
plt.scatter(proj[:,0],proj[:,1],c=target,cmap=plt.get_cmap('jet',10))
plt.colorbar(ticks=range(10))
plt.clim(-0.5,9.5)
plt.show()

#MDS implementation
model = MDS(n_components=2)
proj = model.fit_transform(data)
plt.scatter(proj[:,0],proj[:,1],c=target,cmap=plt.get_cmap('jet',10))
plt.colorbar(ticks=range(10))
plt.clim(-0.5,9.5)
plt.show()

