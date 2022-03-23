import os
import numpy as np
import random
import matplotlib.pyplot as plt
from os import name
from pydantic import NoneStr
from scipy.spatial.distance import cdist
from sklearn.datasets import make_circles
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sympy import n_order
from sklearn.neighbors import kneighbors_graph
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import AgglomerativeClustering, DBSCAN, SpectralClustering, KMeans
from sklearn.mixture import GaussianMixture
from PIL import Image
import seaborn as sns

def select_centroids(X,k):
    """
    kmeans++ algorithm to select initial points:

    1. Pick first point randomly
    2. Pick next k-1 points by selecting points that maximize the minimum
       distance to all existing clusters. So for each point, compute distance
       to each cluster and find that minimum.  Among the min distances to a cluster
       for each point, find the max distance. The associated point is the new centroid.

    Return centroids as k x p array of points from X.
    """
    n = len(X)
    # Choose first point randomly
    centroids = X[np.random.choice(n, 1)]
    while len(centroids) < k:
        distance = np.zeros(n)
        for i, x in enumerate(X):
            # find minimum distance from each point to a cluster
            distance[i] = np.min(np.linalg.norm(centroids - x, axis=1)) 
        # the point with the maximum distance to its closest cluster is chosen and added to the centroids
        next_centroid = X[np.argmax(distance)].reshape(1,-1)
        centroids = np.append(centroids, next_centroid, axis=0)
    return centroids

def kmeans(X:np.ndarray, k:int, centroids=None, max_iter=30, tolerance=1e-2, seed=None):
    n = len(X)
    if seed:
        np.random.seed(seed)
    
    if centroids == 'kmeans++':
        centroids = select_centroids(X, k)
    else:
        # Intialize centroids by randomly choosing from X
        centroid_idx = np.random.choice(n, k, replace=False)
        centroids = X[centroid_idx]

    # Initialize cluster labels that will be assigned to each observation
    labels = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        old_centroids = centroids.copy()

        # Assign each observation to its closest cluster
        for i, x in enumerate(X):
            labels[i] = np.argmin(np.linalg.norm(centroids - x, axis=1))
        clusters = [X[labels == i] for i in range(k)]

        # Calculate mean of the clusters (centroids)
        centroids = np.array([np.mean(cluster_vectors, axis=0) for cluster_vectors in clusters])

        # If centroids do not change, then stop
        centroids_diff = np.linalg.norm(old_centroids - centroids)/k
#         print(f'{old_centroids}\n{centroids}\n{centroids_diff}\n\n')
        if centroids_diff < tolerance:
            break
    return centroids, labels    

def label_modification(y, labels):
    test_value = 1
    vals, counts = np.unique(y[labels == test_value], return_counts=True)
    index = np.argmax(counts)
    mode = vals[index]
    if mode != test_value: 
        labels = np.where(labels == 1, 0, 1)
    return labels

def leaf_samples(rf, X:np.ndarray):
    """
    Return a list of arrays where each array is the set of X sample indexes
    residing in a single leaf of some tree in rf forest. For example, if there
    are 4 leaves (in one or multiple trees), we might return:

        array([array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
               array([10, 11, 12, 13, 14, 15]), array([16, 17, 18, 19, 20]),
               array([21, 22, 23, 24, 25, 26, 27, 28, 29]))
    """
    n_trees = len(rf.estimators_)
    leaf_samples = []
    leaf_ids = rf.apply(X)  # which leaf does each X_i go to for sole tree?
    for t in range(n_trees):
        # Group by id and return sample indexes
        uniq_ids = np.unique(leaf_ids[:,t])
        sample_idxs_in_leaves = [np.where(leaf_ids[:, t] == id)[0] for id in uniq_ids]
        leaf_samples.extend(sample_idxs_in_leaves)
    return leaf_samples        

def get_init_centroids(X, k, max_iter, seed=1):
    if seed:
        np.random.seed(seed)
    k_centroids = []
    n = len(X)
    for i in range(max_iter):
        centroid_idx = np.random.choice(n, k, replace=False)
        centroids = X[centroid_idx]
        k_centroids.append(centroids)

    kpp_centroids = []
    for i in range(max_iter):
        centroids = select_centroids(X, k)
        kpp_centroids.append(centroids)
    return k_centroids, kpp_centroids

def plot_centroids(X, k_centroids, kpp_centroids):
    _, ax = plt.subplots(figsize=(15,5), nrows=1, ncols=2)
    ax = ax.flatten()
    n = len(k_centroids)
    for index in range(len(ax)):
        if index==0:
            ax[index].set_title('K-Means Initialization')
            centroids = k_centroids
        else:
            ax[index].set_title('K-Means++ Initialization') 
            centroids = kpp_centroids      
        ax[index].scatter(X[:,0], X[:,1])
        ax[index].set_xlabel('X1')
        ax[index].set_ylabel('X2')
        for i in range(n):
            ax[index].scatter(centroids[i][:, 0], centroids[i][:, 1], c=['red', 'orange'])
            ax[index].plot(centroids[i][:, 0], centroids[i][:, 1], c='yellow')
            
def get_accuracies(X, y, k, max_iter):
    k_accuracies = []
    for i in range(max_iter):
        centroids, labels = kmeans(X, k=k, centroids='kmeans', tolerance=.01)
        labels = label_modification(y, labels)
        C = confusion_matrix(y, labels)
        accuracy = np.diag(C).sum()/C.sum()
        k_accuracies.append(accuracy)
        
    kpp_accuracies = []
    for i in range(max_iter):
        centroids, labels = kmeans(X, k=k, centroids='kmeans++', tolerance=.01)
        labels = label_modification(y, labels)
        C = confusion_matrix(y, labels)
        accuracy = np.diag(C).sum()/C.sum()
        kpp_accuracies.append(accuracy)    
        
    return k_accuracies, kpp_accuracies     

def read_breast_cancer():
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target

    sc = StandardScaler()
    X = sc.fit_transform(X)
    return X, y

def cart2pol(X):
    x = X[:, 0]
    y = X[:, 1]
    rho = np.sqrt(x**2 + y**2).reshape(-1, 1)
    phi = np.arctan2(y, x).reshape(-1, 1)
    sc = StandardScaler()
    X = np.hstack((rho, phi))
    return sc.fit_transform(X)

def customSpectralClustering(X, k=2, seed=None):
    # Calculate similarity matrix using nearest-neighbors
    A = kneighbors_graph(X, n_neighbors=10).toarray()

    # Calculate the Laplacian from the similarity matrix
    D = np.diag(A.sum(axis=1))
    L = D-A

    # find the eigenvalues and eigenvectors
    vals, vecs = np.linalg.eig(L)
    vecs = vecs[:,np.argsort(vals)]
    vals = vals[np.argsort(vals)]

    # K-means on first k-1 vectors with nonzero eigenvalues
    selected_vecs = vecs[:, ~np.isclose(vals, 0)][:, 0:k-1]
    _, labels = kmeans(selected_vecs, k, centroids='kmeans++', seed=seed)
    return labels
    
def bootstrap_columns(X):
    n = len(X)
    X_ = X.copy()
    for j in range(X_.shape[1]):
        idx = np.random.randint(0, n, size=n)
        X_[:,j] = X_[idx, j]
    return X_

def walk_leaves_and_compute_similarity(leaves, n):
    similarity_matrix = np.zeros((n, n))
    for leaf in leaves:
        n_leaf = len(leaf)
        for i in range(n_leaf):
            for j in range(n_leaf):
                similarity_matrix[leaf[i], leaf[j]] += 1
    similarity_matrix /= len(leaves)
    return similarity_matrix    

def breiman_similarity_matrix(X):
    # Bootstrap all columns
    n = len(X)
    X_ = bootstrap_columns(X)
    stacked_y = np.array([0]*n + [1]*n)
    stacked_X = np.vstack((X, X_))

    rf = RandomForestClassifier(random_state=0)
    rf.fit(stacked_X, stacked_y)

    leaves = leaf_samples(rf, X)
    similarity_matrix = walk_leaves_and_compute_similarity(leaves, n)
    return similarity_matrix    

def read_image(image_path, grayscale=False):
    if grayscale:
        img = Image.open(image_path).convert('L')
    else:   
        img = Image.open(image_path)
    X = np.array(img)
    h = X.shape[0]
    w = X.shape[1]
    X = X.reshape(-1, 1)
    return img, X, h, w

def show_images(img, img_):
    fig, ax = plt.subplots(figsize=(16, 8), nrows=1, ncols=2)
    ax = ax.flatten()
    ax[0].imshow(img, cmap='gray')
    ax[1].imshow(img_, cmap='gray')
    for i in range(len(ax)):
        ax[i].axis('off')
    fig.tight_layout()
    plt.show()

def convert_bytes(size):
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return "%3.0f %s" % (size, x)
        size /= 1024.0
    return size    

def get_sizes(img, img_, grayscale=True):
    if grayscale:
        original='images/bw_original.png' 
        compressed='images/bw_compressed.png'
    else:
        original = 'images/color_original.png'
        compressed = 'images/color_compressed.png'
    img.save(original)
    img_.save(compressed)
    original_size = convert_bytes(os.stat(original).st_size)
    compressed_size = convert_bytes(os.stat(compressed).st_size)
    return original_size, compressed_size

def get_elbow_plot(X):
    wcss = []
    K = range(1, 10)

    for k in K:
        centroids, _ = kmeans(X, k, 'kmeans++')
        inertia = sum(np.min(cdist(X, centroids, 'sqeuclidean'), axis=1))
        wcss.append(inertia)

    plt.plot(K, wcss, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Inertia')
    plt.title('The Elbow Method using Inertia')
    plt.show()
    
def plot_scatter(X, labels, x_label, y_label, title):
    colors=np.array(['#4574B4','#A40227'])
    plt.scatter(X[:,0], X[:,1], c=colors[labels])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()

def get_labels(X, y, model):
    labels = model.fit_predict(X)
    labels = np.where(labels == -1, 1, labels)
    labels = label_modification(y, labels)
    return labels

def plot_confusion_matrix(y, labels, zero, one, title):
    C = confusion_matrix(y, labels)
    accuracy = np.diag(C).sum()/C.sum()
    group_counts = ["{0:0.0f}".format(value) for value in C.flatten()]
    group_percentages = ["{0:.0%}".format(value) for value in C.flatten()/np.sum(C)]
    labels = [f"{v1} ({v2})" for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.array(labels).reshape(2,2)

    ax = sns.heatmap(C, annot=labels, fmt='', cmap='Blues')
    ax.set_title(title)
    ax.set_ylabel('Actuals')
    ax.set_xlabel(f'Predicted\n\n\nAccuracy: {accuracy:.2f}')

    ax.xaxis.set_ticklabels([zero, one])
    ax.yaxis.set_ticklabels([zero, one])

    plt.show()