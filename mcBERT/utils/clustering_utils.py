import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP


def get_plot_as_img(train_emd, val_emd, labels_train, labels_val):
    sort_train = np.argsort(labels_train)
    hues = [list(labels_train[sort_train]) + list(labels_val)]
    styles = ["Train"] * len(labels_train) + ["Val"] * len(labels_val)

    # Plot mean embeddings
    X = np.concatenate([train_emd[sort_train], val_emd])
    X_cosine_sim = cosine_similarity(X)
    cosine_dist = 1 - X_cosine_sim
    cosine_dist = np.abs(cosine_dist)
    # X_embedded = TSNE(metric="precomputed", init="random").fit_transform(cosine_dist)
    X_embedded = UMAP(
        n_components=2, init="random", random_state=0, metric="cosine", n_jobs=1
    ).fit_transform(cosine_dist)
    fig = plt.figure(figsize=(10, 10))
    #print("hues", hues)
    sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=hues[0], style=styles)
    #prediction, prediction_matches, three_way_tie = get_KNN_umap(X_embedded[:-1, 0],X_embedded[:-1, 1],)
    return fig

def get_plot_as_img_largerpts(train_emd, val_emd, labels_train, labels_val):
    sort_train = np.argsort(labels_train)
    hues = [list(labels_train[sort_train]) + list(labels_val)]
    styles = ["Train"] * len(labels_train) + ["Val"] * len(labels_val)

    # Plot mean embeddings
    X = np.concatenate([train_emd[sort_train], val_emd])
    X_cosine_sim = cosine_similarity(X)
    cosine_dist = 1 - X_cosine_sim
    cosine_dist = np.abs(cosine_dist)
    # X_embedded = TSNE(metric="precomputed", init="random").fit_transform(cosine_dist)
    X_embedded = UMAP(
        n_components=2, init="random", random_state=0, metric="cosine", n_jobs=1
    ).fit_transform(cosine_dist)
    fig = plt.figure(figsize=(10, 10))
    #print("hues", hues)
    sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=hues[0], style=styles,s=100)
    #prediction, prediction_matches, three_way_tie = get_KNN_umap(X_embedded[:-1, 0],X_embedded[:-1, 1],)
    return fig

def get_KNN_umap(other_X, other_Y, other_labels, pt_X, pt_Y, pt_label):
    distances = []
    other_points = list(zip(other_X,other_Y))
    reference_point = (pt_X,pt_Y)
    for i, point in enumerate(other_points):
        distance = np.linalg.norm(np.array(reference_point) - np.array(point))
        distances.append((distance, point))
    distances.sort(key=lambda x: x[0])
    nearest_three = [item[1] for item in distances[:3]]
    nearest_three_labels = other_labels[nearest_three]
    for label in nearest_three_labels:
        num_in_list = 0
        for label2 in nearest_three_labels:
            if(label==label2):
                num_in_list+=1
        if(num_in_list>=2):
            prediction = label
            tie = False
    prediction_matches = 1 if (prediction==pt_label) else 0
    three_way_tie = 1 if tie else 0
    return prediction, prediction_matches, three_way_tie


def apply_UMAP(X):
    # use UMAP to project the data onto a 2D plane
    umap_2d = UMAP(n_components=2, init="random", random_state=0, metric="cosine")
    X_embedded = umap_2d.fit_transform(X)

    return X_embedded
