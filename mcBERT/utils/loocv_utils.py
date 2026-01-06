import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

def get_knn_prediction(train_emd, val_emd, labels_train, labels_val):
    if(len(labels_val)>1):
        print("warning getting KNN for more than a single LOOCV in val")
    sort_train = np.argsort(labels_train)
    labels_train_sorted=list(labels_train[sort_train])
    hues = [list(labels_train[sort_train]) + list(labels_val)]
    styles = ["Train"] * len(labels_train) + ["Val"] * len(labels_val)
    
    # Plot mean embeddings
    X = np.concatenate([train_emd[sort_train], val_emd])
    X_cosine_sim = cosine_similarity(X)
    cosine_dist = 1 - X_cosine_sim
    cosine_dist = np.abs(cosine_dist)
    val_dists = cosine_dist[-1]
    val_dists_noself = val_dists[:-1]
    k=3
    nearest_idxs=np.argpartition(val_dists_noself, k)[:k]
    neighbor_labels = [labels_train_sorted[idx] for idx in nearest_idxs]
    #default to the first nearest neighbor -- may want to mark as 3 way tie
    tie = True
    prediction = labels_train_sorted[nearest_idxs[0]]
    for label in neighbor_labels:
        num_in_list = 0
        for label2 in neighbor_labels:
            if(label==label2):
                num_in_list+=1
        if(num_in_list>=2):
            prediction = label
            tie = False
    prediction_matches = 1 if (prediction==labels_val[0]) else 0
    three_way_tie = 1 if tie else 0
    return prediction, prediction_matches, three_way_tie

def get_knn_prediction_platesep(epoch,train_emd, val_emd, labels_train, labels_val, lines_val):
    if(len(labels_val)>1):
        print("warning getting KNN for more than a single LOOCV in val")

    sort_train = np.argsort(labels_train)
    labels_train_sorted=list(labels_train[sort_train])
    hues = [list(labels_train[sort_train]) + list(labels_val)]
    styles = ["Train"] * len(labels_train) + ["Val"] * len(labels_val)
    num_val = len(labels_val)
    # Plot mean embeddings
    X = np.concatenate([train_emd[sort_train], val_emd])
    X_cosine_sim = cosine_similarity(X)
    cosine_dist = 1 - X_cosine_sim
    cosine_dist = np.abs(cosine_dist)
    final_csv_string= ""
    for index_in_slice, true_val_label in enumerate(labels_val):
        print(label)
        print("line",lines_val[index_in_slice])
        index_in_dists = index_in_slice-len(labels_val)
        val_dists = cosine_dist[index_in_dists]
        val_dists_no_self = val_dists[:-num_val]
        k=3
        nearest_idxs=np.argpartition(val_dists_no_self, k)[:k]
        neighbor_labels = [labels_train_sorted[idx] for idx in nearest_idxs]
        #default to the first nearest neighbor -- may want to mark as 3 way tie
        tie = True
        prediction = labels_train_sorted[nearest_idxs[0]]
        for label in neighbor_labels:
            num_in_list = 0
            for label2 in neighbor_labels:
                if(label==label2):
                    num_in_list+=1
            if(num_in_list>=2):
                prediction = label
                tie = False
        prediction_matches = 1 if (prediction==true_val_label) else 0
        three_way_tie = 1 if tie else 0
        csv_string =f"{epoch},{prediction},{prediction_matches},{tie},{lines_val[index_in_slice]}\n" 
        final_csv_string=final_csv_string+csv_string
    return final_csv_string