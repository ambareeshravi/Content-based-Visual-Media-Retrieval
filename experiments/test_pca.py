from sklearn.decomposition import PCA
import pickle
import pandas as pd
import numpy as np
from random import shuffle
from matplotlib import pyplot as plt

with open("test_results_2311.pkl", "rb") as f:
    test_results = pickle.load(f)
    
results_df = pd.DataFrame.from_dict(test_results).transpose()
results_copy = results_df[['actual', 'features']]
results_dict = results_copy.to_dict('records')
results_list = np.array([np.array([item['actual'], item['features']]) for item in results_dict])
labels, features = results_list[:,0], results_list[:,1]
features = np.array([np.array(f) for f in features])
main, test = results_list[:50], results_list[50:]

main = np.array([np.array(list(i)) for i in main])
test = np.array([np.array(list(i)) for i in test])

main_features = np.array([np.array(i) for i in main[:,1]])
test_features = np.array([np.array(i) for i in test[:,1]])

pca = PCA(n_components=100)
pca_features = pca.fit_transform(features)

random_indices = list(range(299))
shuffle(random_indices)
random_indices = random_indices[:50]

label_dict = dict([(l, idx) for idx, l in enumerate(set(labels))])

def get_retrieval_accuracy(main_label, retrieval_labels):
    return (retrieval_labels.count(main_label) - 1) / len(retrieval_labels)

def cosine_similarity(a,b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def find_closest(test_feat, main_array, threshold = 0.95):
    close_list = list()
    for idx, feat in enumerate(main_array.tolist()):
        score = cosine_similarity(feat, test_feat)
        if score > threshold:
            close_list.append(np.array([idx, score]))
    return np.array(close_list)

def get_average(l):
    return sum(l)/len(l)

def get_pca_accuracy(n_comp):
    pca = PCA(n_components=n_comp)
    pca_features = pca.fit_transform(features)
    lstm_accuracy = list()
    pca_accuracy = list()

    for idx in random_indices:
        lstm_results = find_closest(results_list[:,1][idx], np.array([m for m in results_list[:,1]]))
        lstm_indices, lstm_scores = lstm_results[:,0], lstm_results[:,1]

        pca_results = find_closest(pca_features[idx], pca_features)
        pca_indices, pca_scores = pca_results[:,0], pca_results[:,1]

        lstm_accuracy.append(get_retrieval_accuracy(label_dict[results_list[:,0][idx]], [label_dict[results_list[:,0][i]] for i in np.asarray(lstm_indices, dtype=np.int16)]))
        pca_accuracy.append(get_retrieval_accuracy(label_dict[results_list[:,0][idx]], [label_dict[results_list[:,0][i]] for i in np.asarray(pca_indices, dtype=np.int16)]))
    
    return get_average(lstm_accuracy), get_average(pca_accuracy)

get_pca_accuracy(150)

x = list(range(1,200,10))
y = [get_pca_accuracy(i)[-1] for i in x]

yj = [i*100 for i in y]
plt.plot(x,yj)
plt.xlabel("Number of components in PCA")
plt.ylabel("Retrieval accuracy (%)")
plt.savefig("pca_results.jpeg")