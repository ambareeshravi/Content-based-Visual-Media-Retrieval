import pickle, json
import pandas as pd
import numpy as np

from sklearn.metrics import average_precision_score, precision_recall_curve, plot_precision_recall_curve, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize

with open("test_results_2311.pkl", "rb") as f:
    results = pickle.load(f)
    
results_df = pd.DataFrame.from_dict(results).transpose()
labels_dict = dict([(idx, item) for idx,item in enumerate(['handclapping','jogging','running'])])
match_dict = {'running':0, 'jogging':0, 'handclapping':1}
y_act, y_pred = results_df['actual'].tolist(), results_df['prediction'].tolist()
y_act = [match_dict[i] for i in y_act]
y_pred = [match_dict[i] for i in y_pred]
print(classification_report(y_act, y_pred))
print(confusion_matrix(y_act, y_pred))

with open("LSTM_results.txt", "w") as f:
    f.write(metrics.classification_report(y_act, y_pred))

cm = confusion_matrix(y_act, y_pred)
recall = np.diag(cm) / np.sum(cm, axis = 1)
precision = np.diag(cm) / np.sum(cm, axis = 0)
print(np.mean(recall), np.mean(precision))