# imports

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import cv2
import numpy as np
import sklearn
from glob import glob
from tqdm import tqdm
import pickle, json
import os
from random import shuffle
from matplotlib import pyplot as plt
from scipy.spatial import distance
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from datetime import datetime
import pandas as pd
from itertools import cycle
from time import time

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

from keras.applications.mobilenet import MobileNet
from keras.models import Model, Sequential, load_model as K_load_model
from keras.layers import LSTM, Dense, InputLayer


class DataHandler:
	def __init__(self):
		pass

	def read_image(self, image_path):
		'''
		Reads an image from a given path
		'''
	    return cv2.imread(image_path)

	def resize_image(self, im_array, resolution = (224,224)):
		'''
		Resizes the image
		'''
	    return cv2.resize(im_array, resolution)

	def load_images(self, files_list):
		'''
		Loads images to an array
		'''
	    return np.array([self.read_image(file) for file in files_list])

	def get_frames_video(self, video_path, resize_to = (224,224), output_frames = 16):
		'''
		Gets 'n' number of frames, each of resolution 'w' x 'h' and 3 channels (RGB) from a video

		Uses equidistant sampling of frames
		'''
	    cap = cv2.VideoCapture(video_path)
	    read_count = 1
	    frames_list = list()
	    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	    while cap.isOpened():
	        isRead, frame = cap.read()
	        if not isRead: break
	        if read_count % (int(frame_total/output_frames) -1) == 0:
	            frame = cv2.resize(frame, resize_to)
	            frames_list.append(frame)
	        read_count += 1
	        if len(frames_list) == 16: break
	    return np.array(frames_list)

	def get_category(self, idx):
		'''
		Returns categorial / one hot encoding
		'''
	    a = np.zeros(3)
	    a[idx] = 1
	    return a

	def prep_data(self, data_folder  = "DataSets/KTH/train/", recreate = False):
		'''
		Prepares data for training
		'''
	    if recreate:
	        print("Recreating Data . . .")
	        for folder in tqdm(os.listdir(data_folder)):
	            with open("LSTM_train_data/" + folder+".pkl", "wb") as f:
	                pickle.dump({'folder': np.array([self.get_frame_features(self.get_frames_video(os.path.join(os.path.join(data_folder, folder), file))) for file in os.listdir(os.path.join(data_folder, folder))])}, f)
	            print("Created " + "LSTM_train_data/" + folder+".pkl")
	        print("Done creating the necessary  files . . .")
	        
	        classes = list(set(['walking', 'boxing', 'handwaving']))
	        class_id = dict([(cls, self.get_category(idx)) for idx, cls in enumerate(classes)])

	        with open("LSTM_train_data/class.pkl", "wb") as f:
	            pickle.dump({"classes": classes, "class_id": class_id}, f)

	    trainX, trainY = list(), list()
	    with open("LSTM_train_data/class.pkl", "rb") as f:
	        class_info = pickle.load(f)
	    classes = class_info["classes"]
	    class_id = class_info["class_id"]
	    
	    for cls, label in class_id.items():
	        with open("LSTM_train_data/"+cls+".pkl", "rb") as f:
	            data_list = pickle.load(f)['folder']
	        for dl in data_list:
	            if dl.shape == (16,1024):
	                trainX.append(dl)
	                trainY.append(label)
	    trainX, trainY = np.array(trainX), np.array(trainY)
	    return classes, class_id, trainX, trainY

class Trainer(DataHandler):
	'''
	Trainer class for Conv2D  + LSTM model

	Initially trained as a classifier and then the final feature embedding is used for retrieval tasks
	'''
	def __init__(self):
		'''
		Initializes class variables
		'''
		DataHandler.__init__(self)
		self.version = str(datetime.now())[:16].replace(" ", "_").replace("-", "_").replace(":", "_")
		self.mobile_net = MobileNet()
		self.mobile_net_sub_model = self.get_submodel(self.mobile_net)

	def get_submodel(self, main_model, output_layer = "global_average_pooling2d_1"):
		'''
		Creates and returns a submodel till a given layer
		'''
		return Model(inputs=main_model.inputs, output=main_model.get_layer(output_layer).output)

	def get_frame_features(self, frames_list):
		'''
		Returns frame features from the mobile net submodel
		'''
	    # return mobile_net_submodel.predict(frames_list)
	    return np.array([np.squeeze(self.mobile_net_submodel.predict(np.expand_dims(frame, axis=0))) for frame in frames_list])

	def get_lstm_model(self, no_classes = 3, lstm_hidden_units = 200, image_features_size = 1024, step_size = 16, print_summary = False):
		'''
		Creates and returns the LSTM model for training
		'''
	    lstm_model = Sequential()
	    lstm_model.add(InputLayer(input_shape = (step_size, image_features_size)))
	    lstm_model.add(LSTM(lstm_hidden_units))
	    lstm_model.add(Dense(no_classes))
	    if print_summary: print(lstm_model.summary())
	    return lstm_model	

	def load_model(self, model_path = "model_run_1.h5"):
		'''
		Loads the pretrained model
		'''
	    lstm_model = self.get_lstm_model()
	    lstm_model.load_weights(model_path)
	    return lstm_model

	def get_model_path(self, model_name):
		'''
		Returns models full path
		'''
		model_save_path = os.path.join("trained_models", self.version)
		if not os.path.exists(model_save_path): os.mkdir(model_save_path)
		if model_name == None: model_name = "LSTM_" + self.version + "_e{epoch:02d}_va{val_accuracy:.2f}.hdf5"
		return os.path.join(model_save_path, model_name)

	def train(self, self, model, trainX, trainY, epochs = 20, batch_size = 2, validation_split = 0.1, model_name = None, no_classes = 3, lstm_hidden_units = 200, recreate_data = False):
		'''
		Trains an LSTM model with the given configuration
		'''
	    lstm_model = self.get_lstm_model(no_classes = no_classes, lstm_hidden_units = lstm_hidden_units)
	    classes, class_id, trainX, trainY = self.prep_data(data_folder  = "DataSets/KTH/train/", recreate = recreate_data)
	    
	    model_path = self.get_model_name(model_name)
		callbacks = [ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')]
		
		model.compile(loss='mean_squared_error', optimizer='adam')
	    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_split=validation_split, shuffle=True, verbose=2, callbacks = callbacks)

	    return lstm_model, classes, class_id
    
class Tester(DataHandler):
	'''
	Tester for Conv2D + LSTM model
	'''
	def __init__(self):
		pass

	def find_closest(self, repo_data_use, test_array, threshold = 0.95):
		'''
		Finds the close
		'''
	    close_list = list()
	    for idx, (feat, label) in enumerate(repo_data_use):
	        if cosine_similarity(feat, test_array) > threshold:
	            close_list.append([idx, label, cosine_similarity(feat, test_array)])
	    return close_list

	def check_accuracy(self, retrieved, index):
		'''
		Checks accuracy
		'''
	    correct = 0
	    for i in retrieved[:,1]:
	        if repo_data_test[index][1] in [1,2] and i in [1,2]:
	            correct +=1
	        else:
	            if i == repo_data_test[index][1]:
	                correct += 1
	    return correct / len(retrieved)

	def retrieve(self, index, repo_data_test):
		'''
		Retrieves the closest match and returns retrieval accuracy
		'''
	    retrieved = np.array(self.find_closest(repo_data_test, repo_data_test[index][0], 0.85))
	    accuracy = self.check_accuracy(retrieved, index)
	    return accuracy	

	def load_model(self, model_path = "model_run_1.h5"):
		'''
		Loads the pretrained model
		'''
	    lstm_model = self.get_lstm_model()
	    lstm_model.load_weights(model_path)
	    return lstm_model
                                      
	def model_predict(self, lstm_model, file_path, classes, get_class = False):
		'''
		Predicts a model
		'''
	    if get_class:
	        features = np.expand_dims(self.get_frame_features(self.get_frames_video(file_path)), axis=0)
	        st = time()
	        lstm_output = lstm_model.predict(features)
	        print(time() - st)
	        a =  classes[np.argmax(lstm_output)]
	        return a
	    else: return np.squeeze(lstm_model.predict(np.expand_dims(self.get_frame_features(self.get_frames_video(file_path)), axis=0))    )
	    
	def testfile_class(self, file_name):
	    return os.path.split(file_name)[-1].split("_")[1]

	def calc_precision_recall(self, n_classes, Y_test, y_score, version, plot = True):
	    # For each class
	    precision = dict()
	    recall = dict()
	    average_precision = dict()
	    
	    for i in range(n_classes):
	        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
	                                                            y_score[:, i])
	        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])
	    
	    # A "micro-average": quantifying score on all classes jointly
	    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(), y_score.ravel())
	    average_precision["micro"] = average_precision_score(Y_test, y_score, average="micro")
	    print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))
	    
	    if plot:
	        plt.figure()
	        plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
	                 where='post')
	        plt.fill_between(recall["micro"], precision["micro"], alpha=0.2, color='b')

	        plt.xlabel('Recall')
	        plt.ylabel('Precision')
	        plt.ylim([0.0, 1.05])
	        plt.xlim([0.0, 1.0])
	        plt.title('Average precision score, micro-averaged over all classes: AP={0:0.2f}'.format(average_precision["micro"]))
	        plt.savefig("trained_models/" + version + "/" + version +"_PR_curve.jpg")
	        
	    return average_precision["micro"], precision, recall, average_precision

	def test_model(self, version = "2019_11_23_22_18", test_classes_list = ['jogging', 'handclapping', 'running'], test_classes_dict = {'handclapping':1, 'jogging':2, 'running':2}, test_path = "DataSets/KTH/test/*/*", custom_name =""):
	    lstm_trained_model = self.load_model("trained_models/" + version + "/" + version +".hdf5")
	#     print(lstm_trained_model.summary())
	    lstm_model_sub = self.get_submodel(lstm_trained_model, 'lstm_1')
	    
	    test_results = dict()
	    for idx, i in enumerate(glob(test_path)):
	        try:
	            pred = self.model_predict(lstm_trained_model, i, test_classes_list, get_class = True)
	            features =  self.model_predict(lstm_model_sub, i, test_classes_list)
	            act = self.testfile_class(i)
	            test_results[idx] = {"file" : i, "actual" : act, "prediction" : pred, "features": features, "pred_id": test_classes_dict[pred], "act_id": test_classes_dict[act]}
	        except Exception as e:
	            print(e)
	            continue
	            
	    with open("trained_models/" + version + "/" + version + custom_name + ".pkl", "wb") as f:
	        pickle.dump(test_results, f)

	    results_df = pd.DataFrame.from_dict(test_results).transpose()
	    
	    Y_test = np.array([self.get_category(i) for i in np.array(results_df['act_id'])])
	    y_score = np.array([self.get_category(i) for i in np.array(results_df['pred_id'])])
	    
	    self.calc_precision_recall(3, Y_test, y_score, version, True)

	def get_closest(self, test_files_path, version, data_labels_dict, isResultsAvailable, results_save_file, sub_model_layer = 'lstm_1'):
	    lstm_trained_model = self.load_model("trained_models/" + version + "/" + version +".hdf5")
	    lstm_model_sub = self.get_submodel(lstm_trained_model, sub_model_layer)
	    
	    if not isResultsAvailable:
	        test_videos_features = dict()
	        for fl in tqdm(glob(test_files_path)):
	            try:
	                test_videos_features[fl] = self.model_predict(lstm_model_sub, fl, [], get_class = False)
	            except Exception as e:
	                print("ERROR", fl, e)
	                continue

	        with open(results_save_file, "wb") as f:
	            pickle.dump(test_videos_features, f)
	    else:
	        with open(results_save_file, "rb") as f:
	            test_videos_features = pickle.load(f)
	    
	    data_labels_dict = {'Covering' : 0, 'Uncovering' : 1, "Pushing": 2, "Moving" : 3, "Poking" : 4}

	    repo_data = list()
	    for fl, feat in test_videos_features.items():
	        label = -1
	        for k in data_labels_dict.keys():
	            if k in fl:
	                label = data_labels_dict[k]
	                break
	        repo_data.append(np.array([feat, label]))

	    repo_data = np.array(repo_data)
	    repo_data_use, repo_data_test = repo_data[:80], repo_data[80:]
	    print(self.find_closest(repo_data_test[1][0]), "\n\n", repo_data_test[1][1])
    
if __name__ == '__main__':

	# Testing the model
    test_model()
    
    accuracy_dict = {0:[], 1:[], 2:[]}
    for i in range(len(repo_data_test)):
        accuracy_dict[repo_data_test[i][1]].append(self.retrieve(repo_data_test, i))

    for k, v in accuracy_dict.items():
        print("Accuracy for ", k, " is ", sum(v)/len(v))
        
    CVIR_test_path = "DataSets/Final_Data_Repo/*"
    version = "2019_11_23_22_18"
    data_labels_dict = {'Covering' : 0, 'Uncovering' : 1, "Pushing": 2, "Moving" : 3, "Poking" : 4}
    sub_model_layer = 'lstm_1'

    get_closest(CVIR_test_path, version, data_labels_dict, isResultsAvailable = False, results_save_file = "final_data_repo.pkl", sub_model_layer = sub_model_layer)

    CVIR_test_path = "DataSets/KTH/test/*/*"
    version = "2019_11_23_22_18"
    repo_dict = {'handclapping' : 0, 'jogging' : 1, "running": 2}
    sub_model_layer = 'lstm_3'
    get_closest(CVIR_test_path, version, data_labels_dict, isResultsAvailable = False, results_save_file = "kth_data.pkl", sub_model_layer = sub_model_layer)