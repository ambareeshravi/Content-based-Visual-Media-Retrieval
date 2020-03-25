# imports
import warnings
warnings.filterwarnings('ignore')

import os, cv2, shutil, json
import numpy as np, pandas as pd, pickle as pkl

from glob import glob
from time import time
from datetime import datetime
from tqdm import tqdm

from sklearn.metrics import precision_recall_curve, average_precision_score, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

from keras.applications.mobilenet import MobileNet
from keras.models import Model, load_model as K_load_model
from keras.layers import LSTM, Dense, InputLayer

class DataHandler:
	'''
	Handles all operations with respect to data
	'''
	def __init__(self, videos_path, test_size = 0.05):
		'''
		Initalizes the class variables for data handling
		'''
		self.n_frames = 16
		self.operating_resolution = (224, 224)
		self.test_split = test_size

		self.videos_path = videos_path
		self.image_feature_extractor = self.get_mobilenet_feature_extractor()
	
	def get_mobilenet_feature_extractor(self):
		'''
		Returns the mobilenet feature extractor
		'''
		mobilenet = MobileNet()
		return Model(inputs=mobilenet.inputs, output=mobilenet.get_layer("global_average_pooling2d_1").output)

	def sample_frames(self, video_path):
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
			if read_count % (int(frame_total/self.n_frames) -1) == 0:
				frame = cv2.resize(frame, self.operating_resolution)
				frames_list.append(frame)
			read_count += 1
			if len(frames_list) == self.n_frames: break
		return np.array(frames_list[:self.n_frames])

	def get_frame_features(self, frames):
		'''
		Returns features for each frame
		'''
		return np.squeeze(self.image_feature_extractor.predict(frames))

	def extract_video_features(self, video_file):
		'''
		Returns array of fram features for a video
		'''
		frames = self.sample_frames(video_file)
		return self.get_frame_features(frames)

	def prepare_training_data(self, videos_path):
		'''
		Returns data and labels for all videos in a directory
		'''
		folders = sorted(os.listdir(videos_path))
		classes = dict([(folder, idx) for idx, folder in enumerate(folders)])
		n_classes = len(classes)

		frame_features = list()
		labels = list()
		videos_list = list()

		for folder in folders:
			folder_path = os.path.join(self.videos_path, folder)
			video_files = sorted(glob(os.path.join(folder_path, "*")))

			for video_file in video_files:
				frame_features.append(self.extract_video_features(video_file))
				labels.append(classes[folder])
				videos_list.append(video_file)

		return np.array(frame_features), np.array(labels), np.array(videos_list), classes


	def get_training_data(self, save_data_as = None, data_pickle = None):
		'''
		Prepares the preprocessed training data and labels
		'''
		if data_pickle == None:
			if save_data_as == None: save_data_as = "data.pkl"
			if ".pkl" not in save_data_as: save_data_as += ".pkl"

			X, y, video_list, classes = self.prepare_training_data(video_path)

			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = self.test_split, random_state=42)

			pkl.dump({"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test, "classes": classes, "videos": video_list}, open(save_data_as, "wb"))
		else:
			data_dict = pkl.load(open(data_pickle, "rb"))
			X_train, y_train, X_test, y_test, video_list, classes = data_dict["X_train"], data_dict["y_train"], data_dict["X_test"], data_dict["y_test"], data_dict["videos"], data_dict["classes"]

		return X_train, X_test, y_train, y_test, videos, classes

class Trainer(DataHandler):
	'''
	Handles all the training operations
	'''
	def __init__(self, data_to_use = None):
		'''
		Initializes the training class variables
		'''
		DataHandler.__init(self)
		# general
		self.training_version = str(datetime.datetime.now())[:16].replace("-", "_").replace(" ", "_")
		os.mkdir(self.training_version)
		save_data_as = None
		if data_to_use == None:
			save_data_as = os.path.split(model_path)[0] + "data.pkl"

		# data
		self.X_train, self.X_test, self.y_train, self.y_test, self.videos, self.classes = self.get_training_data(save_data_as = save_data_as, data_pickle = data_to_use)
		self.n_classes = len(self.classes)

		# model params
		self.lstm_hidden_units = 200
		self.image_features_size = 1024
		self.lstm_time_steps = 16

		# training params
		self.epochs = 50
		self.batch_size = 32
		self.validation_split = 0.05

	def get_lstm_model(self, print_summary = False):
		'''
		Creates and returns the LSTM model for training
		'''
		input_layer = InputLayer(input_shape = (self.lstm_time_steps, self.image_features_size), name = 'input_layer')
		lstm1 = LSTM(self.lstm_hidden_units, name = 'lstm_1')(input_layer)
		dense1 = Dense(self.n_classes)(lstm1)

		lstm_model = Model(inputs=input_layer, outputs=dense1)
		if print_summary: print(lstm_model.summary())
		return lstm_model

	def train(self, pretrained_model = None, model_path = None):
		'''
		Runs the training
		'''
		if pretrained_model != None: lstm_model = load_model(pretrained_model)
		else: lstm_model = self.get_lstm_model()

		if model_path == None: model_path = "LSTM_E{epoch:02d}_VA{val_accuracy:.2f}.hdf5"
		model_path = os.path.join(self.training_version, model_path)

		callbacks = [ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')]
		
		model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
		model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, shuffle=True, verbose=2, callbacks = callbacks)

class Tester(DataHandler):
	'''
	Testing and evaluation for Conv2D  + LSTM mdeia retrieval model
	'''
	def __init__(self, test_model_path, test_videos_path, lstm_output_layer = "lstm_1"):
		'''
		Initializes class variables for testing Conv2D + LSTM
		'''
		DataHandler.__init__(self, test_videos_path, test_size = 0.01)
		self.X_test, _, self.y_test, _, self.test_videos, self.classes = self.get_training_data(save_data_as = "test_data.pkl", data_pickle = None)

		# load model
		test_lstm_model = load_model(test_model_path)
		self.test_lstm_sub_model = Model(inputs=mobilenet.inputs, output=mobilenet.get_layer(lstm_output_layer).output)

		self.test_video_features = np.squeeze(self.test_lstm_sub_model.predict(self.extract_video_features(self.X_test)))

		self.similariy_metric = cosine_similarity

	def find_closest(self, test_set, test_sample, threshold = 0.95):
		'''
		Returns the indices of the videos that are closest to the test sample based on a threshold
		'''
		if not isinstance(test_sample, np.ndarray): test_sample = np.squeeze(self.test_lstm_sub_model.predict(self.extract_video_features(test_sample))) # when passed a video path
		return np.array([idx for idx, feature in enumerate(test_set) if (self.similariy_metric(feature, test_sample) >= threshold)])

	def retrieve_videos(self, video):
		'''
		Retrieves videos close to the given video
		'''
		retrieved_indices = self.find_closest(self.test_video_features, video)
		return self.test_videos[retrieved_indices]

	def evaluate(self):
		'''
		Evaluates the performance of a model
		'''
		mean_accuracy_list = list()
		for idx, test_sample in enumerate(self.test_video_features):
			retrieved_indices = self.find_closest(self.test_video_features, test_sample)
			actual_label = self.y_test[idx]
			predicted_labels = self.y_test[retrieved_indices]
			retrieval_accuracy = np.array([1 if l == actual_label else 0 for l in predicted_labels]).mean()
			mean_accuracy_list.append(retrieval_accuracy)
		return np.array(retrieval_accuracy).mean()


if __name__ == '__main__':
	pass