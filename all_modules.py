

###################################
########## SECTION MAIN VARIABLES:


results_cloud_path = "drinbd-n-repeat-seas/results/"
landmarks_cloud_path = "drinbd-n-repeat-seas/landmarks/"
# modules_cloud_path = "drinbd-n-repeat-seas/modules/"

gdrive_folder_path = '/content/drive/MyDrive/'

common_path = "https://github.com/GilbertAK/eurusd_data/blob/main/"
dataset_urls = {"dataset_test1":"https://github.com/GilbertAK/EURUSD_2021_10_1_2021_11_15_ohlcv_1_min/blob/main/EURUSD_2021_10_1_2021_11_15_ohlcv_1_min-.csv",
	"dataset_test12":common_path + "EURUSD-2023-07-25_2023-08-31.csv",
	"dataset_test13":common_path + "EURUSD-2023-08-23_2023-09-29.csv",
	"dataset_test14":common_path + "EURUSD-2023-09-22_2023-10-31.csv",
	"dataset_test15":common_path + "EURUSD-2023-10-24_2023-11-30.csv",
	"dataset_test16":common_path + "EURUSD-2023-11-21_2023-12-29.csv",
	"dataset_test17":common_path + "EURUSD-2023-12-13_2024-01-23.csv",
	"dataset_5_min":"https://firebasestorage.googleapis.com/v0/b/saving-data-2ee4b.appspot.com/o/drown-in-bigdata%2Fdatasets%2FEURUSD-5.0%20Min--2024-3-8%2022-0-0.csv?alt=media",
	"dataset_1_hour":"https://firebasestorage.googleapis.com/v0/b/saving-data-2ee4b.appspot.com/o/drown-in-bigdata%2Fdatasets%2FEURUSD-60.0%20Min--2024-3-18%209-0-0.csv?alt=media",
	"dataset_2_hours":"https://firebasestorage.googleapis.com/v0/b/saving-data-2ee4b.appspot.com/o/drown-in-bigdata%2Fdatasets%2FEURUSD-120.0%20Min--2024-3-18%209-0-0.csv?alt=media",
	"dataset_drinbd_n_repeat_seas_1":"https://firebasestorage.googleapis.com/v0/b/saving-data-2ee4b.appspot.com/o/csv_files%2FEURUSD-1.0%20Min--2024-4-18%200-1-0.csv?alt=media",
	"dataset_drinbd_n_repeat_seas_1_added_1":"https://firebasestorage.googleapis.com/v0/b/saving-data-2ee4b.appspot.com/o/datasets_csv%2Fdataset_drinbd_n_repeat_seas_1_added_1.csv?alt=media",
	"dataset_drinbd_n_repeat_seas_1_added_2":"https://firebasestorage.googleapis.com/v0/b/saving-data-2ee4b.appspot.com/o/datasets_csv%2Fdataset_drinbd_n_repeat_seas_1_added_2.csv?alt=media",
	########## Many Pairs:
	'dataset_drinbd_n_repeat_seas_NZDUSD': f'{common_path_firebase_strg}NZDUSD-1.0%20Min--2024-4-18%200-1-0.csv?alt=media',
	'dataset_drinbd_n_repeat_seas_EURGBP': f'{common_path_firebase_strg}EURGBP-1.0%20Min--2024-4-18%200-1-0.csv?alt=media',
	'dataset_drinbd_n_repeat_seas_USDCHF': f'{common_path_firebase_strg}USDCHF-1.0%20Min--2024-4-18%200-1-0.csv?alt=media',
	'dataset_drinbd_n_repeat_seas_GBPJPY': f'{common_path_firebase_strg}GBPJPY-1.0%20Min--2024-4-18%200-1-0.csv?alt=media',
	'dataset_drinbd_n_repeat_seas_NZDJPY': f'{common_path_firebase_strg}NZDJPY-1.0%20Min--2024-4-18%200-1-0.csv?alt=media',
	'dataset_drinbd_n_repeat_seas_CHFJPY': f'{common_path_firebase_strg}CHFJPY-1.0%20Min--2024-4-18%200-1-0.csv?alt=media',
	'dataset_drinbd_n_repeat_seas_GBPUSD': f'{common_path_firebase_strg}GBPUSD-1.0%20Min--2024-4-18%200-1-0.csv?alt=media',
	'dataset_drinbd_n_repeat_seas_EURJPY': f'{common_path_firebase_strg}EURJPY-1.0%20Min--2024-4-18%200-1-0.csv?alt=media',
	'dataset_drinbd_n_repeat_seas_CADJPY': f'{common_path_firebase_strg}CADJPY-1.0%20Min--2024-4-18%200-1-0.csv?alt=media',
	'dataset_drinbd_n_repeat_seas_USDJPY': f'{common_path_firebase_strg}USDJPY-1.0%20Min--2024-4-18%200-1-0.csv?alt=media',
	'dataset_drinbd_n_repeat_seas_EURAUD': f'{common_path_firebase_strg}EURAUD-1.0%20Min--2024-4-18%200-1-0.csv?alt=media',
	'dataset_drinbd_n_repeat_seas_EURNZD': f'{common_path_firebase_strg}EURNZD-1.0%20Min--2024-4-18%200-1-0.csv?alt=media',
	'dataset_drinbd_n_repeat_seas_GBPCAD': f'{common_path_firebase_strg}GBPCAD-1.0%20Min--2024-4-18%200-1-0.csv?alt=media',
	'dataset_drinbd_n_repeat_seas_EURCHF': f'{common_path_firebase_strg}EURCHF-1.0%20Min--2024-4-18%200-1-0.csv?alt=media',
	'dataset_drinbd_n_repeat_seas_GBPCHF': f'{common_path_firebase_strg}GBPCHF-1.0%20Min--2024-4-18%200-1-0.csv?alt=media',
	'dataset_drinbd_n_repeat_seas_EURCAD': f'{common_path_firebase_strg}EURCAD-1.0%20Min--2024-4-18%200-1-0.csv?alt=media',
	'dataset_drinbd_n_repeat_seas_GBPAUD': f'{common_path_firebase_strg}GBPAUD-1.0%20Min--2024-4-18%200-1-0.csv?alt=media',
	'dataset_drinbd_n_repeat_seas_AUDJPY': f'{common_path_firebase_strg}AUDJPY-1.0%20Min--2024-4-18%200-1-0.csv?alt=media',
	'dataset_drinbd_n_repeat_seas_USDCAD': f'{common_path_firebase_strg}USDCAD-1.0%20Min--2024-4-18%200-1-0.csv?alt=media',
	'dataset_drinbd_n_repeat_seas_AUDUSD': f'{common_path_firebase_strg}AUDUSD-1.0%20Min--2024-4-18%200-1-0.csv?alt=media',
	}

### ===> firebase of account philippenshagi@gmail.com
## rapid link :
## realtime database: 
## https://console.firebase.google.com/project/saving-data-2ee4b/database/saving-data-2ee4b-default-rtdb/data
## firebase storage:
## https://console.firebase.google.com/project/saving-data-2ee4b/storage
firebase_config = {
	"apiKey": "AIzaSyANPOUHxWG48LpdFATg10gJwY42ouX5p04",
	"authDomain": "saving-data-2ee4b.firebaseapp.com",
	"databaseURL": "",
	"projectId": "saving-data-2ee4b",
	"storageBucket": "saving-data-2ee4b.appspot.com",
	"messagingSenderId": "403260823146",
	"appId": "1:403260823146:web:597c2ac9422b7aa37e1c13",
	"measurementId": "G-3P1DE9GSD5"
}



###################################
########## SECTION UTILS:


import datetime
import warnings
import pandas as pd
import os
import platform
import time


def print_style(text, color = None, bold = False, underline = False):
	colors = {"purple":'\033[95m',
			"cyan":'\033[96m',
			"darkcyan":'\033[36m',
			"blue":'\033[94m',
			"green":'\033[92m',
			"yellow":'\033[93m',
			"red":'\033[91m'}
	other_style = {"bold":'\033[1m',
				"underline":'\033[4m'}
	end = '\033[0m'

	if check_environment() == 'Colab':
		if color is None and bold and not underline:
			print(other_style['bold'] + text + end)
		if color is None and not bold and underline:
			print(other_style['underline'] + text + end)
		if color is None and bold and underline:
			print(other_style['bold'] + other_style['underline'] + text + end)
		if bold and not underline and color is not None:
			print(colors[color.lower()] + other_style['bold'] + text + end)
		if underline and not bold and color is not None:
			print(colors[color.lower()] + other_style['underline'] + text + end)
		if underline and bold and color is not None:
			print(colors[color.lower()] + other_style['bold'] + other_style['underline'] + text + end)
		if not bold and not underline and color is not None:
			print(colors[color.lower()] + text + end)
		if not bold and not underline and color is None:
			print(text)
	elif check_environment() == 'Local_PC':
		print(text)


def install_pyrebase():
	### Install Pyrebase4
	for _ in range(2):
		try:
			import pyrebase
			break
		except:
			# !pip install Pyrebase4
			print_style("Installing Pyrebase4 ...\n", color = 'cyan')
			os.system("pip install Pyrebase4")

def manage_wargins():
	pd.options.mode.chained_assignment = None
	warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def check_environment():
	env_ = os.environ
	if 'COLAB_GPU' in env_ or 'COLAB_JUPYTER_IP' in env_:
		return 'Colab'
	if platform.uname().node == "Gilbert-PC":
		return 'Local_PC'

def check_gdrive_connected(waiting = 60):
	if check_environment() == 'Colab':
		if not os.path.isdir(s = "/content/drive/MyDrive/"):
			print("\n")
			print_style("You are not connected to Google Drive", color = "red", bold = True)
			print_style("Please connect !", color = "yellow", bold = True)
			continue_runtime = time.time() + waiting + (3600*2)
			continue_runtime = datetime.datetime.fromtimestamp(continue_runtime)

			now_ = time.time() + (3600*2)
			print("Datetime now is :", datetime.datetime.fromtimestamp(now_))
			print(f"Next lines of code will run within {waiting} second(s). \n\tSo at datetime: {continue_runtime}.")
			time.sleep(waiting)

	else:
		print("We are not using Google Colab !")

def save_pickle(filepath, data):
	with open(filepath, 'wb') as file_pi:
		pickle.dump(data, file_pi)

def load_pickle(filepath):
	with open(filepath, "rb") as file_pi:
		loaded_ = pickle.load(file_pi)
	return loaded_

def check_file_exists(filepath):
	return os.path.isfile(path = filepath)

def wait_gdrive_connected():
	"""
	check if google drive is already connected:
	- if so, pass
	- else: wait.
	"""
	if check_environment() == 'Colab':
		start_time_while_drive = time.time()
		print_style('\nDrive not yet connected !', color = 'red', bold = True)
		while True:
			if not os.path.isdir(s = "/content/drive/MyDrive/"):
				time.sleep(1)			
			else:
				print_style("\nOkay, Drive successfully connected !", color = 'green', bold = True)
				elasped_while_drive = time.time() - start_time_while_drive
				print_style(f"Elapsed time : {round(elasped_while_drive, 2)} seconds or {round(elasped_while_drive/60, 2)} minutes.\n", 
					color = 'cyan', bold = True)
				break


if platform.uname().node != "Gilbert-PC" and check_environment() != 'Colab':
	print("\n\tWe are not neither on Gilbert-PC nor on Colab !!!")
	time.sleep(60)


###################################
########## SECTION DATASCOENCE:

from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import random
import pickle
import time
import pywt

pd.options.mode.chained_assignment = None

class ManageWithWaveletes:
	def __init__(self, signal):
		self.signal = signal


	### 1. Signal Analysis using DWT
	def dwt_signal_analysis(self, plot_result = False):
		# Apply DWT:
		coeffs = pywt.dwt(self.signal, 'db1')
		cA, cD = coeffs
		# ### (cA, cD) : Approximation and detail coefficients.
		# print('len signal :', len(self.signal))
		# print('len of cA :', len(cA))
		# print('len of cD :', len(cD))
		# print("len cA + len cD :", len(cA) + len(cD))
		# print("len(cA) + len(cD) == len(self.signal) :", len(cA) + len(cD) == len(self.signal))

		if plot_result:
			# Plotting
			plt.figure(figsize=(12, 4))
			
			plt.subplot(1, 3, 1)
			plt.plot(self.signal)
			plt.title("Original Signal")

			plt.subplot(1, 3, 2)
			plt.plot(cA)
			plt.title("Approximation Coefficients")

			plt.subplot(1, 3, 3)
			plt.plot(cD)
			plt.title("Detail Coefficients")

			plt.tight_layout()
			plt.show()

		return {"cA":cA, "cD":cD}



	# 2. Denoising Signal Using Wavelet Transform
	def denoise_with_wavelet_trans(self, threshold, plot_result = False, mode = 'soft'):
		
		# mode possibilities : {'soft', 'hard', 'garrote', 'greater', 'less'}

		# Perform a multi-level wavelet decomposition
		coeffs = pywt.wavedec(self.signal, 'db1', level=4)

		# Set a threshold to nullify smaller coefficients (assumed to be noise)
		coeffs_thresholded = [pywt.threshold(c, threshold, mode=mode) for c in coeffs]

		# Reconstruct the signal from the thresholded coefficients
		denoised_signal = pywt.waverec(coeffs_thresholded, 'db1')

		# print("len of self.signal    :", len(self.signal))
		# print("len of denoised_signal :", len(denoised_signal))
		# print('len(self.signal) == len(denoised_signal) :', len(self.signal) == len(denoised_signal))

		if plot_result:

			# Plotting the noisy and denoised signals
			plt.figure(figsize=(12, 4))
			plt.subplot(1, 2, 1)
			plt.plot(self.signal)
			plt.title("Noisy Signal")

			plt.subplot(1, 2, 2)
			plt.plot(denoised_signal)
			plt.title("Denoised Signal")

			plt.tight_layout()
			plt.show()

		return denoised_signal


### STATIONNARISATION DES DONNEES:
### ______________________________
def stationnarizer(df, column_name, limit = None):
	def stationnarity_checker(time_series):
		# print("stationnarity_checker 1")
		result = adfuller(time_series)
		# print("stationnarity_checker 2")
		labels = ["ADF Test Statistic", "p-value", "Number of Observations Used"]
		# print("stationnarity_checker 3")
		if result[1] <= 0.05:
			# print("stationnarity_checker 4")
			return "Données stationnaires."
		else:
			# print("stationnarity_checker 5")
			return "Données NON stationnaires."

	# print("Start stationnarity 1")
	_df = df.copy()
	# print("Start stationnarity 2")
	_df['Diff 0'] = _df[column_name]
	# print("Start stationnarity 3")

	for i in range(1, 1_000_000):
		# print("Start stationnarity 4")
		_df[f'Diff {str(i)}'] = _df[f'Diff {str(i-1)}'] - _df[f'Diff {str(i-1)}'].shift(1)
		# print("Start stationnarity 5")
		_df[f'Diff {str(i)}'].fillna(method = "bfill", inplace = True)
		# print("Start stationnarity 6")
		result_check_stationnarity = stationnarity_checker(time_series = _df[f'Diff {str(i)}'])
		# print("Start stationnarity 7")
		# print("not found !!")

		if result_check_stationnarity == "Données stationnaires.":
			data = _df[f"Diff {str(i)}"].copy()
			# print(len(data))
			# print("Stationnarized at :", i+1)
			return data

		elif limit is not None and i >= limit:
			data = _df[f"Diff {str(i)}"].copy()
			# print(len(data))
			# print("Stationnarized at :", i+1)
			return data


def stationnarizer_v2(df, column_name, limit):
	_df = df.copy()
	_df['Diff 0'] = _df[column_name]
	for i in range(1, limit+1):
		_df[f'Diff {str(i)}'] = _df[f'Diff {str(i-1)}'] - _df[f'Diff {str(i-1)}'].shift(1)
		_df[f'Diff {str(i)}'].fillna(method = "bfill", inplace = True)

	return _df[f'Diff {limit}']


def duplicate_rows(series):
	new_col = [(item, item) for item in series]
	new_col_ = []
	for item in new_col:
		new_col_.append(item[0])
		new_col_.append(item[1])
	return new_col_


def is_pair(value): # odd = impair
	assert isinstance(value, int), '"value" must be an interger.'
	if value%2 == 0:
		return True
	else:
		return False


def add_wavelets_columns(close_column, limit_stationnarization = 3):

	close_column = close_column.copy()
	# print("len close_column :", close_column.shape[0])
	# print(close_column)

	df_dwt = pd.DataFrame()
	manage_with_waveletes = ManageWithWaveletes(signal = close_column)
	dwt_ = manage_with_waveletes.dwt_signal_analysis()
	cA = list(dwt_["cA"])
	cD = list(dwt_["cD"])

	# print("len(cA) :",len(cA))
	# print("len(cD) :",len(cD))

	cA_duplicated = duplicate_rows(series = cA)
	cD_duplicated = duplicate_rows(series = cD)

	if not is_pair(close_column.shape[0]):
		cA_duplicated = cA_duplicated[1:]
		cD_duplicated = cD_duplicated[1:]

	# print("len cA_duplicated is:", len(cA_duplicated))
	# print("len cD_duplicated is:", len(cD_duplicated))

	df_dwt['dwt_cA'] = cA_duplicated
	df_dwt['dwt_cD'] = cD_duplicated
	df_dwt['dwt_cA_stationnarized'] = stationnarizer_v2(df = df_dwt,
											column_name = "dwt_cA",
											limit = limit_stationnarization)
	# print(df_dwt)
	return {"dwt_cA":df_dwt['dwt_cA'].tolist(), 
			"dwt_cD":df_dwt['dwt_cD'].tolist(), 
			"dwt_cA_stationnarized":df_dwt['dwt_cA_stationnarized'].tolist()}


def stationnarize_close_column(df, close_column_name, limit_stationnarization = 3):
	df = df.copy()
	return stationnarizer_v2(df = df,
						column_name = close_column_name,
						limit = limit_stationnarization).to_list()


def add_target(df,
	close_column_name, 
	target_type,
	target_shift,
	ratio_true_trend = None,
	diviseur_ecart_entre_list_values = 1):

	"""
	diviseur_ecart_entre_list_values: Si jamais on cherher à classifier la tendance et qu'en filtrant
	l'etiquettage on obtient une valeur tassez superieur au pourcentage voulu (==> ratio_true_trend)
	alors il faut augmenter la valeur de 'diviseur_ecart_entre_list_values' qui est égale à 1 par
	defaut.
	On peut essayer avec 2 ou 5 ou 10 etc.
	"""

	df = df.copy()
	def get_range_neutral_trend(list_, ratio_true_trend, diviseur_ecart_entre_list_values = 1):
		assert 0.0 < ratio_true_trend < 1.0, "The value of 'ratio_true_trend' must be in the interval ]0,1[."
		ratio_trend_neutral = 1.0 - ratio_true_trend
		ecart = 1e-06/diviseur_ecart_entre_list_values
		step = 0
		while True:
			step += ecart
			interval = (-step, step)
			low_data = [d for d in list_ if d>= interval[0] and d<interval[1]]
			ratio = len(low_data)/len(list_)
			if ratio >= ratio_trend_neutral:
				return {"interval":interval, "ratio_get":1-ratio}
			if ratio >= 1.0:
				break

	df_to_treat = df.copy()
	### ADD TARGET COLUMN:
	assert target_type.lower() == 'classification' or target_type.lower() == 'regression', \
					"The value of 'target_type' must be 'classification' or 'regression'."

	target = df[close_column_name].shift(-abs(target_shift))
	if target_type.lower() == "regression":
		df_to_treat['target'] = target

	else: ### if target type is classification:
		df_target = pd.DataFrame({close_column_name:df[close_column_name], 'target':target})
		df_target['sens'] = df_target['target'] - df_target[close_column_name]
		list_ = df_target[['sens']]
		list_.dropna(inplace = True)
		list_ = list_['sens'].tolist()
		range_neutral_trend_ratio_get = get_range_neutral_trend(list_ = list_,
													ratio_true_trend = ratio_true_trend,
													diviseur_ecart_entre_list_values = diviseur_ecart_entre_list_values)
		range_neutral_trend = range_neutral_trend_ratio_get["interval"]
		ratio_get = range_neutral_trend_ratio_get["ratio_get"]

		if (ratio_true_trend - ratio_get)*100 >= 5:
			good_ratio = False
			text_ = f"Veuillez augmenter la valeur de 'diviseur_ecart_entre_list_values' car le ratio obtenu ratio_get: {ratio_get} est assez écarté de celui voulu, ratio_true_trend: {ratio_true_trend}"
			assert good_ratio == True, text_
		# elif ratio_get - ratio_true_trend >= 5:
		# 	good_ratio = False
		# 	text_ = "Veuillez diminuer la valeur de 'diviseur_ecart_entre_list_values' car le ratio obtenu est assez écarté de celui voulu."
		# 	assert good_ratio == True, text_

		target = np.where(df_target['sens'] < range_neutral_trend[0], -1,
								np.where(df_target['sens'] > range_neutral_trend[1], 1, 0.0))
		df_to_treat['target'] = target
	stop_at = len(df_to_treat) - target_shift
	df_to_treat = df_to_treat.head(stop_at)
	return df_to_treat



def data_scaler(df):
	dataset = df.copy()
	columns = dataset.columns.tolist()
	scaler = MinMaxScaler(feature_range=(0, 1))
	dataset = scaler.fit_transform(dataset)
	dataset = pd.DataFrame(dataset, columns = columns)
	return dataset, scaler


def data_unscaler(df, scaler):
	df = df.copy()
	columns = df.columns.tolist()
	unscaled_df = scaler.inverse_transform(df)
	unscaled_df = pd.DataFrame(unscaled_df, columns = columns)
	return unscaled_df


def native_x_y_spliter(df, data_cols_names, target_col_name, look_back):
	assert look_back >= 1, "The 'look_back' value must be >= 1"
	df.reset_index(inplace = True, drop = True)
	dataX = []
	dataY = []
	for idx in df.index:
		fragment = df.iloc[idx:idx+look_back, :]
		fragment_X = np.array(fragment[data_cols_names])
		fragment_Y = np.array(fragment[target_col_name].tolist()[-1])
		if len(fragment) < look_back:
			break
		dataX.append(fragment_X)
		dataY.append(fragment_Y)
	dataX = np.array(dataX)
	dataY = np.array(dataY)
	return {"dataX":dataX, "dataY":dataY}


###################################
########## SECTION DL MODEL:


# try:
# 	from modules import main_variables
# 	from modules import utils
# except:
import datetime
import time

# try:
# 	from drinbd_n_repeat_seas import main_variables
# 	from drinbd_n_repeat_seas import utils
# except:
# 	import main_variables
# 	import utils

exec_environment = check_environment()

if exec_environment == 'Colab':
	from tensorflow import keras
	from tensorflow.keras.models import load_model
	import tensorflow as tf

	class EachEpochCallback(keras.callbacks.Callback):
		def __init__(self, verbose_epoch_in_callback, 
			epochs, history_filepath, save_prev_epochs_filepath, 
			loss_n_val_loss, log_fitting_filepath):
			
			self.durations = []
			self.verbose_epoch_in_callback = verbose_epoch_in_callback
			self.epochs = epochs
			self.history_filepath = history_filepath
			self.save_prev_epochs_filepath = save_prev_epochs_filepath
			self.log_fitting_filepath = log_fitting_filepath
			self.loss_n_val_loss = loss_n_val_loss
		
		def on_epoch_begin(self, epoch, logs=None):
			self.now = time.time()

		def on_epoch_end(self, epoch, logs):
			later = time.time()
			duration = later - self.now
			self.durations.append(duration)

			with open(self.log_fitting_filepath, "a", encoding = "utf-8") as f:
				f.write(f"Epoch          : {epoch + 1}\n")
				f.write(f"On epoch start : {datetime.datetime.fromtimestamp(self.now)}\n")
				f.write(f"On epoch end   : {datetime.datetime.fromtimestamp(later)}\n")
				f.write(f"Epoch duration : {duration} second(s)\n\n")

			self.loss_n_val_loss['loss'].append(logs["loss"])
			self.loss_n_val_loss['val_loss'].append(logs["val_loss"])
			save_pickle(filepath = self.history_filepath, data = self.loss_n_val_loss)

			with open(self.save_prev_epochs_filepath, "a", encoding = "utf-8") as f:
				f.write(f'{epoch}\n')

			if self.verbose_epoch_in_callback:
				print(f"Epoch : {epoch+1}/{self.epochs}")

			if (epoch+1)%3 == 0:
				staying_epochs = self.epochs - (epoch+1)
				average_duration = sum(self.durations)/len(self.durations)
				staying_time = average_duration * staying_epochs
				finish_at = time.time() + staying_time
				now_time_ = time.time()
				if exec_environment == 'Colab':
					finish_at += (3600*2)
					now_time_ += (3600*2)
				finish_at = datetime.datetime.fromtimestamp(int(finish_at))
				now_time_ = datetime.datetime.fromtimestamp(int(now_time_))

				print("\n")
				print("Time now is            :", now_time_)
				print("Running will finish at :", finish_at)
				print("Staying time           :", round(staying_time/60, 3), "Minutes or", round(staying_time/3600, 3), "Hours")
				print("\n")


def fit_n_save_model(X_train, y_train,
					epochs, test_nbr,
					verbose_epoch_in_callback,
					gdrive_folder_path,
					prevent_wait,
					verbose_model_checkpoint = False,
					batch_size = 32,
					validation_split = 0.3,
					shuffle = False,
					verbose_fit = True,):

	################
	### FILES PATHS:
	################

	if not gdrive_folder_path.endswith("/"):
		gdrive_folder_path += "/"
	
	save_prev_epochs_filepath = gdrive_folder_path + f'prev_epochs_test_nbr_{test_nbr}.txt'
	model_filepath = gdrive_folder_path + "model_test_nbr_" + f'{test_nbr}.h5'
	history_filepath = gdrive_folder_path + f'history_test_nbr_{test_nbr}.pkl'
	log_fitting_filepath = gdrive_folder_path + f'log_fitting_test_nbr_{test_nbr}.txt'

	##########################
	## LOAD OR CREATE HISTORY:
	##########################

	print(f"Checking presence of: {history_filepath.split('/')[-1]}")
	history_filepath_exists = False
	for _ in range(5):
		if check_file_exists(filepath = history_filepath):
			history_filepath_exists = True
			break
		else:
			time.sleep(1)

	if history_filepath_exists:
		loss_n_val_loss = load_pickle(filepath = history_filepath)
		print_style(text = f"\nOkay {history_filepath.split('/')[-1]} found !!!\n", 
						color = 'green', 
						bold = True,
						underline = False)
		# time.sleep(3)
	else:
		for _ in range(10):
			print_style(text = "Saving history will start from zero !!!", 
							color = "red", 
							bold = True, 
							underline = False)
		time.sleep(prevent_wait[0])
		loss_n_val_loss = {'loss':[], 'val_loss':[]}

	#######################################################
	### LOAD OR CREATE MODEL AND LAST EPOCH LANDMARK FILE:
	#######################################################

	print(f"Checking presence of: {model_filepath.split('/')[-1]}")
	model_filepath_exists = False
	for _ in range(5):
		if check_file_exists(filepath = model_filepath):
			model_filepath_exists = True
			break
		else:
			time.sleep(1)

	print(f"Checking presence of: {save_prev_epochs_filepath.split('/')[-1]}")
	save_prev_epochs_filepath_exists = False
	for _ in range(5):
		if check_file_exists(filepath = save_prev_epochs_filepath):
			save_prev_epochs_filepath_exists = True
			break
		else:
			time.sleep(1)

	if model_filepath_exists and save_prev_epochs_filepath_exists:

		### Load the model
		model = load_model(model_filepath)
		with open(save_prev_epochs_filepath, "r", encoding = "utf-8") as f:
			last_epochs = f.readlines()
		last_epochs = [item.strip() for item in last_epochs]
		all_previous_epochs_nbr = len(last_epochs)
		epochs -= all_previous_epochs_nbr
		print_style(text = f"\nOkay {model_filepath.split('/')[-1]} and {save_prev_epochs_filepath.split('/')[-1]} found !!!\n",
					color = "green", 
					bold = True, 
					underline = False)
		# time.sleep(3)
	else:
		for _ in range(10):
			print_style(text = "Fitting model will start from zero !!!", 
						color = "red", 
						bold = True, 
						underline = False)

		time.sleep(prevent_wait[1])

		### Creating the model:
		dropout = 0.2
		model = keras.Sequential()
		model.add(keras.layers.Bidirectional(
			keras.layers.LSTM(units=128, input_shape=(X_train.shape[1], X_train.shape[2]))))
		model.add(keras.layers.Dropout(rate=dropout))
		model.add(keras.layers.Dense(units=1))
		model.compile(loss='mean_squared_error', optimizer='adam')

	###################
	### FIT THE MODEL:
	###################

	each_epoch_callback = EachEpochCallback(verbose_epoch_in_callback = verbose_epoch_in_callback, 
										epochs = epochs, history_filepath = history_filepath, 
										save_prev_epochs_filepath = save_prev_epochs_filepath, 
										loss_n_val_loss = loss_n_val_loss,
										log_fitting_filepath = log_fitting_filepath)

	model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath = model_filepath,
																monitor = 'loss',
																mode = 'min',
																save_best_only = True,
																verbose = verbose_model_checkpoint)

	history = model.fit(
				X_train, y_train,
				epochs = epochs,
				batch_size = batch_size,
				validation_split = validation_split,
				shuffle = shuffle,
				verbose = verbose_fit,
				callbacks = [model_checkpoint_callback, each_epoch_callback],
				)

def load_the_model(gdrive_folder_path, test_nbr):
	if not gdrive_folder_path.endswith("/"):
		gdrive_folder_path += "/"
	model_filepath = gdrive_folder_path + "model_test_nbr_" + f'{test_nbr}.h5'
	return load_model(model_filepath)




###################################
########## SECTION EVALUATE RESULTS:


import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)

def get_balance_sheet(df_results, 
	close_column_name, target_shift, y_pred_col_name, verbose, test_nbr, save_to = None):
	
	def get_rapprochement(y_pred):
		diff_0 = abs(y_pred - 0.0)
		diff_0_5 = abs(y_pred - 0.5)
		diff_1 = abs(y_pred - 1.0)

		all_diffs = [diff_0, diff_0_5, diff_1]
		min_ = min(all_diffs)
		if min_ == diff_0:
			return 0.0
		elif min_ == diff_0_5:
			return 0.5
		else:
			return 1.0

	df = df_results.copy()
	len_df = df.shape[0]
	df['future_price'] = df[close_column_name].shift(-abs(target_shift))
	df['true_sens'] = df['future_price'] - df[close_column_name]
	df['true_sens'] = np.where(df['true_sens'] > 0, 1.0,
						np.where(df['true_sens'] < 0, 0.0, 0.5))

	df.dropna(subset = ['future_price'], inplace = True)
	df.reset_index(inplace = True, drop = True)
	df['y_pred_rppchmt'] = list(map(get_rapprochement, df[y_pred_col_name]))

	### IF THE MARKET IS NEUTRAL: WE DON'T LOSE AND DON'T WIN ANYTHING, SO LET'S DROP THE
	### ROWS WHERE TRUE SENS == 0.5:
	df = df[df['true_sens'] != 0.5]

	### IF THE SIGNAL y_pred_rppchmt IS 0.5, WE DON'T TAKE ANY POSITION, SO LET'S DROP THE ROWS
	### WHERE y_pred_rppchmt == 0.5:
	df = df[df['y_pred_rppchmt'] != 0.5]
	df['result'] = df['y_pred_rppchmt'] == df['true_sens']
	wins = df['result'].tolist().count(True)
	loses = df['result'].tolist().count(False)
	total_trades = wins + loses
	try:
		accuracy = wins/total_trades
		accuracy = str(round(accuracy*100, 2)) + "%"

	except ZeroDivisionError:
		accuracy = 'No trade taken'

	try:
		ratio_trades = total_trades/len_df
		ratio_trades = str(round(ratio_trades*100, 2)) + "%"
	except ZeroDivisionError:
		ratio_trades = 'No trade taken'

	# print("\n\n\t\tPlease wait !!!")
	if verbose:
		print(f"\n\tTest nbr     : {test_nbr} ")
		print("\t_______________________")
		print(f"\n\t  Total Trades : {total_trades}")
		print(f"\t  Df length    : {len_df}")
		print(f"\t  Ratio Trades : {ratio_trades}\n")
		print(f"\t  Wins         : {wins}")
		print(f"\t  Loses        : {loses}")
		print(f"\t  Accuracy     : {accuracy}")

	if save_to is not None:
		with open(save_to, "w", encoding = "utf-8") as f:
			f.write(f"\n\tTest nbr     : {test_nbr} \n")
			f.write("\t_______________________\n")
			f.write(f"\n\t  Total Trades : {total_trades}\n")
			f.write(f"\t  Df length    : {len_df}")
			f.write(f"\n\t  Ratio Trades : {ratio_trades}\n\n")
			f.write(f"\t  Wins         : {wins}\n")
			f.write(f"\t  Loses        : {loses}\n")
			f.write(f"\t  Accuracy     : {accuracy}\n")





###################################
########## SECTION MANAGE FIREBASE:


## ===> it is firebase of account philippenshagi@gmail.com

# rapid link :
# realtime database: 
# https://console.firebase.google.com/project/saving-data-2ee4b/database/saving-data-2ee4b-default-rtdb/data

# firebase storage:
# https://console.firebase.google.com/project/saving-data-2ee4b/storage

# try:
# 	from modules import main_variables
# 	from modules import utils
# except:
# import main_variables
# import utils


### INSTALL AND IMPORT PYREBASE:
###_____________________________
install_pyrebase()
# import pyrebase

class FirebaseStorage:
	def __init__(self, firebase_config):
		import pyrebase
		firebase = pyrebase.initialize_app(firebase_config)
		self.storage = firebase.storage()

	def upload_file(self, local_file_path_name, cloud_file_path_name):
		# print("\n\n*******local_file_path_name :", local_file_path_name)
		uploading = self.storage.child(cloud_file_path_name).put(local_file_path_name)
		url_file_on_cloud = self.storage.child(cloud_file_path_name).get_url(None)
		return url_file_on_cloud

	def download_file(self, cloud_file_path_name, local_file_path_name, verbose = False):
		try:
			self.storage.child(cloud_file_path_name).download(
								'', filename = local_file_path_name)
			if verbose:
				print("File successfully downloaded !")
		except Exception as e:
			print("Error :\n", e)


firebase_storage = FirebaseStorage(
				firebase_config = firebase_config)

def direct_upload_file(local_file_path_name, cloud_file_path_name):
	try:
		url = firebase_storage.upload_file(
									local_file_path_name = local_file_path_name, 
									cloud_file_path_name = cloud_file_path_name)
		return url
	except FileNotFoundError:
		return False

def direct_download_file(cloud_file_path_name, local_file_path_name):
	firebase_storage.download_file(
				cloud_file_path_name = cloud_file_path_name, 
				local_file_path_name = local_file_path_name)

	if check_file_exists(filepath = local_file_path_name):
		return True
	else:
		return False


###################################
########## SECTION REGENERATE SEASONALITY:


from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from collections import deque

import matplotlib.pyplot as plt 
from pprint import pprint
import pandas as pd
import numpy as np
import math

pd.options.mode.chained_assignment = None


### ETS DECOMPOSITION
def seasonal_decompose_(series, freq, model = "additive"):
	try:
		decomposition = seasonal_decompose(series, 
			freq = freq, ## pour une saisonnalité de 60 unités.
			model = model, ## values "additive" or "multiplicative"
			)
	except TypeError:
		decomposition = seasonal_decompose(series, 
			period = freq, ## pour une saisonnalité de 60 unités.
			model = model, ## values "additive" or "multiplicative"
			)
	return decomposition


def fragmentor(len_group, series, limit = None):
	"""
	for example we have:
		series = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
			and
		len_group = 2
			frgts returned is: [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]

	OR
		series = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
			and
		len_group = 3
			frgts returned is: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

	NB: if limit is for example = 2 not None frgts returned is [[1, 2, 3], [4, 5, 6]]
	"""
	frgts = []
	df_ = pd.DataFrame({'column':series})
	for elt in range(len(series)//len_group):
		frgt = series[:len_group]
		[series.remove(item) for item in frgt]
		frgts.append(frgt)
		if limit is not None and len(frgts) == limit:
			break
	return frgts, df_['column'].tolist()



def start_by_first_extrema(series):
	"""
	for example we have a series like: [4,3,5, 1,5,4,9,6,7, 1,5,4,9,6,7, 1,5,4,9,6,7,]
		its max = 9 and min = 1, those are extremum
		the first extrema is the mininum, so 1 because it apears firstly on index 3
		and the last (second) extrema is the maximum, so 9, because it apears firstly 
		on index 6.
		So here we will slice the series from index min(3, 6) => 3

		So synthetically:
		The series : [4, 3, 5, 1, 5, 4, 9, 6, 7, 1, 5, 4, 9, 6, 7, 1, 5, 4, 9, 6, 7]
		Becomes :			  [1, 5, 4, 9, 6, 7, 1, 5, 4, 9, 6, 7, 1, 5, 4, 9, 6, 7]


		Other example:
		The series : [4, 3, 5, 9, 3, 5, 1, 5, 9, 3, 5, 1, 5, 9, 3, 5, 1, 5, 9, 3, 5, 1, 5]
		Becomes :			  [9, 3, 5, 1, 5, 9, 3, 5, 1, 5, 9, 3, 5, 1, 5, 9, 3, 5, 1, 5]

	"""

	series = list(series)
	min_ = min(series)
	max_ = max(series)
	### get index of first min value:
	for idx, item in enumerate(series):
		if item == min_:
			idx_min = idx
			break
	### get index of first max value:
	for idx, item in enumerate(series):
		if item == max_:
			idx_max = idx
			break
	start_by = min(idx_min, idx_max)
	series = series[start_by:]
	return series


def comparator(big_list):
	"""
	This function allows to know if ALL items of the "big_list" are identic,
		if so return True
		else return False
	"""
	falses = 0
	first_list = big_list[0]
	for list_ in big_list:
		if first_list != list_:
			falses += 1
			return False
	if falses == 0:
		return True


def add_seasonality(original_series, seasonal_fragment, seasons_2_add):
	original_series = list(original_series)
	for _ in range(len(seasonal_fragment)):
		original_last_values = list(deque(original_series, len(seasonal_fragment)))
		right_original_series = original_series
		original_series = original_series[:-1]
		checker = original_last_values == seasonal_fragment
		if checker:
			break
	series_with_more_seasons = right_original_series + (seasonal_fragment*seasons_2_add)
	return series_with_more_seasons


def regenerate_seasonality(close_column, 
	seasons_2_add, freq, plot_results = False, model = "additive", limit = 5):
	close_column = close_column.copy()
	dec = seasonal_decompose_(series = close_column, 
							freq = freq, 
							model = model)

	seasonal_ = dec.seasonal

	# plt.subplot(3,1,1)
	# plt.plot(seasonal_)

	df_original_seasonnal = pd.DataFrame({"original":seasonal_})

	seasonal_ = list(seasonal_)
	seasonal_ = start_by_first_extrema(series = seasonal_)
	for len_group in range(freq, len(seasonal_)):
		fragments, seasonal_ = fragmentor(len_group = len_group, series = seasonal_, limit = limit)
		checker_ = comparator(big_list = fragments)
		if checker_:
			seasonal_fragment = fragments[0]
			break

	assert len(seasonal_fragment) == freq, f'length of seasonal_fragment = {len(seasonal_fragment)} and freq = {freq}, however they should be identic. You shall increase the value of "limit".'
	
	series_with_more_seasons = add_seasonality(original_series = df_original_seasonnal['original'], 
									seasonal_fragment = seasonal_fragment, 
									seasons_2_add = seasons_2_add)

	# print("series_with_more_seasons :", series_with_more_seasons)


	if plot_results:
		###########
		plt.subplot(5, 1, 1)
		plt.plot(close_column)

		plt.subplot(5, 1, 2)
		plt.plot(df_original_seasonnal)

		plt.subplot(5, 1, 3)
		plt.plot(series_with_more_seasons)

		plt.subplot(5, 1, 4)
		plt.plot(df_original_seasonnal, '*')

		plt.subplot(5, 1, 4)
		plt.plot(series_with_more_seasons)

		plt.subplot(5, 1, 5)
		plt.plot(seasonal_fragment*int(len(df_original_seasonnal)/freq))

		plt.tight_layout()
		plt.show()
		###########

	# ### return seasonal_fragment
	return series_with_more_seasons


def add_seas_column_to_df(df, size_get_seas, freq, close_col_name, 
	plot_results = False, model = "additive", limit = 5):

	df = df.copy()

	len_df_get_seas = int(df.shape[0]*size_get_seas)
	df_get_seas = df.head(len_df_get_seas)

	# print("df_get_seas :", df_get_seas.shape[0])

	seasons_2_add = math.ceil(df.shape[0]/freq)
	series_with_more_seasons = regenerate_seasonality(
								close_column = df_get_seas[close_col_name],
								seasons_2_add = seasons_2_add,
								freq = freq)

	assert len(series_with_more_seasons) >= df.shape[0], 'length of "series_with_more_seasons" shall be >= to len df original entered in this function: def add_seas_column_to_df.'
	seasonality_column = series_with_more_seasons[:df.shape[0]]

	# print("len seasonality_column :", len(seasonality_column))
	# print("type of seasonality_column :", type(seasonality_column))
	# # print("len df get seas              :", len_df_get_seas)
	# # print("len df original              :", df.shape[0])

	df[f'seasonality_{freq}'] = seasonality_column

	return df


###################################
########## SECTION MANAGE:

# from statsmodels.tsa.seasonal import seasonal_decompose
# from sklearn.model_selection import train_test_split
# from statsmodels.tsa.stattools import adfuller
# from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import time
import os
# import random
# import pickle
# import pywt
# import warnings
# import platform
# import pyrebase

# from drinbd_n_repeat_seas import datascience
# from drinbd_n_repeat_seas import dl_model
# from drinbd_n_repeat_seas import evaluate_results
# from drinbd_n_repeat_seas import main_variables
# from drinbd_n_repeat_seas import manage_firebase
# from drinbd_n_repeat_seas.regenerate_seasonality import add_seas_column_to_df
# from drinbd_n_repeat_seas import utils



if check_environment() == 'Colab':
	### Install Pyrebase4
	for _ in range(2):
		try:
			import pyrebase
			break
		except:
			# !pip install Pyrebase4
			print("Installing Pyrebase4 ...\n")
			os.system("pip install Pyrebase4")



"""
DON'T FORGET:
____________

- pendant le test du code: interrompre le fitting du model puis re-executer le code pour voir si ce
fitting sera resumé (pourra continuer là ou il s'est arreté)

- 

"""



def manage(test_nbr,
		dataset_key,
		close_column_name,
		target_col_name,
		freqs_seasonal,
		target_shift,
		target_type,
		ratio_true_trend,
		train_size,
		look_back,
		epochs,
		verbose_epoch_in_callback,
		verbose_model_checkpoint,
		
		batch_size_fit,
		validation_split_fit,
		shuffle_fit,
		verbose_fit,
		sub_concept_n_size,
		
		# modulo_verbose_x_test_comp,
		# results_cloud_path,
		verbose_eval,
		prevent_wait,
		slice_df_index = None,
		df_tail = None,
		df_head = None,
		):

	### WAIT GOOGLE DRIVE CONNECTED:
	###_____________________________
	wait_gdrive_connected()

	# ### INSTALL PYREBASE:
	# ###__________________
	# install_pyrebase()

	### PRINT TEST NBR:
	###________________

	print("\n"*2)
	for _ in range(5):
		print_style(f"Test number : {test_nbr}", color = 'yellow')
	print("\n"*2)

	### df_tail and df_head must not be different of None simultaneously:
	### the can be None simultaneously or
	### if one of them is None the other shall not
	assert df_head is None or df_tail is None, '"df_head" and "df_tail" must not be != None simultaneously !'
	assert df_head is None or slice_df_index is None, '"df_head" and "slice_df_index" must not be != None simultaneously !'
	assert df_tail is None or slice_df_index is None, '"df_tail" and "slice_df_index" must not be != None simultaneously !'
	if slice_df_index is not None:
		assert isinstance(slice_df_index, tuple), '"slice_df_index" must be a tuple.'

	# ### INITIALIZE FIREBASE STORAGE:
	# ### ____________________________
	# firebase_storage = FirebaseStorage(firebase_config = firebase_config)

	### MANAGE WARNINGS:
	### ________________
	manage_wargins()

	### SET VARIABLES ACCORDING TO ENVIRONMENT:
	### _______________________________________

	if check_environment() == 'Colab':
		### READ CSV FILE (DATASET) USING URL:
		url_dataset = dataset_urls[dataset_key]
		assert "github" in url_dataset or "firebasestorage" in url_dataset, "CSV file must be stored on Github or Firebase Storage."

		if "firebasestorage" in url_dataset:
			df = pd.read_csv(url_dataset)
		elif "github" in url_dataset:
			url_dataset += "?raw=true"
			df = pd.read_csv(url_dataset)

		informative_color = 'cyan'
		alert_color = 'red'
		good_color = 'green'
		bold = True

	elif check_environment() == 'Local_PC':
		### READ CSV FILE (DATASET) LOCALLY:
		df = pd.read_csv(r'C:\Users\LENOVO\AppData\Local\Programs\Python\Python37-32\Lib\site-packages\projects\trading_and_ai\data\recent_data\EURUSD-2023-12-13_2024-01-23.csv')
		

		# ######## df = pd.DataFrame({"date_from":list(range(600, 700)),
		# 					"date_to":list(range(600, 700)),
		# 					"open":list(range(100)),
		# 					"high":list(range(200, 300)),
		# 					"low":list(range(300, 400)),
		# 					"close":list(range(500, 600)),
		# 					})



		informative_color = None
		alert_color = None
		good_color = None
		bold = False

	### Print df original:
	print_style("Df original :", color = informative_color)
	print_style("_____________", color = informative_color)
	try:
		print(df[['date_from', "date_to", "close"]])
	except:
		print(df)
	print("\ndf original columns :", df.columns.tolist(),"\n")
	
	### FRAGMENT LENGTH OF DF ACCORDING TO CONDITIONS AND RESET ITS INDEX:
	df = df[['open', 'high', 'low', 'close']]
	if df_head is not None:
		df = df.head(df_head)
	if df_tail is not None:
		df = df.tail(df_tail)
	if slice_df_index is not None:
		df = df.iloc[slice_df_index[0]: slice_df_index[1], :]
	df.reset_index(inplace = True, drop = True)

	### ADD TARGET:
	###____________

	df = add_target(df = df,
				close_column_name = close_column_name, 
				target_type = target_type,
				target_shift = target_shift,
				ratio_true_trend = ratio_true_trend)

	### DETECT THE TYPE OF SUB-CONCEPT:
	###________________________________

	if isinstance(sub_concept_n_size, tuple):
		assert isinstance(sub_concept_n_size[0], str), 'The first value of "sub_concept_n_size" must be a string.'
		assert sub_concept_n_size[0] == 'first' or sub_concept_n_size[0] == 'second', 'when "sub_concept_n_size" is a tuple, its first value must be equals to "first" or "second".'

		assert isinstance(sub_concept_n_size[1], float), 'The second value of "sub_concept_n_size" must be a float.'
		assert 0.0 < sub_concept_n_size[1] < 1.0, 'when "sub_concept_n_size" is a tuple, its second value must be into the interval ]0.0, 1.0['

		sub_concept = sub_concept_n_size[0]
		size_get_seas = sub_concept_n_size[1]
		len_get_seas = int(df.shape[0]*size_get_seas)

		# print("size_get_seas :", size_get_seas)
		# print("len_get_seas  :", len_get_seas)

	elif isinstance(sub_concept_n_size, str):
		assert sub_concept_n_size == 'third', 'when "sub_concept_n_size" is not a tuple, it must be equals to "third".'
		sub_concept = sub_concept_n_size
		size_get_seas = train_size

	else:
		assert isinstance(sub_concept_n_size, tuple) or isinstance(sub_concept_n_size, str), '"sub_concept_n_size" must be either a tuple or a string.'

	if sub_concept == 'second':
		assert size_get_seas < train_size, 'when sub_concept == "second", "size_get_seas" must be < to "train_size".'
	
	### HANDLE COLUMNS AND SPLIT TRAIN AND TEST:
	###_________________________________________
	
	### 1/3. ADD SEASONALITY COLUMNS:
	###______________________________
	
	for freq in freqs_seasonal:
		df = add_seas_column_to_df(df = df, 
							size_get_seas = size_get_seas, 
							freq = freq, 
							close_col_name = close_column_name)

		# print(df)

	### 2/3. SPLIT TRAIN AND TEST:
	###___________________________

	if sub_concept == 'first':
		### DELETE THE PART OF DF WHICH CONCERNS THE GETTING SEASONALITY:
		df = df.tail(df.shape[0] - len_get_seas)
		df.reset_index(inplace = True, drop = True)
		# print(df)

	### SPLIT TRAIN AND TEST:
	len_train = int(train_size*df.shape[0])
	df_train = df.head(len_train)
	df_test = df.tail(df.shape[0] - len_train)
	df_train.reset_index(inplace = True, drop = True)
	df_test.reset_index(inplace = True, drop = True)

	# print("original df_test")
	# print(df_test)

	# print(df_train)
	# print("df_train columns :", df_train.columns.tolist())
	# print("\n")
	# print(df_test)
	# print("df_test columns :", df_test.columns.tolist())

	### 3/3. ADD WAVELETS COLUMNS AND STATIONNARIZED CLOSE:
	###____________________________________________________

	### TRAIN DATA:
	###____________

	df_train_wavelets_cols = add_wavelets_columns(close_column = df_train[close_column_name])
	df_train['dwt_cA'] = df_train_wavelets_cols['dwt_cA']
	df_train['dwt_cD'] = df_train_wavelets_cols['dwt_cD']
	df_train['dwt_cA_stationnarized'] = df_train_wavelets_cols['dwt_cA_stationnarized']
	df_train['stationnarized_close'] = stationnarize_close_column(df = df_train, 
												close_column_name = close_column_name)

	### TEST DATA:
	###___________

	df_test_wavelets_cols = add_wavelets_columns(close_column = df_test[close_column_name])
	df_test['dwt_cA'] = df_test_wavelets_cols['dwt_cA']
	df_test['dwt_cD'] = df_test_wavelets_cols['dwt_cD']
	df_test['dwt_cA_stationnarized'] = df_test_wavelets_cols['dwt_cA_stationnarized']
	df_test['stationnarized_close'] = stationnarize_close_column(df = df_test,
												close_column_name = close_column_name)

	print(df_train)
	print()
	print_style(f"df_train columns : {df_train.columns.tolist()}", 
				color = informative_color)
	print("\n")
	print(df_test)
	print()
	print_style(f"df_test columns : {df_test.columns.tolist()}", 
				color = informative_color)
	print("\n")

	### SCALE TRAIN AND TEST DATA:
	###___________________________

	### TRAIN DATA:
	###____________
	df_train_scaled, df_train_scaled_scaler = data_scaler(df = df_train)

	### TEST DATA:
	###___________
	df_test_scaled, df_test_scaled_scaler = data_scaler(df = df_test)

	# print(df_train_scaled)
	# print("\n")
	# print(df_test_scaled)
	# print("\n\n")

	### SPLIT X AND Y:
	###_______________

	### SET DATA COLUMNS NAMES:
	###________________________
	data_cols_names_seasonality = [f'seasonality_{d}' for d in freqs_seasonal]
	data_cols_names = ['close', 
						'stationnarized_close',
						# 'soft_0.5', 
						# 'less_0.5', 
						# 'soft_0.5_stationnarized',
						'dwt_cA', 
						'dwt_cD', 
						'dwt_cA_stationnarized']
	data_cols_names += data_cols_names_seasonality

	### TRAIN:
	###_______
	x_y_train = native_x_y_spliter(df = df_train_scaled, 
								data_cols_names = data_cols_names, 
								target_col_name = target_col_name, 
								look_back = look_back)

	X_train = x_y_train['dataX']
	y_train = x_y_train['dataY']

	x_y_test = native_x_y_spliter(df = df_test_scaled, 
								data_cols_names = data_cols_names, 
								target_col_name = target_col_name, 
								look_back = look_back)

	X_test = x_y_test['dataX']
	# y_test = x_y_test['dataY']

	print("\n")
	print_style(f"Shape of X_train :{X_train.shape}", color = informative_color)
	print_style(f"Shape of y_train :{y_train.shape}", color = informative_color)
	print_style(f"Shape of X_test  :{X_test.shape}", color = informative_color)
	print("\n")

	### DOWNLOAD LANDMARKS ABOUT FITTING:
	###__________________________________
	filenames_landmarks = [
				f'prev_epochs_test_nbr_{test_nbr}.txt',
				f'model_test_nbr_{test_nbr}.h5',
				f'history_test_nbr_{test_nbr}.pkl',
				f'log_fitting_test_nbr_{test_nbr}.txt']

	print("\n")
	for filename_landmark in filenames_landmarks:
		if not check_file_exists(gdrive_folder_path + filename_landmark):
			result_downloading = direct_download_file(
				cloud_file_path_name = landmarks_cloud_path + filename_landmark, 
				local_file_path_name = gdrive_folder_path + filename_landmark)

			if result_downloading:
				print_style(f"Successfully downloaded {filename_landmark}", 
					color = good_color, bold = bold)
			else:
				print_style(f"File not downloaded: {filename_landmark}",
					color = alert_color, bold = bold)
	print("\n")

	### FIT THE MODEL:
	###_______________
	fit_n_save_model(X_train = X_train, 
					y_train = y_train,
					epochs = epochs, 
					test_nbr = test_nbr,
					verbose_epoch_in_callback = verbose_epoch_in_callback,
					gdrive_folder_path = gdrive_folder_path,
					prevent_wait = prevent_wait,
					verbose_model_checkpoint = verbose_model_checkpoint,
					batch_size = batch_size_fit,
					validation_split = validation_split_fit,
					shuffle = shuffle_fit,
					verbose_fit = verbose_fit,
					)

	### USE MODEL TO MAKE PREDICTIONS:
	###_______________________________
	model = load_the_model(gdrive_folder_path = gdrive_folder_path, 
									test_nbr = test_nbr)

	y_pred = model.predict(X_test)

	### COMPILE THE PREDICTION RESULTS:
	###________________________________
	df_x_test = pd.DataFrame([item[-1] for item in X_test], columns = data_cols_names)
	df_x_test = df_x_test[[close_column_name]]
	df_results_scaled = pd.DataFrame({'y_pred':[item[0] for item in y_pred]})
	df_results = pd.concat([df_x_test, df_results_scaled], axis = 1)
	df_results_filename = f"df_results_test_nbr_{test_nbr}.csv"
	df_results.to_csv(gdrive_folder_path + df_results_filename)

	### GET BALANCE SHEET:
	###___________________
	balance_sheet_filename = f"Balance_sheet_test_nbr_{test_nbr}.txt"
	get_balance_sheet(df_results = df_results, 
					close_column_name = close_column_name, 
					target_shift = target_shift, 
					y_pred_col_name = "y_pred", 
					verbose = verbose_eval, 
					test_nbr = test_nbr, 
					save_to = gdrive_folder_path + balance_sheet_filename)

	### UPLOAD RESULT FILES:
	###_____________________
	result_filenames = [
		balance_sheet_filename,
		df_results_filename,
		f'prev_epochs_test_nbr_{test_nbr}.txt',
		f'model_test_nbr_{test_nbr}.h5',
		f'history_test_nbr_{test_nbr}.pkl',
		f'log_fitting_test_nbr_{test_nbr}.txt',
	]

	print("\n")
	for result_filename in result_filenames:
		result_upload = direct_upload_file(
			local_file_path_name = gdrive_folder_path + result_filename, 
			cloud_file_path_name = results_cloud_path + result_filename,
			)

		if result_upload:
			print_style(f"Successfully upload result file: {result_filename}:\n\t{result_upload}",
				color = good_color, bold = bold)
		else:
			print_style(f"\nResult file {result_filename} wasn't uploaded !!!\n",
				color = alert_color, bold = bold)

	print("\n")

	### SIGNAL THE ENDING OF CODE:
	###___________________________
	print_style("\n\tFinished", color = good_color, bold = bold)










