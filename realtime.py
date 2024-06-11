

"""
Nous nous sommes basé sur le code du fichier: simulate_as_realtime_shortcut_way.py
"""

###################################
########## SECTION MAIN VARIABLES:


results_cloud_path = "drinbd-n-repeat-seas/results/"
landmarks_cloud_path = "drinbd-n-repeat-seas/landmarks/"
# modules_cloud_path = "drinbd-n-repeat-seas/modules/"

gdrive_folder_path = '/content/drive/MyDrive/'

common_path = "https://github.com/GilbertAK/eurusd_data/blob/main/"
common_path_firebase_strg = "https://firebasestorage.googleapis.com/v0/b/saving-data-2ee4b.appspot.com/o/datasets_csv%2F"

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
	"dataset_drinbd_n_repeat_seas_1_EURUSD":"https://firebasestorage.googleapis.com/v0/b/saving-data-2ee4b.appspot.com/o/csv_files%2FEURUSD-1.0%20Min--2024-4-18%200-1-0.csv?alt=media",
	"dataset_drinbd_n_repeat_seas_1_added_1_EURUSD":"https://firebasestorage.googleapis.com/v0/b/saving-data-2ee4b.appspot.com/o/datasets_csv%2Fdataset_drinbd_n_repeat_seas_1_added_1.csv?alt=media",
	"dataset_drinbd_n_repeat_seas_1_added_2_EURUSD":"https://firebasestorage.googleapis.com/v0/b/saving-data-2ee4b.appspot.com/o/datasets_csv%2Fdataset_drinbd_n_repeat_seas_1_added_2.csv?alt=media",
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


def str_datetime_2_datetime_instance(str_datetime):
	date = str_datetime.split(" ")[0]
	time = str_datetime.split(" ")[1]
	year = int(date.split("-")[0])
	month = int(date.split("-")[1])
	day = int(date.split("-")[2])
	hour = int(time.split(":")[0])
	minute = int(time.split(":")[1])
	second = int(time.split(":")[2])
	return datetime.datetime(year, month, day, hour, minute, second)


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

def manage_warnings():
	pd.options.mode.chained_assignment = None
	warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
	warnings.simplefilter(action='ignore', category=DeprecationWarning)
	warnings.simplefilter(action='ignore', category=RuntimeWarning)
	warnings.simplefilter(action='ignore')

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
########## SECTION DATASCIENCE:

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
		cA_duplicated = cA_duplicated[:-1]
		cD_duplicated = cD_duplicated[:-1]

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


# def fit_n_save_model(X_train, y_train,
# 					epochs, test_nbr,
# 					verbose_epoch_in_callback,
# 					gdrive_folder_path,
# 					prevent_wait,
# 					verbose_model_checkpoint = False,
# 					batch_size = 32,
# 					validation_split = 0.3,
# 					shuffle = False,
# 					verbose_fit = True,):

# 	################
# 	### FILES PATHS:
# 	################

# 	if not gdrive_folder_path.endswith("/"):
# 		gdrive_folder_path += "/"
	
# 	save_prev_epochs_filepath = gdrive_folder_path + f'prev_epochs_test_nbr_{test_nbr}.txt'
# 	model_filepath = gdrive_folder_path + "model_test_nbr_" + f'{test_nbr}.h5'
# 	history_filepath = gdrive_folder_path + f'history_test_nbr_{test_nbr}.pkl'
# 	log_fitting_filepath = gdrive_folder_path + f'log_fitting_test_nbr_{test_nbr}.txt'

# 	##########################
# 	## LOAD OR CREATE HISTORY:
# 	##########################

# 	print(f"Checking presence of: {history_filepath.split('/')[-1]}")
# 	history_filepath_exists = False
# 	for _ in range(5):
# 		if check_file_exists(filepath = history_filepath):
# 			history_filepath_exists = True
# 			break
# 		else:
# 			time.sleep(1)

# 	if history_filepath_exists:
# 		loss_n_val_loss = load_pickle(filepath = history_filepath)
# 		print_style(text = f"\nOkay {history_filepath.split('/')[-1]} found !!!\n", 
# 						color = 'green', 
# 						bold = True,
# 						underline = False)
# 		# time.sleep(3)
# 	else:
# 		for _ in range(10):
# 			print_style(text = "Saving history will start from zero !!!", 
# 							color = "red", 
# 							bold = True, 
# 							underline = False)
# 		time.sleep(prevent_wait[0])
# 		loss_n_val_loss = {'loss':[], 'val_loss':[]}

# 	#######################################################
# 	### LOAD OR CREATE MODEL AND LAST EPOCH LANDMARK FILE:
# 	#######################################################

# 	print(f"Checking presence of: {model_filepath.split('/')[-1]}")
# 	model_filepath_exists = False
# 	for _ in range(5):
# 		if check_file_exists(filepath = model_filepath):
# 			model_filepath_exists = True
# 			break
# 		else:
# 			time.sleep(1)

# 	print(f"Checking presence of: {save_prev_epochs_filepath.split('/')[-1]}")
# 	save_prev_epochs_filepath_exists = False
# 	for _ in range(5):
# 		if check_file_exists(filepath = save_prev_epochs_filepath):
# 			save_prev_epochs_filepath_exists = True
# 			break
# 		else:
# 			time.sleep(1)

# 	if model_filepath_exists and save_prev_epochs_filepath_exists:

# 		### Load the model
# 		model = load_model(model_filepath)
# 		with open(save_prev_epochs_filepath, "r", encoding = "utf-8") as f:
# 			last_epochs = f.readlines()
# 		last_epochs = [item.strip() for item in last_epochs]
# 		all_previous_epochs_nbr = len(last_epochs)
# 		epochs -= all_previous_epochs_nbr
# 		print_style(text = f"\nOkay {model_filepath.split('/')[-1]} and {save_prev_epochs_filepath.split('/')[-1]} found !!!\n",
# 					color = "green", 
# 					bold = True, 
# 					underline = False)
# 		# time.sleep(3)
# 	else:
# 		for _ in range(10):
# 			print_style(text = "Fitting model will start from zero !!!", 
# 						color = "red", 
# 						bold = True, 
# 						underline = False)

# 		time.sleep(prevent_wait[1])

# 		### Creating the model:
# 		dropout = 0.2
# 		model = keras.Sequential()
# 		model.add(keras.layers.Bidirectional(
# 			keras.layers.LSTM(units=128, input_shape=(X_train.shape[1], X_train.shape[2]))))
# 		model.add(keras.layers.Dropout(rate=dropout))
# 		model.add(keras.layers.Dense(units=1))
# 		model.compile(loss='mean_squared_error', optimizer='adam')

# 	###################
# 	### FIT THE MODEL:
# 	###################

# 	each_epoch_callback = EachEpochCallback(verbose_epoch_in_callback = verbose_epoch_in_callback, 
# 										epochs = epochs, history_filepath = history_filepath, 
# 										save_prev_epochs_filepath = save_prev_epochs_filepath, 
# 										loss_n_val_loss = loss_n_val_loss,
# 										log_fitting_filepath = log_fitting_filepath)

# 	model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath = model_filepath,
# 																monitor = 'loss',
# 																mode = 'min',
# 																save_best_only = True,
# 																verbose = verbose_model_checkpoint)

# 	history = model.fit(
# 				X_train, y_train,
# 				epochs = epochs,
# 				batch_size = batch_size,
# 				validation_split = validation_split,
# 				shuffle = shuffle,
# 				verbose = verbose_fit,
# 				callbacks = [model_checkpoint_callback, each_epoch_callback],
# 				)

def load_the_model(model_eurusd_concerned, gdrive_folder_path, test_nbr):
	if not gdrive_folder_path.endswith("/"):
		gdrive_folder_path += "/"
	model_filepath = gdrive_folder_path + "model_test_nbr_" + f'{test_nbr}-{model_eurusd_concerned}.h5'
	return load_model(model_filepath)


###################################
########## SECTION EVALUATE RESULTS:


import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)

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

def get_balance_sheet(df_results, 
	close_column_name, target_shift, y_pred_col_name, verbose, test_nbr, save_to = None):

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


def get_balance_sheet_v2(df_results, 
						close_column_name, 
						target_shift, 
						y_pred_col_name):

	"""
	Dans cette v2, la computation est la même que dans la version précédente sauf:
		- pour le test nbr qui ne sera plus parmi les params
		- pour le save_to: il n'y aura pas ici le possibilté d'enregistrer le résultat
		- pour le verbose: il n'y a pas ici le printing de résultat
		- l'accuracy et le ratio_trades seront returnés
		- le df_results sera fragmentés aux 58_986 primières lignes (59_000 - 14)

	"""
	df = df_results.copy()
	# df = df.head(58_986)
	# df.reset_index(inplace = True, drop = True)
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
		accuracy = round(accuracy*100, 2)

	except ZeroDivisionError:
		accuracy = 'No trade taken'

	try:
		ratio_trades = total_trades/len_df
		ratio_trades = round(ratio_trades*100, 2)
	except ZeroDivisionError:
		ratio_trades = 'No trade taken'

	return {"accuracy":accuracy,
			"ratio_trades":ratio_trades,
			"total_trades":total_trades}


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

	# len_df_get_seas = int(df.shape[0]*size_get_seas)
	len_df_get_seas = 19_999
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
########## SECTION IQ OPTION:


import os
def installer_modules_n_clone_repo():
	print("Cloning IQ Option Repo. ...")
	os.system("git clone https://github.com/GilbertAK/iqoptionapi.git")
	print("Installing pylint ...")
	os.system("pip install pylint")
	# print("Installing requests ...")
	# os.system("pip install requests")
	print("Installing websocket-client ...")
	os.system("pip install websocket-client==0.56.0")

if check_environment() == 'Colab':
	installer_modules_n_clone_repo()

from iqoptionapi.stable_api import IQ_Option
import time
import pandas as pd
import datetime
import math

def connect_2_iq_option_account(iq_email, 
								iq_password, 
								account_type = "PRACTICE"):
	account = IQ_Option(iq_email, iq_password)
	print('\n')
	print('           Connexion à IQ Option ...')
	print('           Compte :', iq_email)
	account.connect()
	account.change_balance(account_type) # ### or REAL
	print('           Connecté à IQ Option.')
	return account


def timestamp_converter(x):
	heure = datetime.datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S")
	return heure


def get_multi(start_str_datetime):
	# start_str_datetime == for example: "2023-12-13 04:44:00"
	date_ = start_str_datetime.split(" ")[0]
	time_ = start_str_datetime.split(" ")[1]
	#############################
	year = int(date_.split("-")[0])
	month = int(date_.split("-")[1])
	day = int(date_.split("-")[2])
	hour = int(time_.split(":")[0])
	minute = int(time_.split(":")[1])
	second = int(time_.split(":")[2])
	#############################
	start_time = datetime.datetime(year, month, day, hour, minute, second).timestamp()
	delta_time = time.time() - start_time
	delta_days = delta_time/86400
	delta_weeks = delta_days/7
	week_ends = int(delta_weeks)
	days_to_exclude = week_ends*2 - 1
	stayed_days = delta_days - days_to_exclude
	stayed_minutes = math.ceil(stayed_days * 1440)
	multi = math.ceil(stayed_minutes/1_000)
	return multi


def get_nbr_candles(start_str_datetime):
	# start_str_datetime == for example: "2023-09-22 01:15:00"
	date_ = start_str_datetime.split(" ")[0]
	time_ = start_str_datetime.split(" ")[1]
	#############################
	year = int(date_.split("-")[0])
	month = int(date_.split("-")[1])
	day = int(date_.split("-")[2])
	hour = int(time_.split(":")[0])
	minute = int(time_.split(":")[1])
	second = int(time_.split(":")[2])
	#############################
	start_time = datetime.datetime(year, month, day, hour, minute, second).timestamp()
	delta_time = time.time() - start_time
	nbr_candles = math.ceil(delta_time/60)
	return nbr_candles


def get_concerned_candles(df, start_str_datetime):
	df = df.copy()
	df.reset_index(inplace = True, drop = True)
	from_index = list(df[df['date_from'] == start_str_datetime].index)
	assert len(from_index) == 1, "len(from_index) == 1 in the function def get_concerned_candles(...)"
	from_index = from_index[0]
	take_only = df.shape[0] - from_index
	new_df = df.tail(take_only)
	new_df.reset_index(inplace = True, drop = True)
	assert new_df.shape[0] < df.shape[0], 'The length of "new_df" should be < to the length of "df" !!!'
	return new_df


def get_big_data_candles(account, pair, timeframe, start_str_datetime, verbose = False):
	timestamp = time.time()
	total = []

	if verbose:
		print("Downloading candles ...")
	multi = get_multi(start_str_datetime = start_str_datetime)
	for i in range(multi):
		start_time = time.time()
		X = account.get_candles(pair, timeframe, 1000, timestamp)
		total = X+total
		timestamp = int(X[0]['from'])-1

	list_date_from = []
	list_date_to = []
	list_open = []
	list_high = []
	list_low = []
	list_close = []
	list_volume = []
	for idx_velas, velas in enumerate(total):
		start_time = time.time()
		list_date_from.append(str(timestamp_converter(velas['from'])))
		list_date_to.append(str(timestamp_converter(velas['to'])))
		list_open.append(velas['open'])
		list_high.append(velas['max'])
		list_low.append(velas['min'])
		list_close.append(velas['close'])
		list_volume.append(velas['volume'])

	df = pd.DataFrame({'date_from':list_date_from, 
		'date_to':list_date_to, 
		'open':list_open, 
		'high':list_high, 
		'low':list_low, 
		'close':list_close, 
		'volume':list_volume})
	new_df = get_concerned_candles(df = df, start_str_datetime = start_str_datetime)
	assert new_df.head(1)['date_from'].tolist()[0] == start_str_datetime, "new_df.head(1)['date_from'].tolist()[0] == start_str_datetime"
	return new_df


def get_small_data_candles(account, pair, timeframe, nbr_candles):
	total = []
	timestamp = time.time()
	total = account.get_candles(pair, timeframe, nbr_candles, timestamp)

	list_date_from, list_date_to, list_open, list_high, list_low, list_close, list_volume = [], [], [], [], [], [], []
	for idx_velas, velas in enumerate(total):
		start_time = time.time()
		list_date_from.append(str(timestamp_converter(velas['from'])))
		list_date_to.append(str(timestamp_converter(velas['to'])))
		list_open.append(velas['open'])
		list_high.append(velas['max'])
		list_low.append(velas['min'])
		list_close.append(velas['close'])
		list_volume.append(velas['volume'])

	df = pd.DataFrame({'date_from':list_date_from, 
		'date_to':list_date_to, 
		'open':list_open, 
		'high':list_high, 
		'low':list_low, 
		'close':list_close, 
		'volume':list_volume})
	return df


class Digital:
	def __init__(self, account):
		self.account = account
	
	def pass_order(self, amount, pair, order_type, expiration):
		check, _id = self.account.buy_digital_spot(pair, amount, order_type, expiration)

		"""
		Pair : USDJPY
		Check : True
		Id    : 18965320252

		Pair : CADCHF
		Check : False
		Id    : {'message': 'quotesApplication.ConsumeQuoteByTimeDeprecated: invalid instrument: doCADCHF202405140851PT1MPSPT'}

		Pair : CADCHF
		Check : False
		Id    : {'message': 'quotesApplication.ConsumeQuoteByTimeDeprecated: invalid instrument: doCADCHF202405140852PT1MPSPT'}
		Type of id - message: <class 'str'>
		"""
		return check, _id

	def get_history(self, nbr_data):
		data = self.account.get_position_history_v2(instrument_type = "digital-option",
								limit = nbr_data, offset = 0, start = 0, end = 0)
		history_node = data[1]['positions']
		return {"all_data":data, "history_node":history_node}

	def trade_closed(self, _id):
		history_node = self.get_history(nbr_data = 30)['history_node']
		all_ids_closed_trades = [item['raw_event']['order_ids'][0] for item in history_node]

		if _id in all_ids_closed_trades:
			return True
		else:
			return False

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

pairs_list = ["EURUSD", "EURCHF", "EURAUD", "GBPCHF", "GBPAUD", "GBPCAD", "USDCHF"]
model_reference_cloud_path = "drinbd-n-repeat-seas/models-de-reference/"
# models_filenames = {
# 				"EURUSD":'model_test_nbr_drinbd_n_repeat_seas_53.4.h5', 
# 				"EURCHF":'model_test_nbr_drinbd_n_repeat_seas_55.EURCHF.h5', 
# 				"GBPCHF":'model_test_nbr_drinbd_n_repeat_seas_55.GBPCHF.h5',
# 				"EURAUD":'model_test_nbr_drinbd_n_repeat_seas_56_euraud.1.h5', 
# 				"GBPAUD":'model_test_nbr_drinbd_n_repeat_seas_56_gbpaud.1.h5', 
# 				"GBPCAD":'model_test_nbr_drinbd_n_repeat_seas_56_gbpcad.1.h5', 
# 				"USDCHF":'model_test_nbr_drinbd_n_repeat_seas_56_usdchf.1.h5'
# 				}


models_filenames = {
				"EURUSD-1":'model_test_nbr_drinbd_n_repeat_seas_53.1.h5', 
				"EURUSD-2":'model_test_nbr_drinbd_n_repeat_seas_53.2.h5', 
				"EURUSD-3":'model_test_nbr_drinbd_n_repeat_seas_53.3.h5', 
				"EURUSD-4":'model_test_nbr_drinbd_n_repeat_seas_53.4.h5', 
				"EURUSD-5":'model_test_nbr_drinbd_n_repeat_seas_53.5.h5', 
				}

def download_model(model_eurusd_concerned, test_nbr):	
	assert model_eurusd_concerned.upper() in list(models_filenames.keys()), \
	f'The model_eurusd_concerned your enter "{model_eurusd_concerned}" \
	must be within the list:{list(models_filenames.keys())}'

	model_cloud_filepath = model_reference_cloud_path + models_filenames[model_eurusd_concerned.upper()]
	# print(model_cloud_filepath)
	model_local_filepath = gdrive_folder_path + "model_test_nbr_" + f'{test_nbr}-{model_eurusd_concerned}.h5'
	# print(model_local_filepath)

	result_download = direct_download_file(
		cloud_file_path_name = model_cloud_filepath,
		local_file_path_name = model_local_filepath, 
		)
	print("\n")
	if result_download:
		for _ in range(5):
			print_style(f'Model successfully downloaded.', 
				color = 'green', bold = True)
	else:
		for _ in range(20):
			print_style(f'Attention, Model Not downloaded !!!', 
				color = 'red', bold = True)
	print("\n")


def check_gpu_connected():
	try:
		if int(os.environ["COLAB_GPU"]) > 0:
			# print("a GPU is connected.")
			return True
	except:
		# print("GPU is not connected !")
		return False


def get_signal(y_pred):
	if y_pred == 0.0:
		return 'put'
	elif y_pred == 0.5:
		return 'neutral'
	elif y_pred == 1.0:
		return 'call'


def manage(list_model_names,
		close_column_name,
		target_col_name,
		freqs_seasonal,
		target_shift,
		target_type,
		ratio_true_trend,
		look_back,
		epochs,
		verbose_epoch_in_callback,
		verbose_model_checkpoint,
		batch_size_fit,
		validation_split_fit,
		shuffle_fit,
		verbose_fit,
		verbose_eval,
		prevent_wait,
		iq_email,
		iq_password,
		risk_factor,
		minimal_balance_tradable,
		test_nbr = "drinbd_n_repeat_seas_realtime",
		account_type = "PRACTICE",
		):
	
	### Asserts:
	###_________
	assert 0.0 < risk_factor < 1.0, "0.0 < risk_factor < 1.0"

	### WAIT GOOGLE DRIVE CONNECTED:
	###_____________________________
	wait_gdrive_connected()

	### CHECK IF GPU IS CONNECTED:
	###___________________________
	if check_environment() == 'Colab':
		gpu_connected = check_gpu_connected()
		if not gpu_connected:
			print_style("\nPLEASE CONNECT THE GPU !!!", 
							color = 'red', bold = True)
			print_style("The continuation of code will run only when the GPU will be connected !!!\n", 
							color = 'cyan')
			while True:
				time.sleep(1)
		else:
			print_style("\nGPU is connected !!!\n", 
							color = 'green', bold = True)

	### PRINT TEST NBR:
	###________________

	print("\n"*2)
	for _ in range(5):
		print_style(f"Test number : {test_nbr}", color = 'yellow')
	print("\n"*2)

	# ### df_tail and df_head must not be different of None simultaneously:
	# ### the can be None simultaneously or
	# ### if one of them is None the other shall not
	# assert df_head is None or df_tail is None, '"df_head" and "df_tail" must not be != None simultaneously !'
	# assert df_head is None or slice_df_index is None, '"df_head" and "slice_df_index" must not be != None simultaneously !'
	# assert df_tail is None or slice_df_index is None, '"df_tail" and "slice_df_index" must not be != None simultaneously !'
	# if slice_df_index is not None:
	# 	assert isinstance(slice_df_index, tuple), '"slice_df_index" must be a tuple.'

	### MANAGE WARNINGS:
	### ________________
	manage_warnings()

	### SET VARIABLES ACCORDING TO ENVIRONMENT:
	### _______________________________________

	if check_environment() == 'Colab':
		account = connect_2_iq_option_account(iq_email = iq_email, 
									iq_password = iq_password, 
									account_type = account_type)

		# 2023-12-13 04:44:00
		start_str_datetime = "2023-12-13 04:44:00"

		informative_color = 'cyan'
		alert_color = 'red'
		good_color = 'green'
		bold = True

	elif check_environment() == 'Local_PC':
		### READ CSV FILE (DATASET) LOCALLY:
		df = pd.read_csv(r'C:\Users\LENOVO\AppData\Local\Programs\Python\Python37-32\Lib\site-packages\projects\trading_and_ai\data\recent_data\EURUSD-2023-12-13_2024-01-23.csv')
		
		informative_color = None
		alert_color = None
		good_color = None
		bold = False

	long_df = get_big_data_candles(account = account, 
						pair = "EURUSD", 
						timeframe = 60, 
						start_str_datetime = start_str_datetime, 
						verbose = True)

	assert long_df.head(1)['date_from'].tolist()[0] == start_str_datetime, "long_df.head(1)['date_from'].tolist()[0] == start_str_datetime"
	
	# ### FRAGMENT LENGTH OF DF ACCORDING TO CONDITIONS AND RESET ITS INDEX:
	# long_df = long_df[['date_from', 'date_to', 'open', 'high', 'low', 'close']]
	# if df_head is not None:
	# 	long_df = long_df.head(df_head)
	# if df_tail is not None:
	# 	long_df = long_df.tail(df_tail)
	# if slice_df_index is not None:
	# 	long_df = long_df.iloc[slice_df_index[0]: slice_df_index[1], :]

	### We need just some columns:
	long_df = long_df[['date_from', 'date_to', 'open', 'high', 'low', 'close']]	
	long_df.reset_index(inplace = True, drop = True)

	# print("\n\n")
	print("\n\nLong df :\n")
	print(long_df[['date_from', 'date_to', 'close']], 
	"\n________________________________\n")

	### Instanciate the Digital class:
	###_______________________________
	digital = Digital(account = account)

	### Long while loop:
	###_________________

	first_iteration = True
	while True:
		last_datetime_from_long_df = long_df.tail(1)['date_from'].tolist()[0]
		nbr_candles = get_nbr_candles(
						start_str_datetime = last_datetime_from_long_df)
		small_df = get_small_data_candles(
						account = account,
						pair = 'EURUSD',
						timeframe = 60,
						nbr_candles = nbr_candles)
		last_datetime_from = small_df.tail(1)['date_from'].tolist()[0]
		datetime_last_row = str_datetime_2_datetime_instance(
								str_datetime = last_datetime_from)
		right_timing = False
		datetime_now = datetime.datetime.now()
		if datetime_last_row.year == datetime_now.year and \
			datetime_last_row.month == datetime_now.month and \
			datetime_last_row.day == datetime_now.day and \
			datetime_last_row.hour == datetime_now.hour and \
			datetime_last_row.minute == datetime_now.minute:
			right_timing = True

		if right_timing and (0.1 <= datetime.datetime.now().second <= 7):
			start_time_iter = time.time()

			### The last row of long_df must be identic to the first row of small_df:
			assert long_df.tail(1)['date_from'].tolist()[0] == small_df.head(1)['date_from'].tolist()[0], \
			"long_df.tail(1)['date_from'].tolist()[0] == small_df.head(1)['date_from'].tolist()[0]"
			### We need just some columns:
			small_df = small_df[['date_from', 'date_to', 'open', 'high', 'low', 'close']]
			### Before to concatenate long_df and small_df, we must delete the first row of 
			### small_df because it is the same as the last row of the long_df.
			small_df = small_df.tail(small_df.shape[0] - 1)
			small_df.reset_index(inplace = True, drop = True)

			df = pd.concat([long_df, small_df], axis = 0)
			df.reset_index(inplace = True, drop = True)
			if first_iteration:
				df_first_iter = df.copy()
				df_first_iter = df_first_iter.head(129_000)
				df_first_iter.reset_index(inplace = True, drop = True)
				print("This is the first iteration ...")
				print("df_first_iter :")
				print(df_first_iter[["date_from", "date_to", "open", "close"]])

			assert df.columns.tolist() == long_df.columns.tolist(), "df.columns.tolist() == long_df.columns.tolist()"
			assert df.columns.tolist() == small_df.columns.tolist(), "df.columns.tolist() == small_df.columns.tolist()"
			assert df.shape[0] == long_df.shape[0] + small_df.shape[0], "df.shape[0] == long_df.shape[0] + small_df.shape[0]"
			
			### ADD TARGET:
			###____________
			df = df[['open', 'high', 'low', 'close']]

			# """
			# AVANT LE add_target, LA DERNIÈRE LIGNE DE DF EST CELLE DU CLOSE INSTABLE QUE 
			# NOUS CHERCHONS À PRÉDIRE.
			# """

			if first_iteration:
				df_first_iter = add_target(df = df_first_iter,
						close_column_name = close_column_name, 
						target_type = target_type,
						target_shift = target_shift,
						ratio_true_trend = ratio_true_trend)
			
			df['target'] = np.nan
			df = df.head(df.shape[0] - target_shift)

			# """
			# NOUS VENONS DE FAIRE UN add_target AU DF; DONC LA LIGNE DU CLOSE INSTABLE 
			# VIENT D'ÊTRE SUPPRIMÉE.
			# """

			### DETECT THE TYPE OF SUB-CONCEPT:
			###________________________________

			# if isinstance(sub_concept_n_size, tuple):
			# 	assert isinstance(sub_concept_n_size[0], str), 'The first value of "sub_concept_n_size" must be a string.'
			# 	assert sub_concept_n_size[0] == 'first' or sub_concept_n_size[0] == 'second', 'when "sub_concept_n_size" is a tuple, its first value must be equals to "first" or "second".'

			# 	assert isinstance(sub_concept_n_size[1], float), 'The second value of "sub_concept_n_size" must be a float.'
			# 	assert 0.0 < sub_concept_n_size[1] < 1.0, 'when "sub_concept_n_size" is a tuple, its second value must be into the interval ]0.0, 1.0['

			# 	sub_concept = sub_concept_n_size[0]
			# 	size_get_seas = sub_concept_n_size[1]
			# 	len_get_seas = int(df.shape[0]*size_get_seas)

			# 	# print("size_get_seas :", size_get_seas)
			# 	# print("len_get_seas  :", len_get_seas)

			# elif isinstance(sub_concept_n_size, str):
			# 	assert sub_concept_n_size == 'third', 'when "sub_concept_n_size" is not a tuple, it must be equals to "third".'
			# 	sub_concept = sub_concept_n_size
			# 	size_get_seas = train_size

			# else:
			# 	assert isinstance(sub_concept_n_size, tuple) or isinstance(sub_concept_n_size, str), '"sub_concept_n_size" must be either a tuple or a string.'

			# if sub_concept == 'second':
			# 	assert size_get_seas < train_size, 'when sub_concept == "second", "size_get_seas" must be < to "train_size".'
				
			### HANDLE COLUMNS AND SPLIT TRAIN AND TEST:
			###_________________________________________
			
			### 1/3. ADD SEASONALITY COLUMNS:
			###______________________________

			if first_iteration:
				for freq in freqs_seasonal:
					df_first_iter = add_seas_column_to_df(df = df_first_iter, 
										size_get_seas = 0.0, 
										freq = freq, 
										close_col_name = close_column_name)

			for freq in freqs_seasonal:
				df = add_seas_column_to_df(df = df, 
									size_get_seas = 0.0, 
									freq = freq, 
									close_col_name = close_column_name)
			# print(df)
			### 2/3. SPLIT TRAIN AND TEST:
			###___________________________

			# ### if sub_concept == 'first':
			### DELETE THE PART OF DF WHICH CONCERNS THE GETTING SEASONALITY:

			if first_iteration:
				df_first_iter = df_first_iter.tail(df_first_iter.shape[0] - 19_999)
				df_first_iter.reset_index(inplace = True, drop = True)

			df = df.tail(df.shape[0] - 19_999)
			df.reset_index(inplace = True, drop = True)
			# print(df)

			### SPLIT TRAIN AND TEST:
			# len_train = int(train_size*df.shape[0])
			# df_train = df.head(50_000)

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

			if first_iteration:
				df_first_iter_test = df_first_iter.tail(df_first_iter.shape[0] - 50_000)
				df_first_iter_test.reset_index(inplace = True, drop = True)

			df_test = df.tail(df.shape[0] - 50_000)
			df_test_length = df_test.shape[0]
			# df_train.reset_index(inplace = True, drop = True)
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

			# ### TRAIN DATA:
			# ###____________

			# df_train_wavelets_cols = add_wavelets_columns(close_column = df_train[close_column_name])
			# df_train['dwt_cA'] = df_train_wavelets_cols['dwt_cA']
			# df_train['dwt_cD'] = df_train_wavelets_cols['dwt_cD']
			# df_train['dwt_cA_stationnarized'] = df_train_wavelets_cols['dwt_cA_stationnarized']
			# df_train['stationnarized_close'] = stationnarize_close_column(df = df_train, 
			# 											close_column_name = close_column_name)

			### TEST DATA:
			###___________

			if first_iteration:
				df_first_iter_test_wavelets_cols = add_wavelets_columns(
						close_column = df_first_iter_test[close_column_name])
				df_first_iter_test['dwt_cA'] = df_first_iter_test_wavelets_cols['dwt_cA']
				df_first_iter_test['dwt_cD'] = df_first_iter_test_wavelets_cols['dwt_cD']
				df_first_iter_test['dwt_cA_stationnarized'] = df_first_iter_test_wavelets_cols['dwt_cA_stationnarized']
				df_first_iter_test['stationnarized_close'] = stationnarize_close_column(
													df = df_first_iter_test,
													close_column_name = close_column_name)
				date_from_column = df_first_iter_test['date_from'].tolist()
				date_from_column = date_from_column[look_back-1:]
				df_first_iter_test = df_first_iter_test[data_cols_names + ['target']]

			df_test_wavelets_cols = add_wavelets_columns(close_column = df_test[close_column_name])
			df_test['dwt_cA'] = df_test_wavelets_cols['dwt_cA']
			df_test['dwt_cD'] = df_test_wavelets_cols['dwt_cD']
			df_test['dwt_cA_stationnarized'] = df_test_wavelets_cols['dwt_cA_stationnarized']
			df_test['stationnarized_close'] = stationnarize_close_column(df = df_test,
														close_column_name = close_column_name)

			# # print(df_train)
			# # print()
			# # print_style(f"df_train columns : {df_train.columns.tolist()}", 
			# # 			color = informative_color)
			# print("\n")
			# print(df_test)
			# print()
			# print_style(f"df_test columns : {df_test.columns.tolist()}", 
			# 			color = informative_color)
			# print("\n")

			### SCALE TRAIN AND TEST DATA:
			###___________________________

			### TRAIN DATA:
			###____________
			# df_train_scaled, df_train_scaled_scaler = data_scaler(df = df_train)

			### TEST DATA:
			###___________
			if first_iteration:
				df_first_iter_test_scld, _ = data_scaler(df = df_first_iter_test)

			df_test_scaled, df_test_scaled_scaler = data_scaler(df = df_test)
			# print(df_train_scaled)
			# print("\n")
			# print(df_test_scaled)
			# print("\n\n")

			### SPLIT X AND Y:
			###_______________

			### TRAIN:
			###_______
			# x_y_train = native_x_y_spliter(df = df_train_scaled, 
			# 							data_cols_names = data_cols_names, 
			# 							target_col_name = target_col_name, 
			# 							look_back = look_back)

			# X_train = x_y_train['dataX']
			# y_train = x_y_train['dataY']

			### TEST:
			###______
			df_test_scaled = df_test_scaled.tail(look_back)
			df_test_scaled.reset_index(inplace = True, drop = True)

			if first_iteration:
				print_style("\nThis is the first iteration !",
								color = informative_color)
				print_style("Please wait, getting X_test and y_test.\n",
								color = informative_color)
				x_y_test_first_iter = native_x_y_spliter(
									df = df_first_iter_test_scld, 
									data_cols_names = data_cols_names, 
									target_col_name = target_col_name, 
									look_back = look_back)
				X_test_first_iter = x_y_test_first_iter['dataX']
				y_test_first_iter = x_y_test_first_iter['dataY']

			x_y_test = native_x_y_spliter(df = df_test_scaled, 
										data_cols_names = data_cols_names, 
										target_col_name = target_col_name, 
										look_back = look_back)

			X_test = x_y_test['dataX']
			# y_test = x_y_test['dataY']

			# print("\n")
			# print_style(f"Shape of X_train :{X_train.shape}", color = informative_color)
			# print_style(f"Shape of y_train :{y_train.shape}", color = informative_color)
			# print_style(f"\nShape of X_test  :{X_test.shape}\n", color = informative_color)

			### DOWNLOAD THE MODEL:
			###____________________
			# ### download_model(pair = "EURUSD", test_nbr = test_nbr)

			for model_eurusd_concerned in list_model_names:
				if not check_file_exists(
					filepath = gdrive_folder_path + "model_test_nbr_" + f'{test_nbr}-{model_eurusd_concerned}.h5'
					):
					download_model(model_eurusd_concerned = model_eurusd_concerned, 
								test_nbr = test_nbr)

			# ### DOWNLOAD LANDMARKS ABOUT FITTING:
			# ###__________________________________
			# filenames_landmarks = [
			# 			f'prev_epochs_test_nbr_{test_nbr}.txt',
			# 			f'model_test_nbr_{test_nbr}.h5',
			# 			f'history_test_nbr_{test_nbr}.pkl',
			# 			f'log_fitting_test_nbr_{test_nbr}.txt']

			# print("\n")
			# for filename_landmark in filenames_landmarks:
			# 	if not check_file_exists(gdrive_folder_path + filename_landmark):
			# 		result_downloading = direct_download_file(
			# 			cloud_file_path_name = landmarks_cloud_path + filename_landmark, 
			# 			local_file_path_name = gdrive_folder_path + filename_landmark)

			# 		if result_downloading:
			# 			print_style(f"Successfully downloaded {filename_landmark}", 
			# 				color = good_color, bold = bold)
			# 		else:
			# 			print_style(f"File not downloaded: {filename_landmark}",
			# 				color = alert_color, bold = bold)
			# print("\n")

			# ### FIT THE MODEL:
			# ###_______________
			# fit_n_save_model(X_train = X_train, 
			# 				y_train = y_train,
			# 				epochs = epochs, 
			# 				test_nbr = test_nbr,
			# 				verbose_epoch_in_callback = verbose_epoch_in_callback,
			# 				gdrive_folder_path = gdrive_folder_path,
			# 				prevent_wait = prevent_wait,
			# 				verbose_model_checkpoint = verbose_model_checkpoint,
			# 				batch_size = batch_size_fit,
			# 				validation_split = validation_split_fit,
			# 				shuffle = shuffle_fit,
			# 				verbose_fit = verbose_fit,
			# 				)

			### USE MODEL TO MAKE PREDICTIONS:
			###_______________________________
			# model = load_the_model(
			# 	!!!!!!!!!! ,

			# 	gdrive_folder_path = gdrive_folder_path, 
			# 								test_nbr = test_nbr)

			y_preds = []
			if first_iteration:
				y_preds_first_iter = []
			for model_eurusd_concerned in list_model_names:
				model = load_the_model(
					model_eurusd_concerned = model_eurusd_concerned, 
					gdrive_folder_path = gdrive_folder_path, 
					test_nbr = test_nbr)

				if first_iteration:
					y_pred_first_iter = model.predict(X_test_first_iter)
					y_preds_first_iter.append(y_pred_first_iter)
					# print_style(f"len y_preds_first_iter : {len(y_preds_first_iter)}",
					# 							color = 'yellow')

				y_pred = model.predict(X_test, verbose = False)
				assert len(y_pred) == 1, "len(y_pred) == 1"
				y_pred = float(y_pred[0])
				y_preds.append(y_pred)
			y_preds_rappr = list(map(get_rapprochement, y_preds))

			signals = list(map(get_signal, y_preds_rappr))

			### PASS ORDER(S):
			###_______________
			atleast_one_trade_taken = False
			if not first_iteration:
				balance_before = account.get_balance()
				balance_condition = False
				if balance_before > minimal_balance_tradable:
					balance_condition = True
				total_amount_2_invest = risk_factor*balance_before
				#### Signals which are not neutral:
				nbr_signals_diff_neutral = signals.count('put') + signals.count('call')
				divisor_total_amount = nbr_signals_diff_neutral
				try:
					single_trade_amount = round(total_amount_2_invest/divisor_total_amount, 3)
					single_trade_amount = 1.0 if single_trade_amount < 1.0 else single_trade_amount
				except ZeroDivisionError:
					single_trade_amount = None

				ids = []
				signals_diff_neutral = []
				trades_taken_at = []
				checks = []
				y_preds_tradable = []
				for signal, y_pred_ in zip(signals, y_preds):
					if balance_condition and signal != "neutral":
						check, _id = digital.pass_order(amount = single_trade_amount, 
														pair = "EURUSD", 
														order_type = signal, 
														expiration = 1)
						if check:
							atleast_one_trade_taken = True
						checks.append(check)
						ids.append(_id)
						signals_diff_neutral.append(signal)
						trades_taken_at.append(datetime.datetime.fromtimestamp(int(time.time())))
						y_preds_tradable.append(y_pred_)

			### COMPILE THE PREDICTION RESULTS:
			###________________________________
			if first_iteration:
				df_first_iter_x_test = pd.DataFrame([item[-1] for item in X_test_first_iter], 
									columns = data_cols_names)

				# print_style(f"df_first_iter_x_test", color = 'yellow')
				# print(df_first_iter_x_test, "\n")

				df_first_iter_x_test = df_first_iter_x_test[[close_column_name]]
				# print_style("df_first_iter_x_test", color = 'yellow')
				# print(df_first_iter_x_test, "\n")

				# print_style(f"len date_from_column : {len(date_from_column)}", color = 'yellow')

				df_first_iter_x_test['date_from'] = date_from_column
				# print_style("df_first_iter_x_test['date_from']", color = 'yellow')
				# print(df_first_iter_x_test['date_from'])	

				df_first_iter_results = df_first_iter_x_test
				# print_style("df_first_iter_results", color = 'yellow')
				# print(df_first_iter_results)

				for idx_, y_pred_first_iter in enumerate(y_preds_first_iter):
					y_pred_first_iter = [item[0] for item in y_pred_first_iter]
					# print_style(f'len y_pred_first_iter {len(y_pred_first_iter)}', color = 'yellow')

					df_first_iter_results['y_pred_' + str(idx_ + 1)] = y_pred_first_iter

					# # df_results_filename = f"df_results_test_nbr_{test_nbr}.csv"
					# # #### df_results.to_csv(gdrive_folder_path + df_results_filename, index = False)
					# # df_results.to_csv(simulating_realtime_path + \
					# # 	f"Df_results_test_nbr_{test_nbr} - Last_index_{str(last_index)}.csv")

				### GET BALANCE SHEET:
				###___________________

				all_new_accuracies = []
				all_new_ratio_trades = []
				for idx_ in range(len(list_model_names)):

					# balance_sheet_filename = f"Balance_sheet_test_nbr_{test_nbr}.txt"
					new_accuracy_n_ratio_trades = get_balance_sheet_v2(
									df_results = df_first_iter_results, 
									close_column_name = close_column_name, 
									target_shift = target_shift, 
									y_pred_col_name = "y_pred_" + str(idx_ + 1))
					all_new_accuracies.append(new_accuracy_n_ratio_trades['accuracy'])
					all_new_ratio_trades.append(new_accuracy_n_ratio_trades['ratio_trades'])
					# print_style(f"Total trades : {new_accuracy_n_ratio_trades['total_trades']}", color = 'yellow')

				reference_accuracies = [81.83, 78.45, 79.97, 80.69, 81.51]
				reference_ratio_trades = [11.15, 12.11, 10.87, 12.56, 11.45]
				print("\n")
				for reference_acc, new_acc in zip(
								reference_accuracies, all_new_accuracies):
					print_style(f"Reference Accuracy  : {reference_acc} %", color = informative_color)
					print_style(f"VERSUS New Accuracy : {new_acc} %\n", color = informative_color)

				print_style("_________________________________________", color = informative_color)
				for reference_ratio_trds, new_ratio_trds in zip(
								reference_ratio_trades, all_new_ratio_trades):
					print_style(f"Reference Ratio Trades  : {reference_ratio_trds} %", color = informative_color)
					print_style(f"VERSUS New Ratio Trades : {new_ratio_trds} %\n", color = informative_color)

				#####################################
				### NOW SET first_iteration TO FALSE:
				###__________________________________
				first_iteration = False

			# ### UPLOAD RESULT FILES:
			# ###_____________________
			# result_filenames = [
			# 	balance_sheet_filename,
			# 	df_results_filename,
			# 	# f'prev_epochs_test_nbr_{test_nbr}.txt',
			# 	f'model_test_nbr_{test_nbr}.h5',
			# 	# f'history_test_nbr_{test_nbr}.pkl',
			# 	# f'log_fitting_test_nbr_{test_nbr}.txt',
			# ]

			# print("\n")
			# for result_filename in result_filenames:
			# 	result_upload = direct_upload_file(
			# 		local_file_path_name = gdrive_folder_path + result_filename, 
			# 		cloud_file_path_name = results_cloud_path + result_filename,
			# 		)

			# 	if result_upload:
			# 		print_style(f"Successfully upload result file: {result_filename}:\n\t{result_upload}",
			# 			color = good_color, bold = bold)
			# 	else:
			# 		print_style(f"\nResult file {result_filename} wasn't uploaded !!!\n",
			# 			color = alert_color, bold = bold)
			# print("\n")


			### PRINT ITERATION TIME:
			###_______________________
			iteration_time_delta = time.time() - start_time_iter

			### CREATE AND SAVE DF INFO, IF ATLEAST ONE TRADE TAKEN:
			###_____________________________________________________

			if atleast_one_trade_taken:
				balance_after = account.get_balance()

				### Reset df_info:
				###_______________
				try:
					del df_info
				except UnboundLocalError:
					pass

				### Re-create df_info:
				###___________________
				df_info = pd.DataFrame({
					"id":ids,
					"checks":checks,
					"original_y_pred":y_preds_tradable,
					"signal":signals_diff_neutral,
					"trade_taken_at (GMT+0:00)":trades_taken_at,
					})

				df_info["nbr_signals_diff_neutral"] = nbr_signals_diff_neutral
				df_info["currency_pair"] = "EURUSD"
				df_info["df_test_length"] = df_test_length
				df_info["balance_before ($)"] = balance_before
				df_info["balance_after ($)"] = balance_after
				df_info["iteration_time_delta (s)"] = iteration_time_delta
				df_info["risk_factor"] = risk_factor
				df_info["single_trade_amount ($)"] = single_trade_amount
				df_info["total_amount_2_invest ($)"] = total_amount_2_invest
				df_info["account_type"] = account_type

				### Save df_info:
				###______________
				filepath_df_info = '/content/drive/MyDrive/df_info_trades.csv'
				df_info.to_csv(filepath_df_info, 
								index = False, 
								mode = 'a', 
								header = not os.path.isfile(path = filepath_df_info))

			### WAIT FOR THE NEXT MINUTE:
			###__________________________
			datetime_now = datetime.datetime.fromtimestamp(int(time.time()))
			print_style(f"Wait for the next minute ! Since: {datetime_now}", 
						color = informative_color)
			time.sleep((60 - datetime_now.second) - 5)

