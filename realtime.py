

"""
DON'T FORGET:

- SI LA BALANCE EST INFERIEURE OU EGALE A UNE CERTAINE VALEUR, IL NE FAUDRA PAS 
	PRENDRE DE TRADE !!!
	==> CETTE CERTAINE VALEUR DOIT ETRE UN PARAMETRE DE LA CLASS IQBOT.

- SI LE AMOUNT EST < 1, ALORS 1 

- freqs_seasonal DOIT AVOIR UNE VALEUR PAR DEFAUT

"""



"""
TO TEST:

- ### CHECK IF GPU IS CONNECTED:

- ### DOWNLOAD INFORMATION FILES IF NECESSARY:

- the code in file: send_info_files.py

"""







################ SECTION MAIN_VARIABLES:

model_reference_cloud_path = "drinbd-n-repeat-seas/models-de-reference/"
infos_filenames_cloud_path = "drinbd-n-repeat-seas/infos-files/"

gdrive_path = "/content/drive/MyDrive/"

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

pairs_list = ["EURUSD", "EURCHF", "EURAUD", "GBPCHF", "GBPAUD", "GBPCAD", "USDCHF"]

models_filenames = {
				"EURUSD":'model_test_nbr_drinbd_n_repeat_seas_53.4.h5', 
				"EURCHF":'model_test_nbr_drinbd_n_repeat_seas_55.EURCHF.h5', 
				"GBPCHF":'model_test_nbr_drinbd_n_repeat_seas_55.GBPCHF.h5',
				"EURAUD":'model_test_nbr_drinbd_n_repeat_seas_56_euraud.1.h5', 
				"GBPAUD":'model_test_nbr_drinbd_n_repeat_seas_56_gbpaud.1.h5', 
				"GBPCAD":'model_test_nbr_drinbd_n_repeat_seas_56_gbpcad.1.h5', 
				"USDCHF":'model_test_nbr_drinbd_n_repeat_seas_56_usdchf.1.h5'
				}

common_path_firebase_strg = "https://firebasestorage.googleapis.com/v0/b/saving-data-2ee4b.appspot.com/o/datasets_csv%2F"
dataset_urls = {
	'EURUSD':'https://firebasestorage.googleapis.com/v0/b/saving-data-2ee4b.appspot.com/o/csv_files%2FEURUSD-1.0%20Min--2024-4-18%200-1-0.csv?alt=media',
	'EURCHF': f'{common_path_firebase_strg}EURCHF-1.0%20Min--2024-4-18%200-1-0.csv?alt=media',
	'EURAUD': f'{common_path_firebase_strg}EURAUD-1.0%20Min--2024-4-18%200-1-0.csv?alt=media',
	'GBPCHF': f'{common_path_firebase_strg}GBPCHF-1.0%20Min--2024-4-18%200-1-0.csv?alt=media',
	'GBPAUD': f'{common_path_firebase_strg}GBPAUD-1.0%20Min--2024-4-18%200-1-0.csv?alt=media',
	'GBPCAD': f'{common_path_firebase_strg}GBPCAD-1.0%20Min--2024-4-18%200-1-0.csv?alt=media',
	'USDCHF': f'{common_path_firebase_strg}USDCHF-1.0%20Min--2024-4-18%200-1-0.csv?alt=media',
}

##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################

################ SECTION UTILS:

import datetime
import warnings
import pandas as pd
import os
import platform
import time

def check_gpu_connected():
	try:
		if int(os.environ["COLAB_GPU"]) > 0:
			# print("a GPU is connected.")
			return True
	except:
		# print("GPU is not connected !")
		return False

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

def manage_warnings():
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

##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################

################ SECTION MANAGE FIREBASE:


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

firebase_storage = FirebaseStorage(firebase_config = firebase_config)

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

def download_model(pair):
	assert pair.upper() in pairs_list, f'The pair your enter "{pair}" must be within the list:{pairs_list}'
	model_filename = models_filenames[pair.upper()]
	result_download = direct_download_file(
		cloud_file_path_name = model_reference_cloud_path + model_filename,
		local_file_path_name = gdrive_path + model_filename, 
		)

	print("\n")
	if result_download:
		for _ in range(5):
			print_style(f'Model successfully downloaded : {model_filename}', 
				color = IqBot.GOOD_COLOR, bold = IqBot.BOLD)
	else:
		for _ in range(20):
			print_style(f'Attention, Model Not downloaded !!! : {model_filename}', 
				color = IqBot.ALERT_COLOR, bold = IqBot.BOLD)
	print("\n")

##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################

################ SECTION IQ_OPTION_API:
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

class Binary:
	def __init__(self, account):
		self.account = account

	def pass_order(self, amount, pair, order_type, expiration):
		check, _id = self.account.buy(amount, pair, order_type, expiration)
		"""
           Connexion à IQ Option ...
           Compte : georgeskalume05@gmail.com
           Connecté à IQ Option.
			check : True
			id : 11533722642

			check : True
			id : 11533722672
			
			check : False
			id : Cannot purchase an option (active is suspended)

			##################

			check : True
			id : 11533921841
			Type of id : <class 'builtin_function_or_method'>
			str_id : 11533921841
			Type of str_id : <class 'str'>

			check : False
			id : Cannot purchase an option (active is suspended)
			Type of id : <class 'builtin_function_or_method'>
			str_id : Cannot purchase an option (active is suspended)
			Type of str_id : <class 'str'>

		"""
		return check, _id

	def get_history(self, nbr_data):
		# ## data = self.account.get_optioninfo(nbr_data)
		data = self.account.get_optioninfo_v2(nbr_data)
		history_node = data['msg']['closed_options']
		return {"all_data":data, "history_node":history_node}

	def trade_closed(self, _id):
		history_node = self.get_history(nbr_data = 30)['history_node']
		all_ids_closed_trades = [hist['id'][0] for hist in history_node]
		if _id in all_ids_closed_trades:
			return True
		else:
			return False

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
		data = account.get_position_history_v2(instrument_type = "digital-option",
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

def timestamp_converter(x):
	heure = datetime.datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S")
	return heure

def get_multi(start_timestamp):
	assert isinstance(start_timestamp, dict), 'start_timestamp must be a "dict."'
	# timestamp = datetime.datetime(year_, month_, day_, hour_, min_, sec_).timestamp()
	year_ = start_timestamp['year']
	month_ = start_timestamp['month']
	day_ = start_timestamp['day']
	hour_ = start_timestamp['hour']
	min_ = start_timestamp['minute']
	sec_ = start_timestamp['second']

	start_time = datetime.datetime(year_, month_, day_, hour_, min_, sec_).timestamp()
	delta_time = time.time() - start_time
	delta_days = delta_time/86400
	delta_weeks = delta_days/7
	week_ends = int(delta_weeks)
	days_to_exclude = week_ends*2 - 1
	stayed_days = delta_days - days_to_exclude
	stayed_minutes = math.ceil(stayed_days * 1440)
	multi = math.ceil(stayed_minutes/1_000)
	return multi


def get_nbr_candles(start_timestamp):
	assert isinstance(start_timestamp, dict), 'start_timestamp must be a "dict."'
	# timestamp = datetime.datetime(year_, month_, day_, hour_, min_, sec_).timestamp()
	year_ = start_timestamp['year']
	month_ = start_timestamp['month']
	day_ = start_timestamp['day']
	hour_ = start_timestamp['hour']
	min_ = start_timestamp['minute']
	sec_ = start_timestamp['second']
	start_time = datetime.datetime(year_, month_, day_, hour_, min_, sec_).timestamp()
	delta_time = time.time() - start_time
	delta_minutes = nbr_candles = math.ceil(delta_time/60)
	return nbr_candles


def get_concerned_candles(df, start_timestamp):
	def add_zero_if_necessary(number):
		str_number = str(number)
		if len(str_number) == 1:
			str_number = "0" + str_number
		return str_number

	df = df.copy()
	df.reset_index(inplace = True, drop = True)

	year = add_zero_if_necessary(start_timestamp['year'])
	month = add_zero_if_necessary(start_timestamp['month'])
	day = add_zero_if_necessary(start_timestamp['day'])
	hour = add_zero_if_necessary(start_timestamp['hour'])
	minutes = add_zero_if_necessary(start_timestamp['minute'])
	second = add_zero_if_necessary(start_timestamp['second'])

	start_timestamp_ = str(year) +"-"+ str(month) +"-"+ str(day) + " " + str(hour) +":"+ str(minutes) +":"+ str(second)

	# date_from_ = df['date_from'].tolist()[0]
	# print(df, "\n")
	# print("Type of date_from :", type(date_from_))
	print("start_timestamp_  :", start_timestamp_)

	from_index = list(df[df['date_from'] == start_timestamp_].index)
	assert len(from_index) == 1, 'We should have only one index from "from_index = list(df[df[\'date_from\'] == start_timestamp_].index)"'
	from_index = from_index[0]
	take_only = df.shape[0] - from_index
	new_df = df.tail(take_only)
	new_df.reset_index(inplace = True, drop = True)
	assert new_df.shape[0] < df.shape[0], 'The length of "new_df" should be < to the length of "df" !!!'
	return new_df

def get_big_data_candles(account, pair, timeframe, start_timestamp, verbose = False):
	timestamp = time.time()
	total = []

	if verbose:
		print("Downloading candles ...")
	multi = get_multi(start_timestamp = start_timestamp)
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
	new_df = get_concerned_candles(df = df, start_timestamp = start_timestamp)
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

##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################

################ SECTION DL MODEL:

import datetime
import time

if check_environment() == 'Colab':
	from tensorflow import keras
	from tensorflow.keras.models import load_model
	import tensorflow as tf

##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################

################ SECTION DATASCIENCE:

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

##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################

################ SECTION REGENERATE SEASONALITY:

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


def add_seas_column_to_df(df, 
						# ### size_get_seas, 
						nbr_rows_get_seas, 
						freq, 
						close_col_name, 
						plot_results = False, 
						model = "additive", 
						limit = 5):
	df = df.copy()
	# ### len_df_get_seas = int(df.shape[0]*size_get_seas)
	len_df_get_seas = nbr_rows_get_seas

	df_get_seas = df.head(len_df_get_seas)
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

##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################

################ SECTION MANAGE:


class IqBot:
	### SET VARIABLES ACCORDING TO ENVIRONMENT:
	### _______________________________________
	if check_environment() == 'Colab':
		INFORMATIVE_COLOR = 'cyan'
		ALERT_COLOR = 'red'
		GOOD_COLOR = 'green'
		BOLD = True
	elif check_environment() == 'Local_PC':
		INFORMATIVE_COLOR = None
		ALERT_COLOR = None
		GOOD_COLOR = None
		BOLD = False

	def __init__(self, 
				iq_email, 
				iq_password, 
				waintings, 
				candles_timeframe,
				loop_window,
				freqs_seasonal,
				close_column_name,
				verbose_predict,
				trade_binary,
				trade_digital,
				take_simultaneous_trades_different_pairs,
				look_back = 15,
				nbr_rows_get_seas = 19_999, 
				nbrs_rows_train_part = 50_000,
				account_type = 'PRACTICE'):


		### CONNECT TO THE IQ OPTION ACCOUNT:
		###__________________________________
		self.account = connect_2_iq_option_account(iq_email = iq_email, 
									iq_password = iq_password, 
									account_type = account_type)

		### INITIALIZE PARAMS:
		###___________________
		self.candles_timeframe = candles_timeframe
		self.freqs_seasonal = freqs_seasonal
		self.nbr_rows_get_seas = nbr_rows_get_seas
		self.close_column_name = close_column_name
		self.nbrs_rows_train_part = nbrs_rows_train_part
		self.look_back = look_back
		self.verbose_predict = verbose_predict
		self.binary = Binary(account = self.account)
		self.digital = Digital(account = self.account)
		self.take_simultaneous_trades_different_pairs = take_simultaneous_trades_different_pairs
		self.trade_binary = trade_binary
		self.trade_digital = trade_digital

		### ASSERTS:
		###_________
		assert len(waintings) == 3, 'Il doit y avoir 3 valeurs dans la liste "waintings"'
		assert len(loop_window) == 2, 'Il doit y avoir 2 valeurs dans la liste "loop_window"'

		### WAIT TO BE CONNECTED TO GOOGLE DRIVE:
		###______________________________________
		wait_gdrive_connected()

		### CHECK IF GPU IS CONNECTED:
		###___________________________
		if check_environment() == 'Colab':
			gpu_connected = check_gpu_connected()
			if not gpu_connected:
				print_style("\nPLEASE CONNECT THE GPU !!!", 
								color = IqBot.ALERT_COLOR, bold = IqBot.BOLD)
				print_style("The continuation of code will run only when the GPU will be connected !!!\n", 
								color = IqBot.INFORMATIVE_COLOR)
				while True:
					time.sleep(1)
			else:
				print_style("\nGPU is connected !!!\n", 
								color = IqBot.GOOD_COLOR, bold = IqBot.BOLD)

		### MANAGE WARNINGS:
		### ________________
		manage_warnings()

		### DOWNLOAD INFORMATION FILES IF NECESSARY:
		###_________________________________________
		infos_filenames = [f"Infos-realtime-drinbd-n-repeat-seas-pair-{pair}.txt" for pair in pairs_list]
		# infos_filenames_cloud_path

		infos_files_existed = False
		for infos_filename in infos_filenames:
			if check_file_exists(filepath = gdrive_path + infos_filename):
				infos_files_existed = True
				break

		if not infos_files_existed:
			for infos_filename in infos_filenames:
				file_downloaded = direct_download_file(cloud_file_path_name = infos_filenames_cloud_path + infos_filename, 
														local_file_path_name = gdrive_path + infos_filename)
				if file_downloaded:
					print_style(f"Successfully downloaded : {infos_filename}", 
									color = IqBot.GOOD_COLOR, bold = IqBot.BOLD)
				else:
					for _ in range(5):
						print_style(f"File not downloaded : {infos_filename}", 
										color = IqBot.ALERT_COLOR, bold = IqBot.BOLD)
						print_style(f"\tMaybe it doesn't exist yet in on Firebase Storage",
										color = IqBot.ALERT_COLOR, bold = IqBot.BOLD)
					print("\n")

			for infos_filename in infos_filenames:
				try:
					with open(gdrive_path + infos_filename, "r", encoding = "utf-8") as f:
						data = f.readlines()
					data = [d.strip() for d in data]
					data = [d for d in data if d != ""]
					last_value_in_info_file = data[-1]
					if "Timestamp of last sending to firebase:" in last_value_in_info_file:
						last_timestamp_info_files_sent = int(last_value_in_info_file.split(" ")[-1])
						last_datetime_info_files_sent = datetime.datetime.fromtimestamp(last_timestamp_info_files_sent)
						print_style(f"Le fichier a été envoyé sur Firebase Storage en cette datetime: {last_datetime_info_files_sent}", 
										color = IqBot.INFORMATIVE_COLOR, bold = IqBot.BOLD)

						time_elapsed_from_sent = time.time() - last_timestamp_info_files_sent
						if time_elapsed_from_sent//60 > 5:
							for _ in range(20):
								print('Il y a plus de 5 minutes que les fichiers txt "Info files" ont été envoyé sur Firebase Storage',
										color  = IqBot.ALERT_COLOR, bold = IqBot.BOLD)
							time.sleep(waintings[2])
					else:
						for _ in range(10):
							print_style(f"Vous re-éxecuter le code après une simple interruption de Google Colab.",
								color = IqBot.INFORMATIVE_COLOR)
						print("\n")
				except FileNotFoundError:
					pass

	def main(self, pair, risk_factor, expiration):
		### DOWNLOAD MODEL FROM FIREBASE STORAGE:
		###______________________________________
		download_model(pair = pair)

		### LOAD THE MODEL:
		###________________
		model_filepath = gdrive_path + models_filenames[pair.upper()]
		print_style(f"\nModel filepath: {model_filepath}\n", color = IqBot.INFORMATIVE_COLOR)
		model = load_model(model_filepath = model_filepath)
		# print_style('Model successfully loaded :', color = IqBot.GOOD_COLOR, bold = IqBot.BOLD)
		# print(model)

		### INFOS FILENAME:
		###________________
		infos_filename = f"Infos-realtime-drinbd-n-repeat-seas-pair-{pair}.txt"

		### SOME USEFULL FUNCTIONS:
		###________________________

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

		def get_first_date_from_value(df):
			df = df.copy()
			df = df.head(2)
			df_date_from = df['date_from'].tolist()
			df_date_from_first_value = df_date_from[0]
			return df_date_from_first_value

		def get_last_date_from_value(df):
			df = df.copy()
			df = df.tail(2)
			values = df['date_from'].tolist()
			last_value = values[-1]
			return last_value

		def get_timestamp_from_datetime_as_str(str_datetime):
			date = str_datetime.split(" ")[0]
			time = str_datetime.split(" ")[1]
			year = int(date.split("-")[0])
			month = int(date.split("-")[1])
			day = int(date.split("-")[2])
			hour = int(time.split(":")[0])
			minute = int(time.split(":")[1])
			second = int(time.split(":")[2])

			return {'year':year, 'month':month, 'day':day, 
					'hour':hour, 'minute':minute, 'second':second}

		### GET BIG DATA CANDLES:
		###______________________		
		reference_df_firebase = pd.read_csv(dataset_urls[pair.upper()])
		first_date_from_of_reference_df_firebase = get_first_date_from_value(df = reference_df_firebase)
		big_df_start_timestamp = get_timestamp_from_datetime_as_str(str_datetime = first_date_from_of_reference_df_firebase)
		big_df = get_big_data_candles(pair = pair, 
							account = self.account, 
							timeframe = self.candles_timeframe, 
							start_timestamp = big_df_start_timestamp,
							verbose = True)

		msg_date_from_diff = '''La première valeur de la colonne "date_from" de "big_df" 
(le df téléchargé maintenant) devrait être égale à celle de df enregistré 
sur Firebase Storage (le df utilisé lors de l\'entrainement du modèle).
'''
		assert get_first_date_from_value(big_df) == get_first_date_from_value(reference_df_firebase), msg_date_from_diff

		### THE LONG LOOP:
		###_______________
		opened_binary_trades_ids = []
		opened_digital_trades_ids = []
		while True:
			### RUN THE ITERATION ONLY ON THE BEGINNING OF A MINUTE:
			if loop_window[0] <= datetime.datetime.now().second <= loop_window[1]:
				trade_is_taken = False
				starting_treatment_time = time.time()
				### DOWNLOAD THE SMALL DF:
				last_date_from_of_big_df = get_last_date_from_value(df = big_df)
				start_timestamp_small_df = get_timestamp_from_datetime_as_str(
											str_datetime = last_date_from_of_big_df)
				nbr_candles = get_nbr_candles(start_timestamp = start_timestamp_small_df)

				small_df = get_small_data_candles(account = self.account, 
											pair = pair, 
											timeframe = self.candles_timeframe, 
											nbr_candles = nbr_candles)

				message_date_from_big_df_small_df_diff = '''La DERNIÈRE valeur de la colonne "date_from" de "big_df" doit être égale 
à la PREMIÈRE valeur de la même colonne pour le "small_df" !'''

				assert get_last_date_from_value(big_df) == get_first_date_from_value(small_df), message_date_from_big_df_small_df_diff

				# ### VERIFIER SI LE SMALL DF RESPECTE LA CONDITION DE PRISE DE TRADE:
				condition_trade_by_last_datetime = False

				last_date_from_of_small_df = get_last_date_from_value(small_df)
				timestamp_last_date_from_of_small_df = get_timestamp_from_datetime_as_str(
														str_datetime = last_date_from_of_small_df)

				if int(timestamp_last_date_from_of_small_df['year']) == int(datetime.datetime.now().year) and\
				 int(timestamp_last_date_from_of_small_df['month']) == int(datetime.datetime.now().month) and\
				 int(timestamp_last_date_from_of_small_df['day']) == int(datetime.datetime.now().day) and\
				 int(timestamp_last_date_from_of_small_df['hour']) == int(datetime.datetime.now().hour) and\
				 int(timestamp_last_date_from_of_small_df['minute']) == int(datetime.datetime.now().minute):
					condition_trade_by_last_datetime = True

				### CREATE COMPLETE DF:
				###____________________
				### WE SHALL FIRST, DROP THE FIRST VALUE OF SMALL DF IN ORDER TO AVOID 
				### TO REPEAT THE ROW IN THE COMPLETE DF:
				small_df = small_df.tail(len(small_df)-1)
				complete_df = pd.concat([big_df, small_df], axis = 0)
				complete_df.reset_index(inplace = True, drop = True)
				complete_df_shape = complete_df.shape

				### ADD COLUMNS TO complete_df:
				###____________________________

				### ADD SEASONALITY COLUMNS:
				###_________________________

				for freq in self.freqs_seasonal:
					complete_df = add_seas_column_to_df(df = complete_df, 
										nbr_rows_get_seas = self.nbr_rows_get_seas, 
										freq = freq, 
										close_col_name = self.close_column_name)

				### DELETE THE PART OF COMPLETE DF WHICH CONCERNS THE GETTING SEASONALITY:
				complete_df = complete_df.tail(complete_df.shape[0] - self.nbr_rows_get_seas)
				complete_df.reset_index(inplace = True, drop = True)

				### GET TEST PART/ DROP TRAIN PART:
				###________________________________
				df_test = complete_df.tail(complete_df.shape[0] - self.nbrs_rows_train_part)
				df_test.reset_index(inplace = True, drop = True)

				### ADD WAVELETS COLUMNS AND STATIONNARIZED CLOSE:
				###_______________________________________________
				### ==> TEST DATA:
				###_______________
				df_test_wavelets_cols = add_wavelets_columns(
												close_column = df_test[self.close_column_name])
				df_test['dwt_cA'] = df_test_wavelets_cols['dwt_cA']
				df_test['dwt_cD'] = df_test_wavelets_cols['dwt_cD']
				df_test['dwt_cA_stationnarized'] = df_test_wavelets_cols['dwt_cA_stationnarized']
				df_test['stationnarized_close'] = stationnarize_close_column(df = df_test,
												close_column_name = self.close_column_name)
				df_test_shape = df_test.shape

				### SCALE TEST DATA:
				###_________________
				df_test_scaled, _ = data_scaler(df = df_test)

				### SET DATA COLUMNS NAMES:
				###________________________
				data_cols_names_seasonality = [f'seasonality_{d}' for d in self.freqs_seasonal]
				data_cols_names = ['close', 
									'stationnarized_close',
									# 'soft_0.5', 
									# 'less_0.5', 
									# 'soft_0.5_stationnarized',
									'dwt_cA', 
									'dwt_cD', 
									'dwt_cA_stationnarized']
				data_cols_names += data_cols_names_seasonality

				### TEST:
				###______
				### HERE, WE NEED ONLY THE LAST FRAGMENT OF df_test_scaled (tail(look_back)):
				df_test_scaled = df_test_scaled.tail(self.look_back)
				df_test_scaled.reset_index(inplace = True, drop = True)

				### ADD SIMULATED TARGET, JUST FOR COMPUTATION OF X AND Y:
				df_test_scaled['simulated_target'] = 0.0
				x_y_test = native_x_y_spliter(df = df_test_scaled, 
											data_cols_names = data_cols_names, 
											target_col_name = 'simulated_target', 
											look_back = self.look_back)

				last_X_test = x_y_test['dataX']
				# #### y_test = x_y_test['dataY']

				### USE THE MODEL TO MAKE A PREDICTION:
				###____________________________________
				prediction_start_time = time.time()
				y_pred = model.predict(last_X_test, verbose = self.verbose_predict)
				prediction_time_taken = round(time.time() - prediction_start_time, 3)
				y_pred = [item[0] for item in y_pred]
				assert len(y_pred) == 1, 'Il devrait y avoir une seule valeur "y_pred" !'
				y_pred = float(y_pred[0])
				assert isinstance(y_pred, float), 'La valeur de "y_pred" devrait être un nombre décimal !'

				y_pred_rapprochmt = get_rapprochement(y_pred = y_pred)

				if y_pred_rapprochmt == 0.0:
					trade_signal = 'put'
				elif y_pred_rapprochmt == 0.5:
					trade_signal = 'neutral'
				elif y_pred_rapprochmt == 1.0:
					trade_signal = 'call'

				if self.take_simultaneous_trades_different_pairs:
					### CHECK OPENED / CLOSED TRADES:
					###______________________________
					### binary:
					for opened_binary_trade_id in opened_binary_trades_ids:
						binary_trade_closed = self.binary.trade_closed(
														_id = opened_binary_trade_id)
						if binary_trade_closed:
							try:
								opened_binary_trades_ids.remove(opened_binary_trade_id)
							except ValueError:
								pass

					### digital:
					for opened_digital_trade_id in opened_digital_trades_ids:
						digital_trade_closed = self.digital.trade_closed(
														_id = opened_digital_trade_id)
						if digital_trade_closed:
							try:
								opened_digital_trades_ids.remove(opened_digital_trade_id)
							except ValueError:
								pass

				### CONDITION ABOUT SIMULTANEOUS TRADES:
				###_____________________________________
				"""
				# take_binary_trade_simult_condition = True
				# take_digital_trade_simult_condition = True
				# ### simultanéité autorisée ou pas, si opened_binary_trades_ids == 0
				# ### on a le droit de prendre un nouveau trade binary
				# if len(opened_binary_trades_ids) == 0:
				# 	take_binary_trade_simult_condition = True
				# ### simultanéité autorisée ou pas, si opened_binary_trades_ids == 0
				# ### on a le droit de prendre un nouveau trade digital
				# if len(opened_digital_trades_ids) == 0:
				# 	take_digital_trade_simult_condition = True
				# ### si simultanéité autorisée, on a le droit de prendre un trade
				# ### qu'il y ait un/plusieurs autre(s) trade(s) ouvert(s) ou pas.
				# if self.take_simultaneous_trades_different_pairs:
				# 	take_binary_trade_simult_condition = True
				# 	take_digital_trade_simult_condition = True
				# ### si simultanéité non autorisée, on a pas le droit de prendre 
				# ### un nouveau trade s'il y en a déjà un/plusieurs qui est/sont ouvert(s)
				# else:
				# 	if len(opened_binary_trades_ids) > 0:
				# 		take_binary_trade_simult_condition = False
				# 	if len(opened_digital_trades_ids) > 0:
				# 		take_digital_trade_simult_condition = False
				"""

				take_binary_trade_simult_condition = True
				take_digital_trade_simult_condition = True
				if not self.take_simultaneous_trades_different_pairs:
					if len(opened_binary_trades_ids) > 0:
						take_binary_trade_simult_condition = False
					if len(opened_digital_trades_ids) > 0:
						take_digital_trade_simult_condition = False					

				### TAKE A TRADE OR WAIT:
				###______________________
				if condition_trade_by_last_datetime and trade_signal != "neutral":
					balance_before = self.account.get_balance()
					amount = risk_factor * balance_before

					if self.trade_binary and take_binary_trade_simult_condition:
						### TAKE BINARY POSITION:
						###______________________
						check_binary, id_binary = self.binary.pass_order(
							amount = amount, 
							pair = pair, 
							order_type = trade_signal, 
							expiration = expiration,)
						if check_binary:
							trade_is_taken = True
							opened_binary_trades_ids.append(id_binary)

					if self.trade_digital and take_digital_trade_simult_condition:
						### TAKE DIGITAL POSITION:
						###_______________________
						check_digital, id_digital = self.digital.pass_order(
							amount = amount, 
							pair = pair, 
							order_type = trade_signal, 
							expiration = expiration,)
						if check_digital:
							trade_is_taken = True
							opened_digital_trades_ids.append(id_digital)

					trades_taken_at_datetime = datetime.datetime.fromtimestamp(
																			int(time.time()))

				### PRINTINGS:
				###___________

				if trade_is_taken:
					print_style(f"\nTrade is taken on : {pair}", color = IqBot.INFORMATIVE_COLOR)
					print_style(f"Balance before      : {balance_before} $", color = IqBot.INFORMATIVE_COLOR)
					print_style(f"Invested amount     : {amount} $", color = IqBot.INFORMATIVE_COLOR)
					print_style(f"Datetime            : {trades_taken_at_datetime}", color = IqBot.INFORMATIVE_COLOR)
					print_style(f"Trade signal        : {trade_signal.upper()}", color = IqBot.INFORMATIVE_COLOR)
				if "Cannot purchase an option (active is suspended)" in str(id_binary):
					print_style(f"\nBinary trade not taken because: {pair} is suspended !", color = IqBot.INFORMATIVE_COLOR)
					print_style(f"Datetime                 : {trades_taken_at_datetime}", color = IqBot.INFORMATIVE_COLOR)
					print_style(f"Trade signal             : {trade_signal.upper()}", color = IqBot.INFORMATIVE_COLOR)
				if "invalid instrument" in str(id_digital):
					print_style(f"\nDigital trade not taken because: {pair} is suspended !", color = IqBot.INFORMATIVE_COLOR)
					print_style(f"Datetime                 : {trades_taken_at_datetime}", color = IqBot.INFORMATIVE_COLOR)
					print_style(f"Trade signal             : {trade_signal.upper()}", color = IqBot.INFORMATIVE_COLOR)
				print("\n")

				ending_treatment_time = time.time()
				iteration_treatment_time_taken = round(ending_treatment_time - starting_treatment_time, 3)

				### SAVE INFORMATIONS:
				###___________________
				with open(gdrive_path + infos_filename, "a", encoding = "utf-8") as f:
					f.write(f"Currency pair         : {pair}\n")
					f.write(f"Shape of Dataframe    : {complete_df_shape}\n")
					f.write(f"Shape of Df test      : {df_test_shape}\n")
					f.write(f"Trade signal          : {trade_signal}\n")
					f.write(f"Starting treatment    : {int(datetime.datetime.fromtimestamp(int(starting_treatment_time)))}\n")
					f.write(f"Ending treatment      : {int(datetime.datetime.fromtimestamp(int(ending_treatment_time)))}\n")
					f.write(f"Iteration time delta  : {iteration_treatment_time_taken} second(s)\n")
					f.write(f"Prediction time delta : {prediction_time_taken} second(s)\n")
					if trade_is_taken:
						f.write(f"Trade taken at        : {trades_taken_at_datetime}\n")
						f.write(f"Risk factor           : {risk_factor}\n")
						f.write(f"Balance before        : {balance_before} $\n")
						f.write(f"Invested amount       : {amount} $\n")

						try:
							f.write(f"Check Binary      : {check_binary}\n")
						except:
							pass
						try:
							f.write(f"Id Binary order   : {id_binary}\n")
						except:
							pass
						try:
							f.write(f"Check Digital      : {check_digital}\n")
						except:
							pass
						try:
							f.write(f"Id Digital order   : {id_digital}\n")
						except:
							pass
					f.write("\n")

				### WAIT FOR THE NEXT MINUTE:
				###__________________________
				datetime_now = datetime.datetime.fromtimestamp(int(time.time()))
				print_style(f"\nThe Currency Pair {pair} is waiting for the next minute.", 
					color = IqBot.INFORMATIVE_COLOR)
				print_style(f"Datetime now : {datetime_now}\n",
					color = IqBot.INFORMATIVE_COLOR)
				time.sleep((60 - datetime_now.second) - 5)

