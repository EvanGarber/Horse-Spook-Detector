import pandas as pd
import numpy as np
import csv
import torch
from typing import Optional
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import LinearSegmentedColormap

# def get_numpy_data(csv_path: str) -> np.ndarray:
#   dataframe = get_pandas_data(csv_path)
'''
Get data from csv_path, associate labels, and interpolate to standard sample rate
'''
def get_pandas_data(csv_path: str, interpolate: bool = False) -> pd.DataFrame:
  # get data
  dataframe = pd.read_csv(csv_path, na_values=['nan', ' nan'])
  dataframe = dataframe.set_index('millis')

  if (interpolate):
  # standardize sample rate
    new_index = pd.Index(np.arange(dataframe.index.min(), dataframe.index.max(), 30))
    dataframe = dataframe.reindex(dataframe.index.union(new_index))
    dataframe = dataframe.interpolate()
    dataframe = dataframe.loc[(dataframe.index - dataframe.index.min()) % 30 == 0]

  # assign labels
  dataframe['label'] = 0
  with open('formatted_data/spook_annotations.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for file_name, start_time, end_time in reader:
      start_time = int(start_time)
      end_time = int(end_time)
      if file_name in csv_path:
        dataframe.loc[(dataframe.index >= start_time) & (dataframe.index <= end_time), 'label'] = 1

  return dataframe

'''
Get summarized features from windows of a chosen length
'''
def get_window_features(window_length: int, overlapping: bool = False, interpolate: bool = False) -> np.ndarray:
  cosmic_spooking = get_pandas_data('formatted_data/cosmic_spooking.csv', interpolate)
  cosmic_standing = get_pandas_data('formatted_data/cosmic_standing.csv', interpolate)
  cosmic_walking = get_pandas_data('formatted_data/cosmic_walking.csv', interpolate)
  coy_spooking = get_pandas_data('formatted_data/coy_spooking.csv', interpolate)
  coy_standing = get_pandas_data('formatted_data/coy_standing.csv', interpolate)
  coy_walking = get_pandas_data('formatted_data/coy_walking.csv', interpolate)

  timeseries = [cosmic_spooking, cosmic_standing, cosmic_walking, coy_spooking, coy_standing, coy_walking]
  D = len(cosmic_spooking.columns)
  series_window_features_array = []
  series_window_labels_array = []

  for df in timeseries:
    df = df.drop(['imu_temp_c','imu_temp_l','imu_temp_r'], axis=1)
    # df.to_numpy() would be N x D,
    # D should be 23 for our data
    # windowed view creates an array of = W x D x L where L is the length of a window and W is # windows (N - L)

    windows = np.lib.stride_tricks.sliding_window_view(df.to_numpy(), window_length, axis=0)
    if (not overlapping):
      windows = windows[::window_length]
    # extract label - if there is any point that is classified as spooking, classify whole window as spooking
    windows_label: np.array = windows[:,-1,window_length//2]

    windows = windows[:,:-1,:]
    # Min in each sensor reading, W x D
    window_min = windows.min(axis=2)
    # Max in each sensor reading, W x D
    window_max = windows.max(axis=2)

    window_abs_max = np.stack((np.abs(window_min), np.abs(window_max))).max(axis=0)
    # Mean in each sensor reading, W x D
    window_mean = windows.mean(axis=2)
    # Variance in each sensor reading, W x D
    window_var = windows.var(axis=2)
    # Indices from min to max
    window_min_index = np.argmin(windows, axis=2)
    window_max_index = np.argmax(windows, axis=2)
    max_to_min_dist = window_max_index - window_min_index
    # Concat 'em, W x (# time extracted feats * D)
    window_features = np.concatenate((window_min, window_max, window_abs_max, window_mean, window_var, max_to_min_dist), axis=1)
    
    series_window_features_array.append(window_features)
    series_window_labels_array.append(windows_label)
  datapoints = np.concatenate(series_window_features_array, axis=0)
  labels = np.concatenate(series_window_labels_array, axis=0)
  print(datapoints.shape)
  print(labels.shape)
  return datapoints, labels

def get_all_windows_raw(window_length: int, overlapping: bool = False, interpolate: bool = False) -> np.ndarray:
  cosmic_spooking = get_pandas_data('formatted_data/cosmic_spooking.csv', interpolate)
  cosmic_standing = get_pandas_data('formatted_data/cosmic_standing.csv', interpolate)
  cosmic_walking = get_pandas_data('formatted_data/cosmic_walking.csv', interpolate)
  coy_spooking = get_pandas_data('formatted_data/coy_spooking.csv', interpolate)
  coy_standing = get_pandas_data('formatted_data/coy_standing.csv', interpolate)
  coy_walking = get_pandas_data('formatted_data/coy_walking.csv', interpolate)

  timeseries = [cosmic_spooking, cosmic_standing, cosmic_walking, coy_spooking, coy_standing, coy_walking]

  window_array = []
  window_labels_array = []

  for df in timeseries:
    windows, windows_label = get_windows_raw(df, window_length, overlapping)
    window_array.append(windows)
    window_labels_array.append(windows_label)
  datapoints = np.concatenate(window_array, axis=0)
  labels = np.concatenate(window_labels_array, axis=0)

  return datapoints, labels

def get_windows_scaled(window_length: int, overlapping: bool = False, interpolate: bool = False) -> (np.ndarray, np.ndarray):
  cosmic_spooking = get_pandas_data('formatted_data/cosmic_spooking.csv', interpolate)
  cosmic_standing = get_pandas_data('formatted_data/cosmic_standing.csv', interpolate)
  cosmic_walking = get_pandas_data('formatted_data/cosmic_walking.csv', interpolate)
  coy_spooking = get_pandas_data('formatted_data/coy_spooking.csv', interpolate)
  coy_standing = get_pandas_data('formatted_data/coy_standing.csv', interpolate)
  coy_walking = get_pandas_data('formatted_data/coy_walking.csv', interpolate)

  timeseries = [cosmic_spooking, cosmic_standing, cosmic_walking, coy_spooking, coy_standing, coy_walking]
  D = len(cosmic_spooking.columns)
  window_array = []
  window_labels_array = []

  for df in timeseries:
    df = df.drop(['imu_temp_c','imu_temp_l','imu_temp_r'], axis=1)
    # df.to_numpy() would be N x D,
    # D should be 23 for our data
    # windowed view creates an array of = W x D x L where L is the length of a window and W is # windows (N - L)
    series = df.to_numpy()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(series)
    windows = np.lib.stride_tricks.sliding_window_view(series, window_length, axis=0)
    if (not overlapping):
      windows = windows[::window_length]
    # extract label - if there is any point that is classified as spooking, classify whole window as spooking
    windows_label: np.array = windows[:,-1,window_length//2]
    
    window_array.append(windows)
    window_labels_array.append(windows_label)
  datapoints = np.concatenate(window_array, axis=0)
  labels = np.concatenate(window_labels_array, axis=0)

  return datapoints, labels

def get_raw_features(interpolate: bool = False) -> np.ndarray:
  cosmic_spooking = get_pandas_data('formatted_data/cosmic_spooking.csv', interpolate)
  cosmic_standing = get_pandas_data('formatted_data/cosmic_standing.csv', interpolate)
  cosmic_walking = get_pandas_data('formatted_data/cosmic_walking.csv', interpolate)
  coy_spooking = get_pandas_data('formatted_data/coy_spooking.csv', interpolate)
  coy_standing = get_pandas_data('formatted_data/coy_standing.csv', interpolate)
  coy_walking = get_pandas_data('formatted_data/coy_walking.csv', interpolate)

  timeseries = [cosmic_spooking, cosmic_standing, cosmic_walking, coy_spooking, coy_standing, coy_walking]
  features_array = []
  labels_array = []

  for df in timeseries:
    # df = df.drop(['imu_temp_c','imu_temp_l','imu_temp_r'], axis=1)
    # df.to_numpy() would be N x D,
    # D should be 23 for our data
    data = df.to_numpy()
    labels = data[:,-1]
    data = data[:,:-1]
    labels_array.append(labels)
    features_array.append(data)
  datapoints = np.concatenate(features_array, axis=0)
  labels = np.concatenate(labels_array, axis=0)

  return datapoints, labels

def get_precision(predictions: np.ndarray, actual: np.ndarray):
  # True Positive / (True Positive + False Positive)
  tp = ((predictions == 1) & (actual == 1)).sum()
  tp_fp = np.count_nonzero(predictions)
  return tp/tp_fp

def get_accuracy(predictions: np.ndarray, actual: np.ndarray):
  # (True Positive + True Negative) / Total
  total = predictions.shape[0]
  print(total)
  correctly_classified = np.count_nonzero(predictions == actual)
  return correctly_classified/total


def get_recall(predictions: np.ndarray, actual: np.ndarray):
  # True Positive / (True Positive + False Negative)
  tp = ((predictions == 1) & (actual == 1)).sum()
  tp_fn = np.count_nonzero(actual)
  return tp/tp_fn


def get_f1(predictions: np.ndarray, actual: np.ndarray):
  precision = get_precision(predictions, actual)
  recall = get_recall(predictions, actual)
  return 2 * precision * recall / (precision + recall)

def plot_axis(axs, df):
  # axs.plot(df.index, df['acc_x_c'], label='Acc X', linewidth=2)
  # axs.plot(df.index, df['acc_y_c'], label='Acc Y', linewidth=2)
  # axs.plot(df.index, df['acc_z_c'], label='Acc Z', linewidth=2)
  axs.plot(df.index, df['obj_temp'], label='Ear Temp', linewidth=2)
  axs.plot(df.index, df['amb_temp'], label='Ambient Temp', linewidth=2)
  axs.plot(df.index, df['imu_temp_c'], label='IMU C Temp', linewidth=2)
  axs.plot(df.index, df['imu_temp_l'], label='IMU L Temp', linewidth=2)
  axs.plot(df.index, df['imu_temp_r'], label='IMU R Temp', linewidth=2)

def plot_detections(model, path, window_length, overlapping, interpolate, threshold = None):
  with torch.no_grad():
    model.eval()
    df = get_pandas_data(path, interpolate)

    # mask = (df.index > 20000) & (df.index < 25000)
    # df = df[mask]

    windows, windows_label = get_windows_raw(df, window_length, overlapping)
    windows = np.swapaxes(windows, 1, 2)
    detection_probabilities = []
    df = df.drop(['imu_temp_c','imu_temp_l','imu_temp_r'], axis=1)
    # each window is 
    for window in windows:
      window = torch.tensor(np.expand_dims(window, 0), dtype=torch.float32) #add batch dimension
      out = model(window).squeeze()
      detection_probabilities.append(out.item())
    
    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    # Custom colormap for gradient
    cmap = LinearSegmentedColormap.from_list("confidence_cmap", ["white", "#000000"])

    # Plot the gradient as an image
    miny = -2
    maxy = 10
    extent = [df.index.min(), df.index.max(), miny, maxy]  # Adjust extent to match your plot range

    detections = np.array(detection_probabilities)
    if (threshold != None):
      detections = (detections >= threshold).astype(int)
    # Create 2D arrays for the labeling images
    if (overlapping):
      detections_image = np.expand_dims(detections, axis=0)
    else:
      detections_image = np.expand_dims(detections.repeat(window_length), axis=0) # Single-row
      # detections_image = np.expand_dims(detections, axis=0)
      # new_detections_image = np.zeros(detections_image.shape[1] * window_length)
      # new_detections_image[::window_length] = detections_image
      # detections_image = np.expand_dims(new_detections_image, axis=0)
    detections_image = np.repeat(detections_image, 4, axis=0)  # Repeat rows to create a 2D background
    ax.imshow(detections_image, aspect='auto', extent=extent, origin='lower', cmap=cmap, alpha=1)

    # Plot sensor data
    ax.set_ylim(miny, maxy)
    print(df['label'].shape)
    print(detections_image.shape)
    print(detections_image.max())
    plot_axis(ax, df)

  
def get_windows_raw(df: pd.DataFrame, window_length: int, overlapping: bool = False) -> np.ndarray:
  df = df.drop(['imu_temp_c','imu_temp_l','imu_temp_r'], axis=1)
  # df.to_numpy() would be N x D,
  # D should be 23 for our data
  # windowed view creates an array of = W x D x L where L is the length of a window and W is # windows (N - L)
  windows = np.lib.stride_tricks.sliding_window_view(df.to_numpy(), window_length, axis=0)
  if (not overlapping):
    windows = windows[::window_length]
  # extract label - if there is a mid-point that is classified as spooking, classify whole window as spooking
  windows_label = np.max(windows[:,-1,:], axis=-1)

  return windows, windows_label
