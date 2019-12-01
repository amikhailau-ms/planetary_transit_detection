from light_curve import kepler_io
import matplotlib.pyplot as plt
import numpy as np
import os
import csv

KEPLER_CSV_FILE = "E:\Course project data\csv\q1_q17_dr24_tce_2019.10.27_12.08.25.csv"
KEPLER_DATA_DIR = "E:\Course project data\light_curves"
IGNORED_KEPLER_ID = [4820550, 9489524]
KEPLER_DATA_SIZE = 2690 - len(IGNORED_KEPLER_ID)
TRAINING_TO_ALL = 0.8

def find_interval_asc(array, value):
    index_closest = 0
    max_index = len(array) - 1
    while array[index_closest] < value:
        index_closest += 1
        if index_closest >= max_index:
            break
    return index_closest - 1, index_closest

csv_data = open(KEPLER_CSV_FILE, "r")
csv_reader = csv.DictReader(csv_data)
csv_transit_data = np.empty([KEPLER_DATA_SIZE, 2009])
row_counter = 0
for row in csv_reader:
    kepid = np.int32(row["kepid"])
    if kepid in IGNORED_KEPLER_ID:
        continue
    period = np.float32(row["tce_period"])
    first_transit = np.float64(row["tce_time0bk"])
    transit_time = np.float64(row["tce_duration"]) / 24
    
    file_names = kepler_io.kepler_filenames(KEPLER_DATA_DIR, kepid)
    if len(file_names) == 0:
        continue
    all_time, all_flux = kepler_io.read_kepler_light_curve(file_names)
    
    time = np.concatenate(all_time)
    while first_transit < time[0]:
        first_transit += period
    idx1, idx2 = find_interval_asc(time, first_transit)
    length_left = len(time[:idx1])
    length_right = len(time[idx2 + 1:])
    index_first = 0
    index_last = 0
    if length_left < 500:
        index_first = idx1 - length_left
        index_last = idx2 + 1 + 1000 - length_left
    elif length_right < 500:
        index_first = idx1 - 1000 + length_right
        index_last = idx2 + 1 + length_right
    else:
        index_first = idx1 - 500
        index_last = idx2 + 1 + 500
    time = time[index_first:index_last]
    first_point = time[0]
    last_point = time[len(time) - 1]
    
    for f in all_flux:
        f /= np.median(f)
    flux = np.concatenate(all_flux)
    flux = flux[index_first:index_last]
    
    time_window = (last_point - first_point)
    first_transit = (first_transit - first_point) / time_window
    transit_time /= time_window
    for i in range(len(time)):
        time[i] = (time[i] - first_point) / time_window
    
    new_set = [[np.float64(kepid)], [np.float64(1002.0)], [np.float64(time_window)], time, flux, [np.float64(first_transit)], [np.float64(transit_time)]]
    new_array = np.concatenate(new_set)
    for i in range(len(new_array)):
        csv_transit_data[row_counter][i] = new_array[i]
    row_counter += 1

train_last = int(KEPLER_DATA_SIZE * TRAINING_TO_ALL)
np.savez("dataset/train.npz", csv_transit_data[:train_last])
np.savez("dataset/test.npz", csv_transit_data[train_last:])

plt.plot(csv_transit_data[KEPLER_DATA_SIZE - 1][3:1005], csv_transit_data[KEPLER_DATA_SIZE - 1][1005:2007], ".")
plt.xlim(np.amin(csv_transit_data[KEPLER_DATA_SIZE - 1][3:1005]), np.amax(csv_transit_data[KEPLER_DATA_SIZE - 1][3:1005]))
plt.axvline(x=csv_transit_data[KEPLER_DATA_SIZE - 1][2007], linewidth=1, color="g")
plt.show()
csv_data.close()
