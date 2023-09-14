
"""
This is the script used to combine all collected csv data files into
a single csv file.
"""

import numpy as np
import csv
import time
import os

import labels

# home_path = 'c:/Users/siddh/OneDrive/Documents/School/UMASS/Junior/Sem 2/CS 328/proj/assignment-2-part-1-group_13/assignment-2-part-1/'
home_path = f'{os.path.dirname(os.path.realpath(__file__))}'
data_dir = f'{home_path}/data'

# print the available class label (see labels.py)
act_labels = labels.activity_labels
print(act_labels)

# specify the data files and corresponding activity label
csv_files = []
activity_list = []

ext_sober_accel = '/data-sober/accel/'
for filename in os.listdir(f'{data_dir}{ext_sober_accel}'):
    csv_files.append(f'{ext_sober_accel}{filename}')
    activity_list.append('sober')

# ext_tipsy_accel = '/data-tipsy/accel/'
# for filename in os.listdir(f'{data_dir}{ext_tipsy_accel}'):
#     csv_files.append(f'{ext_tipsy_accel}{filename}')
#     activity_list.append('tipsy')

ext_drunk_accel = '/data-drunk/accel/'
for filename in os.listdir(f'{data_dir}{ext_drunk_accel}'):
    csv_files.append(f'{ext_drunk_accel}{filename}')
    activity_list.append('drunk')
    
ext_sober_gyro = '/data-sober/gyro/'
for filename in os.listdir(f'{data_dir}{ext_sober_gyro}'):
    csv_files.append(f'{ext_sober_gyro}{filename}')
    activity_list.append('sober')

# ext_tipsy_gyro = '/data-tipsy/gyro/'
# for filename in os.listdir(f'{data_dir}{ext_tipsy_gyro}'):
#     csv_files.append(f'{ext_tipsy_gyro}{filename}')
#     activity_list.append('tipsy')

ext_drunk_gyro = '/data-drunk/gyro/'
for filename in os.listdir(f'{data_dir}{ext_drunk_gyro}'):
    csv_files.append(f'{ext_drunk_gyro}{filename}')
    activity_list.append('drunk')

print(csv_files)
print(activity_list)

# Specify final output file name. 
output_filename = "data/all_labeled_data.csv"
all_data = []

zip_list = zip(csv_files, activity_list)
for f_name, act in zip_list:
    if act in act_labels:
        label_id = act_labels.index(act)
    else:
        print("Label: " + act + " NOT in the activity label list! Check labels.py")
        exit()
    print("Process file: " + f_name + " and assign label: " + act + " with label id: " + str(label_id))

    with open(f'{data_dir}{f_name}', "r") as f:
        reader = csv.reader(f, delimiter = ",")
        headings = next(reader)
        for row in reader:
            row.append(str(label_id))
            all_data.append(row)

with open(os.path.join(home_path,output_filename), 'w',  newline='') as f:
    writer = csv.writer(f)
    writer.writerows(all_data)
    print("Data saved to: " + output_filename)


