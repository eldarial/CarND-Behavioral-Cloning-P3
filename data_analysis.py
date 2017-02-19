import csv
import numpy as np
import matplotlib.pyplot as plt
import random

# function to filter images
def filter_samples(list_im_angles, num_max_per_sample = 1200):
    filter_list = []

    all_angles = []
    for sample_line in list_im_angles:
        all_angles.append(float(sample_line[3]))
    all_angles = np.array(all_angles)


    bin_angles = np.arange(-25.05,25.1,0.1)
    hist, label = np.histogram(all_angles, bins=bin_angles)
    inds = np.digitize(all_angles, bin_angles)
    for k_bin in range(bin_angles.shape[0]):
        list_filter_bin = [list_im_angles[k] for k in range(len(inds)) if inds[k]==k_bin]
        if len(list_filter_bin) > num_max_per_sample:
            list_filter_bin = random.sample(list_filter_bin, num_max_per_sample)
        filter_list = filter_list + list_filter_bin
    return filter_list


# read csv file
samples = []
list_log = ['/home/darial/udacar/beta_simulator_linux/almost_log.csv']
#list_log = ['driving_log.csv']
for log_file in list_log:
    with open(log_file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

filter_line_samples = filter_samples(samples)

angles = []
for sample_line in samples:
    angles.append(float(sample_line[3]))
angles = np.array(angles)

angles_filtered = []
for sample_line in filter_line_samples:
     angles_filtered.append(float(sample_line[3]))
angles_filtered = np.array(angles_filtered)

bin_angles = np.arange(-25.05,25.1,0.1)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.hist(angles,bins=bin_angles,color='green')

plt.xlabel('angles')
plt.ylabel('count')
plt.title('Original Data Histogram')

fig = plt.figure()
ax = fig.add_subplot(111)

ax.hist(angles_filtered,bins=bin_angles,color='blue')

plt.xlabel('angles')
plt.ylabel('count')
plt.title('Filtered Data Histogram')

plt.show()
