import numpy as np
import pandas as pd
import glob
import os

# Set parameters
WINDOW_SIZE = 50  # Sampling rate = 50 Hz so it means 1sec(use 50 sampling)
THRESHOLD = 60  # If over 60 % of data are activity, then it is not non-activity
# Sampling rate = 50 Hz so it means 200 ms(10 sampling is overrap) less
# than WINDOW_SIZE!!!
SLIDE_SIZE = 10


def data_import(pattern_x, pattern_y):
    xx = np.empty([0, WINDOW_SIZE, 180], float)
    yy = np.empty([0, 8], float)

# CSI DATA
    for f in sorted(glob.glob(pattern_x)):
        print("input_file_name=", f)
        tmp1 = pd.read_csv(f, header=None).values
        x2 = np.empty([0, WINDOW_SIZE, 180], float)

        # data import by slide window
        k = 0
        while k <= (len(tmp1) + 1 - 2 * WINDOW_SIZE):
            x = np.dstack(np.array(tmp1[k:k + WINDOW_SIZE, 1:181]).T)
            x2 = np.concatenate((x2, x), axis=0)
            k += SLIDE_SIZE
        xx = np.concatenate((xx, x2), axis=0)
    xx = xx.reshape(len(xx), -1)

# ANNOT. DATA
    for f in sorted(glob.glob(pattern_y)):
        print("annotation_file_name=", f)
        tmp2 = pd.read_csv(f, header=None).values

        # data import by slide window
        y = np.zeros(((len(tmp2) + 1 - 2 * WINDOW_SIZE) / SLIDE_SIZE + 1, 8))
        k = 0
        while k <= (len(tmp2) + 1 - 2 * WINDOW_SIZE):
            y_pre = pd.read_csv(f, header=None).loc[k:k + WINDOW_SIZE]
            num_labels = pd.value_counts(y_pre[0]).to_dict()
            # print num_labels

            if num_labels.get("bed", 0) > WINDOW_SIZE * THRESHOLD / 100:
                y[k / SLIDE_SIZE, :] = np.array([0, 1, 0, 0, 0, 0, 0, 0])
            elif num_labels.get("fall", 0) > WINDOW_SIZE * THRESHOLD / 100:
                y[k / SLIDE_SIZE, :] = np.array([0, 0, 1, 0, 0, 0, 0, 0])
            elif num_labels.get("walk", 0) > WINDOW_SIZE * THRESHOLD / 100:
                y[k / SLIDE_SIZE, :] = np.array([0, 0, 0, 1, 0, 0, 0, 0])
            elif num_labels.get("pickup", 0) > WINDOW_SIZE * THRESHOLD / 100:
                y[k / SLIDE_SIZE, :] = np.array([0, 0, 0, 0, 1, 0, 0, 0])
            elif num_labels.get("run", 0) > WINDOW_SIZE * THRESHOLD / 100:
                y[k / SLIDE_SIZE, :] = np.array([0, 0, 0, 0, 0, 1, 0, 0])
            elif num_labels.get("sitdown", 0) > WINDOW_SIZE * THRESHOLD / 100:
                y[k / SLIDE_SIZE, :] = np.array([0, 0, 0, 0, 0, 0, 1, 0])
            elif num_labels.get("standup", 0) > WINDOW_SIZE * THRESHOLD / 100:
                y[k / SLIDE_SIZE, :] = np.array([0, 0, 0, 0, 0, 0, 0, 1])
            else:
                y[k / SLIDE_SIZE, :] = np.array([2, 0, 0, 0, 0, 0, 0, 0])
            k += SLIDE_SIZE

        yy = np.concatenate((yy, y), axis=0)
    print(xx.shape, yy.shape)
    return (xx, yy)


LABELS = ["bed", "fall", "pickup", "run", "sitdown", "standup", "walk"]
INPUT_X_PATTERN = "161208_activity_data/input*{0}*.csv"
INPUT_Y_PATTERN = "161208_activity_data/annot*{0}*.csv"
OUTPUT_FOLDER = "input_files/"
OUTPUT_X_PATTERN = "input_files/xx_{0}_{1}_{2}.csv"
OUTPUT_Y_PATTERN = "input_files/yy_{0}_{1}_{2}.csv"

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

for label in LABELS:
    in_x_pattern = INPUT_X_PATTERN.format(label)
    in_y_pattern = INPUT_Y_PATTERN.format(label)
    outfile_x = OUTPUT_X_PATTERN.format(WINDOW_SIZE, THRESHOLD, label)
    outfile_y = OUTPUT_Y_PATTERN.format(WINDOW_SIZE, THRESHOLD, label)

    x, y = data_import(in_x_pattern, in_y_pattern)
    pd.DataFrame(x).to_csv(outfile_x, index=False)
    pd.DataFrame(y).to_csv(outfile_y, index=False)
    print(str(label) + "finish!")
