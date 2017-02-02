from __future__ import print_function
import gzip
import os
import numpy as np,numpy
import csv
import glob

class DataSet(object):
  def __init__(self, images, labels, fake_data=False):
    assert images.shape[0] == labels.shape[0], (
          "images.shape: %s labels.shape: %s" % (images.shape,
                                                 labels.shape))
    self._num_examples = images.shape[0]
    images = images.reshape(images.shape[0],
                              images.shape[1] * images.shape[2])

    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
  @property
  def images(self):
    return self._images
  @property
  def labels(self):
    return self._labels
  @property
  def num_examples(self):
    return self._num_examples
  @property
  def epochs_completed(self):
    return self._epochs_completed
  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]

def csv_import( arg1 ):
	x_dic = {}
	y_dic = {}
	print("csv file importing...")

	for i in ["bed", "fall", "pickup", "run", "sitdown", "standup", "walk"]:

	        xx = np.array([[ float(elm) for elm in v] for v in csv.reader(open("./input_files/xx_50_60_" + str(i) + ".csv","r"))])
		yy = np.array([[ float(elm) for elm in v] for v in csv.reader(open("./input_files/yy_50_60_" + str(i) + ".csv","r"))])

		# eliminate the NoActivity Data
		rows, cols = np.where(yy>0)
		xx = np.delete(xx, rows[ np.where(cols==0)],0)
		yy = np.delete(yy, rows[ np.where(cols==0)],0)

        	xx = xx.reshape(len(xx),50,180)

		# Choise of data type
		# Amplitude only (args = 1)
		if arg1 == 1:
	        	xx = xx[:,:,:90]

		# Phase only (args = 2)
		elif arg1  == 2:
			xx = xx[:,:,90:]

		# Amplitude + Phase
		else:
			xx = xx[:,:,:]

		x_dic[str(i)] = xx
		y_dic[str(i)] = yy

	return x_dic["bed"], x_dic["fall"], x_dic["pickup"], x_dic["run"], x_dic["sitdown"], x_dic["standup"], x_dic["walk"], \
		y_dic["bed"], y_dic["fall"], y_dic["pickup"], y_dic["run"], y_dic["sitdown"], y_dic["standup"], y_dic["walk"]

