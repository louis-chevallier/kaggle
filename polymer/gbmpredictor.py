import utillc
from utillc import *
import json

import pandas as pd
import os, sys, glob
import tqdm, re
import pickle
from collections import defaultdict
import hashlib
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import numpy as np

print_everything()

"""

with a neural decision tree :
 - joint multitarget possible
 - missing value in target possible ( backprop only for record where target is valid)

 - how to deal with missing values in X ??
   => for each variable having missing value, add an indicator variable ( present/not present)
      when the value is missing, put variable mean, random value ( along with indicator )
      DecisionTree does support missing value in X ! 
      (when computiong best split, for a given variable, consider only the rows that have a valid value)

note : catboost has multi target capability ( as opposed to lightgbm and xgboost )

"""


class GBMPredictor :
	def __init__(self, mdl) :
		self.mdl = mdl
		#EKOX(mdl)
		self.thresholds = {}
		for i, tsd in enumerate(self.mdl["tree_info"]) :
			tree = tsd["tree_structure"]
			self.parse(tree, "%03d" % i)
		EKOX(len(self.thresholds))
	def parse(self, tree, tag) :
		if "threshold" in tree :
			feat_idx = tree["split_feature"]
			threshold = tree["threshold"]
			self.thresholds[tag] = threshold
			self.parse(tree["left_child"], tag + "_l")
			self.parse(tree["right_child"], tag + "_r")
		else :
			value = tree["leaf_value"]
			return value

	def pred2(self, row, tree, tag="") :
		if "threshold" in tree :
			feat_idx = tree["split_feature"]
			threshold = tree["threshold"]
			t = row[feat_idx] <= threshold
			tt = t if tree["decision_type"] == "<=" else not t
			side = "left_child" if t else "right_child"
			return self.pred2(row, tree[side])
		else :
			value = tree["leaf_value"]
			return value
			
	def pred1(self, row, tree_info) :
		v = 0
		for i, tsd in enumerate(tree_info) :
			tree = tsd["tree_structure"]
			v += self.pred2(row, tree, "%03d" % i)
		return v

	def predict(self, x) :
		h, _ = x.shape
		p = np.zeros(h)
		for i, row in enumerate(x) :
			p[i] = self.pred1(row, self.mdl["tree_info"])
			
		return p



def main() :
	tasks = glob.glob("data/*")
	for t in tasks :
		files = glob.glob(t + "/*.csv")
		if len(files) == 0 :
			files = glob.glob(t + "/*.xlsx")
		EKOX(files)
		try :
			df = pd.read_csv(files[0])
		except:
			df = pd.read_excel(files[0])
		EKON(df.head())
		ddd = dd[t]
		df.drop(ddd[0], axis='columns')
		pred = df[dd[t][1]]
		train = df.drop([dd[t][1]], axis='columns')
		



		
		


if __name__ == "__main__" :
		main()
