import os
#!pip install utillc
#!pip install rdkit

lh = 'Localhost'
on_kaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', lh) != lh
import utillc
from utillc import *
print_everything()

EKOX(on_kaggle)
EKOX(utillc.__version__)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import NMF
from sklearn.impute import SimpleImputer
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split

import torch
import torchvision.transforms as transforms

do_deep=False

if do_deep :
	from transformers import BertTokenizerFast, BertModel
import os, sys
from tqdm import tqdm
import pandas as pd
import numpy as np

import seaborn
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold,KFold,StratifiedGroupKFold,GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD,PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error as mae
import pickle

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit import DataStructs
from rdkit import RDLogger	
RDLogger.DisableLog('rdApp.*')	
os.environ["TOKENIZERS_PARALLELISM"] = "false"



data_folder = "/kaggle/input/neurips-open-polymer-prediction-2025/" if on_kaggle else "/mnt/hd1/data/kaggle/polymer"

EKOX(data_folder)

EKOX(torch.__version__)
EKOX(torch.cuda.is_available())
EKOX(torch.version.cuda)
a=torch.rand(5,3).cuda()
EKOX(torch.cuda.get_device_properties(0))
EKOT('so far so good, torch works with cuda')


for dirname, _, filenames in os.walk(data_folder) : 
	for filename in filenames:
		EKON(os.path.join(dirname, filename))


deep_keys = []
if do_deep :
	checkpoint = 'unikei/bert-base-smiles'
	tokenizer = BertTokenizerFast.from_pretrained(checkpoint)

	todev = lambda m : m.cuda()
	bert_model = todev(BertModel.from_pretrained(checkpoint))


	example = 'O=C([C@@H](c1ccc(cc1)O)N)N[C@@H]1C(=O)N2[C@@H]1SC([C@@H]2C(=O)O)(C)C'
	tokens = tokenizer(example, return_tensors='pt')
	tokens = dict([ (k, todev(v)) for k,v in tokens.items()])
	predictions = bert_model(**tokens)
	deep_keys = predictions.keys()
	EKOX(deep_keys)
	EKON([ predictions[k].shape for k in deep_keys])

j = lambda p : os.path.join(data_folder, p)
train = pd.read_csv(j("train.csv"))
test = pd.read_csv(j("test.csv"))
EKOX(len(test))
targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
EKOX(test.shape)

EKON(train.shape)
EKON(train.isnull().sum())
EKON(train.isin([np.inf, -np.inf]).sum())
def compute_all_descriptors(smiles):
	mol = Chem.MolFromSmiles(smiles)
	if mol is None:
		return [None] * len(desc_names)
	return [desc[1](mol) for desc in Descriptors.descList]

def compute_deep_desc(smiles) :
	tokens = tokenizer(smiles, return_tensors='pt')
	tokens = dict([ (k, todev(v)) for k,v in tokens.items()])
	predictions = bert_model(**tokens)
	dc = lambda x : x.detach().cpu().numpy()
	res = [ dc(predictions[k]) for k in deep_keys]
	return res

desc_names = [desc[0] for desc in Descriptors.descList]

tqdm = lambda x : x

def comp(ss, fn) :
	if os.path.exists(fn) :
		return pd.read_pickle(fn)
	else :
		descriptors = [compute_all_descriptors(smi) for smi in tqdm(ss['SMILES'].to_list()) ]
		descriptors = pd.DataFrame(descriptors, columns=desc_names)
		res = pd.concat([ss,descriptors],axis=1)
		
		if do_deep :
			deep_desc =	 [compute_deep_desc(smi) for smi in tqdm(ss['SMILES'].to_list()) ]
			deep_desc = pd.DataFrame(deep_desc, columns=deep_keys)
			res = pd.concat([res, deep_desc],axis=1)
			
		res.to_pickle(fn)
		return res

train = comp(train, "train.pckl")
test = comp(test, "test.pckl")

sz = lambda x : x.shape[0] * x.shape[1]
EKON(train.isin([np.inf, -np.inf]).sum() / sz(train))
EKON(train.isnull().sum() / sz(train))


EKOX(train.shape)
EKOX(train.columns.shape)
#EKOX(train.columns)

train_data_df = train.drop(["SMILES"] + list(deep_keys), axis='columns')
train_data = train_data_df.to_numpy()
EKOX(train_data_df.describe())
train_data_X_df = train.drop(["SMILES"] + targets + list(deep_keys), axis='columns')

train_data_X = train_data_X_df.to_numpy()
EKOX(train_data_X_df.describe())
EKOX(train[targets].describe())

train_data_y = train[targets].to_numpy()


def check(d, cols) :
	EKOX(cols)
	EKOX(d.shape)
	EKON(np.isnan(d).sum() / sz(d))
	EKON(np.isinf(d).sum() / sz(d))
	EKON((d <= 0).sum() / sz(d))
	EKON((d > 0).sum() / sz(d))

	# suppress inf
	d = np.clip(d, -20, 20) 

	# suppress columns full of nan
	nn = ~np.all(np.isnan(d), axis=0)
	EKOX(nn.shape)
	ddd, lll = d[:,nn], list(cols[nn])
	return ddd, lll, pd.DataFrame(ddd, columns=lll)


def lgb_kfold(train_df, test_df, target, feats, folds):	   
	params = {	  
		 'objective' : 'mae',#'binary', 
		 'metric' : 'mae', 
		 'num_leaves': 100,
		 'min_data_in_leaf': 30,#30,
		 'learning_rate': 0.01,
		 'max_depth': -1,
		 'max_bin': 256,
		 'boosting': 'gbdt',
		 'feature_fraction': 0.7,
		 'bagging_freq': 1,
		 'bagging_fraction': 0.7,
		 'bagging_seed': 42,
		 "lambda_l1":1,
		 "lambda_l2":1,
		 'verbosity': -100,
		 'num_boost_round' : 20000,
		 'device_type' : 'cpu'		  
	}	   

	EKON(train_df.shape, target)
	
	oof_preds = np.zeros(train_df.shape[0])
	sub_preds = np.zeros(test_df.shape[0])
	cv_list = []
	df_importances = pd.DataFrame()

	for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, train_df[target])):	  
		EKON('n_fold:',n_fold, target)

		train_x = train_df[feats].iloc[train_idx].values
		train_y = train_df[target].iloc[train_idx].values
		valid_x = train_df[feats].iloc[valid_idx].values
		valid_y = train_df[target].iloc[valid_idx].values
		test_x = test_df[feats]
		EKON(train_x.shape)
		EKON(valid_x.shape)	  
		EKON(test_x.shape)	
		EKOX(train_y.shape)
		dtrain = lgb.Dataset(train_x, label=train_y, )
		dval = lgb.Dataset(valid_x, label=valid_y, reference=dtrain, ) 
		callbacks = [
		lgb.log_evaluation(period=1000,),
		lgb.early_stopping(300)	   
		]
		bst = lgb.train(params, dtrain,valid_sets=[dval,dtrain],callbacks=callbacks,	) 

		#---------- feature_importances ---------#
		feature_importances = sorted(zip(feats, bst.feature_importance('gain')),key=lambda x: x[1], reverse=True)#[:100]
		#for f in feature_importances[:30]: EKON (f)	   
			
		new_feats = []
		importances = []
		for f in feature_importances:
			new_feats.append(f[0])
			importances.append(f[1])
		df_importance = pd.DataFrame()
		df_importance['feature'] = new_feats
		df_importance['importance'] = importances
		df_importance['fold'] = n_fold
		
		oof_preds[valid_idx] = bst.predict(valid_x, num_iteration=bst.best_iteration)
		# oof_cv = rmse(valid_y,  oof_preds[valid_idx])
		# cv_list.append(oof_cv)
		# EKON (cv_list)
		
		sub_preds += bst.predict(test_x, num_iteration=bst.best_iteration) / n_splits
		
		#bst.save_model(model_path+'lgb_fold_' + str(n_fold) + '.txt', num_iteration=bst.best_iteration)	 

		df_importances = pd.concat([df_importances,df_importance])	  
	EKON(target, oof_preds.shape, train_df[target].shape)	
	cv = mae(train_df[target],	oof_preds)
	EKON(cv)	
	return oof_preds,sub_preds




EKOT("X");
train_data_X, cols_X, train_data_X_df = check(train_data_X, train_data_X_df.columns)
EKOT("y")
train_data_y, cols_y, train_data_y_df = check(train_data_y, train[targets].columns)

EKOX(len(cols_X))
EKOX(train_data_X.shape)

EKON(train_data_X_df.describe())
EKON(train_data_y_df.describe())
EKO()

scaler_X = StandardScaler()
scaler_y = StandardScaler()
h,w = train_data_X.shape
train_data_X_scaled = scaler_X.fit_transform(train_data_X)
train_data_y_scaled = scaler_y.fit_transform(train_data_y)

EKOX(train_data_y.shape)
EKOT("x")
_, _, train_data_X_scaled_df = check(train_data_X_scaled, train_data_X_df.columns)
EKOT("y")
_, _, train_data_y_scaled_df = check(train_data_y_scaled, train_data_y_df.columns)


fillnanXy = False
if fillnanXy :
		imp = IterativeImputer(max_iter=10, random_state=0)
		_, ww = train_data_y_scaled.shape
		EKOX(ww)
		a = np.hstack((train_data_X_scaled, train_data_y_scaled))
		ai = imp.fit_transform(a)
		train_data_y_scaled = ai[:, -ww:]
		EKOX(train_data_y_scaled.shape)

fillnanX = True
if fillnanX :
		imp = SimpleImputer(missing_values=np.nan, strategy='mean')		
		a = np.asarray(train_data_X_scaled)
		train_data_X_scaled = imp.fit_transform(a)
		
EKOT("x")
_, _, train_data_X_scaled_df = check(train_data_X_scaled, train_data_X_df.columns)
EKOT("y")
_, _, train_data_y_scaled_df = check(train_data_y_scaled, train_data_y_df.columns)

EKON(train_data_X_scaled_df.describe())
EKON(train_data_y_scaled_df.describe())
		
_, _, train_data_y_scaled_df = check(train_data_y_scaled, train_data_y_df.columns)		



		
n_splits = 5
seed = 817
folds = KFold(n_splits=n_splits, random_state=seed, shuffle=True)

feats = cols_X
EKOX(train_data_X_scaled_df.shape)
EKOX(train_data_y_scaled_df.shape)
train_df = pd.concat((train_data_X_scaled_df, train_data_y_scaled_df), axis=1)

EKOX(len(test))

test_scaled = test[cols_X]
test_scaled = pd.DataFrame(scaler_X.transform(test_scaled.values), columns=cols_X)


EKON(train_df.describe())


X_train, X_valid, y_train, y_valid = train_test_split(train_df.drop(targets, axis='columns'),
													  train_df[targets],
													  test_size=1/6, random_state=42)

class Polymer(Dataset):
	def __init__(self, X, y=None, transform=None):
		self.X = X
		self.y = y
		self.transform = transform
		
	def __len__(self):
		return len(self.X.index)
	
	def __getitem__(self, index):
		image = self.X.iloc[index, ].values.astype(np.float32)
		if self.y is not None:
			yy = self.y.iloc[index]
			return np.asarray(image), np.asarray(yy)
		else:
			return image

transform=transforms.Compose([
	transforms.ToTensor()
])


_, Dx = X_train.shape
_, Dy = y_train.shape

train_dataset = Polymer(X=X_train, y=y_train, transform=transform)
valid_dataset = Polymer(X=X_valid, y=y_valid, transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=12, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=12, shuffle=False)

class MLP(nn.Module):
	def __init__(self):
		super(MLP, self).__init__()
		self.layers = nn.Sequential(
			nn.Linear(Dx, 100),
			nn.ReLU(),
			nn.Linear(100, 100),
			nn.ReLU(),
			nn.Linear(100, 100),
			nn.ReLU(),
			nn.Linear(100, 100),
			nn.ReLU(),
			nn.Linear(100, 100),
			nn.ReLU(),
			nn.Linear(100, 100),
			nn.ReLU(),
			nn.Linear(100, 100),
			nn.ReLU(),
			nn.Linear(100, 100),
			nn.ReLU(),
			nn.Linear(100, 100),
			nn.ReLU(),
			nn.Linear(100, Dy)
		)
		
	def forward(self, x):
		x = x.view(x.size(0), -1)
		x = self.layers(x)
		return x

model = MLP()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def my_loss(output, target) :
		m = torch.isnan(target)
		o, t = output[~m], target[~m]
		return torch.nn.functional.l1_loss(o, t)



optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

mean_train_losses = []
mean_valid_losses = []
valid_acc_list = []
epochs = 15


def deep() :
		for epoch in range(epochs):
			model.train()

			train_losses = []
			valid_losses = []
			for i, (images, labels) in enumerate(train_loader):
				optimizer.zero_grad()
				outputs = model(images)
				loss = loss_fn(outputs, labels)
				loss = my_loss(outputs, labels)
				loss.backward()
				optimizer.step()
				train_losses.append(loss.item())


			model.eval()
			correct = 0
			total = 0
			with torch.no_grad():
				for i, (images, labels) in enumerate(valid_loader):
					outputs = model(images)
					loss = my_loss(outputs, labels)
					valid_losses.append(loss.item())

					m = torch.isnan(labels)
					o, t = outputs[~m], labels[~m]
					"""
					EKOX(outputs)
					EKOX(labels)
					EKOX(m)
					EKOX(o)
					EKOX(t)
					EKOX(o.shape)
					EKOX(t.shape)
					EKOX(outputs.shape)
					EKOX(labels.shape)
					EKOX(torch.abs(o-t).shape)
					EKOX((o-t).abs().mean(dim=0).shape)
					sys.exit(0)
					"""
					

			mean_train_losses.append(np.mean(train_losses))
			mean_valid_losses.append(np.mean(valid_losses))
			EKON(epoch, epoch+1, np.mean(train_losses), np.mean(valid_losses))
		EKO()
		fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))
		ax1.plot(mean_train_losses, label='train')
		ax1.plot(mean_valid_losses, label='valid')
		lines, labels = ax1.get_legend_handles_labels()
		ax1.legend(lines, labels, loc='best')

		ax2.plot(valid_acc_list, label='valid acc')
		ax2.legend()
		plt.show()


def main() :

		deep()

		
		for t in targets:
			EKOT(t)
			if len(test)< 2 :
				test[t] = 0
			else:
				train_df_f = train_df[train[t].notnull()]
				oof_preds,sub_preds = lgb_kfold(train_df_f, test_scaled, t, cols_X, folds)
				test[t] = sub_preds

		EKOX(test.columns)
		final_vals = scaler_y.inverse_transform(test[targets].values)

		t1 = test['id']
		t2 = pd.DataFrame(final_vals, columns=targets)
		res = pd.concat((t1, t2), axis=1)

		# Generate submission

		res[['id','Tg', 'FFV', 'Tc', 'Density', 'Rg']].to_csv('submission.csv',index=False)
		EKOT("done")

if __name__ == "__main__" :
		main()
		
