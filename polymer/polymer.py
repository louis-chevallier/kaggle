from utillc import *
print_everything()


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from transformers import BertTokenizerFast, BertModel
import os, sys, tqdm
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold,KFold,StratifiedGroupKFold,GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD,PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit import DataStructs
from rdkit import RDLogger  
RDLogger.DisableLog('rdApp.*')  
os.environ["TOKENIZERS_PARALLELISM"] = "false"



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

data_folder = "/kaggle/input/neurips-open-polymer-prediction-2025/"
data_folder = "/kaggle/input"
data_folder = "/mnt/hd1/data/kaggle/polymer"


for dirname, _, filenames in os.walk(data_folder) : 
    for filename in filenames:
        EKON(os.path.join(dirname, filename))



checkpoint = 'unikei/bert-base-smiles'
tokenizer = BertTokenizerFast.from_pretrained(checkpoint)

todev = lambda m : m.cuda()
bert_model = todev(BertModel.from_pretrained(checkpoint))


example = 'O=C([C@@H](c1ccc(cc1)O)N)N[C@@H]1C(=O)N2[C@@H]1SC([C@@H]2C(=O)O)(C)C'
tokens = tokenizer(example, return_tensors='pt')
tokens = dict([ (k, todev(v)) for k,v in tokens.items()])
predictions = bert_model(**tokens)
keys = predictions.keys()
EKOX(keys)
EKON([ predictions[k].shape for k in keys])

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
	res = [ dc(predictions[k]) for k in keys]
	return res

desc_names = [desc[0] for desc in Descriptors.descList]

def comp(ss, fn) :
	if os.path.exists(fn) :
		return pd.read_pickle(fn)
	else :
		descriptors = [compute_all_descriptors(smi) for smi in tqdm.tqdm(ss['SMILES'].to_list()) ]
		descriptors = pd.DataFrame(descriptors, columns=desc_names)

		deep_desc =  [compute_deep_desc(smi) for smi in tqdm.tqdm(ss['SMILES'].to_list()) ]
		deep_desc = pd.DataFrame(deep_desc, columns=keys)
		
		res = pd.concat([ss,descriptors, deep_desc],axis=1)
		res.to_pickle(fn)
		return res

train = comp(train, "train.pckl")
test = comp(test, "test.pckl")
EKON(train.isin([np.inf, -np.inf]).sum())
EKON(train.isnull().sum())
EKOX(train.shape)
EKOX(train.columns.shape)
EKOX(train.columns)
