
DATASET_FOLDER=/mnt/hd1/data/kaggle-dataset

start :
	CUDA_VISIBLE_DEVICES=0 python polymer.py
#	CUDA_VISIBLE_DEVICES=0 python multi.py

conv :
	jupytext --to notebook polymer.py; mv polymer.ipynb upload/polymer_v0.ipynb
	cp  polymer.py upload
#	pip download utillc rdkit
#	mv *.whl upload



upload : conv
	kaggle datasets list -m
	kaggle datasets status lmdm99/wheels-dataset
#	kaggle datasets init -p upload
	-kaggle datasets version -m "Updated source" -p upload -t -r skip -d
	kaggle datasets list -m
