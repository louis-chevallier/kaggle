empty :=
space1 := $(empty) $(empty)
space := $() $()
DATASET_FOLDER=/mnt/hd1/data/kaggle/datasets
COMMENT=$(subst $(space),_,$(shell git log -1 --pretty=%B)$(shell git log --pretty=format:'%h' -n 1))
DATE=$(shell date +%y-%m-%d-%R)
$(warning $(COMMENT))


start :
	CUDA_VISIBLE_DEVICES=0 python tree.py 2>&1 | tee trace-$(DATE).trc


start3 :
CUDA_VISIBLE_DEVICES=0 python polymer.py 2>&1 | tee trace-$(DATE).trc
#	CUDA_VISIBLE_DEVICES=0 python multi.py


start2 :
	python exam.py
start1 :
	CUDA_VISIBLE_DEVICES=0 python cylinder.py

conv :
	jupytext --to notebook polymer.py; mv polymer.ipynb upload/polymer_v0.ipynb
	cp  polymer.py upload
#	pip download utillc rdkit
#	mv *.whl upload

upload :
	-mkdir -p $(DATASET_FOLDER)/sources
	-mkdir -p $(DATASET_FOLDER)/wheels
	cp  dataset-metadata-sources.json $(DATASET_FOLDER)/sources/dataset-metadata.json
	cp  dataset-metadata-wheels.json $(DATASET_FOLDER)/wheels/dataset-metadata.json
	cp  *.py  $(DATASET_FOLDER)/sources
	(cd $(DATASET_FOLDER)/wheels; pip download utillc rdkit)
	kaggle datasets list -m

#	kaggle datasets status lmdm99/wheels-dataset
#	kaggle datasets init -p upload
	kaggle datasets version -m "$(COMMENT)" -p  $(DATASET_FOLDER)/sources -t -r zip
#	kaggle datasets create  -p  $(DATASET_FOLDER)/sources -t -r zip
#	kaggle datasets create  -p  $(DATASET_FOLDER)/wheels -t -r zip
#	kaggle datasets version -p  $(DATASET_FOLDER)/wheels -t -r zip
	kaggle datasets list -m
