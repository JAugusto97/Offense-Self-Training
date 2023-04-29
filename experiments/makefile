download-datasets:
	gdown https://drive.google.com/drive/u/3/folders/1rGblqA0Wh0vhDFrjasMGvOYpyDiD3jVW -O ./ --folder 

augment:
	python augment.py
	
olid:
	python train.py olidv1 \
		--min_confidence_threshold 0.8 \
		--exp_name olid \
		--batch_size $(BATCH_SIZE) \
		--seed $(SEED)

convabuse:
	python train.py convabuse \
		--min_confidence_threshold 0.8 \
		--exp_name convabuse \
		--batch_size $(BATCH_SIZE) \
		--seed $(SEED)
