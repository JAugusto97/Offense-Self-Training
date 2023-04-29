download-datasets:
	gdown https://drive.google.com/drive/folders/1PFVxeUSDiXdTE4wElkiSe3wajgAW2owB -O ./ --folder 

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
