download-datasets:
	gdown https://drive.google.com/drive/folders/1PFVxeUSDiXdTE4wElkiSe3wajgAW2owB -O ./ --folder 

augment:
	python augment.py
	
olid:
	python train.py olidv1 \
		--min_confidence_threshold 0.8 \
		--exp_name olid \
		--batch_size $(BATCH_SIZE) \
		--seed $(SEED) \
		--augmentation_type $(AUGMENTATION_TYPE) \
		--pretrained_bert_name $(PRETRAINED_NAME)

convabuse:
	python train.py convabuse \
		--min_confidence_threshold 0.8 \
		--exp_name convabuse \
		--batch_size $(BATCH_SIZE) \
		--seed $(SEED) \
		--augmentation_type $(AUGMENTATION_TYPE) \
		--pretrained_bert_name $(PRETRAINED_NAME)

waseem:
	python train.py waseem \
		--min_confidence_threshold 0.8 \
		--exp_name waseem \
		--batch_size $(BATCH_SIZE) \
		--seed $(SEED) \
		--augmentation_type $(AUGMENTATION_TYPE) \
		--pretrained_bert_name $(PRETRAINED_NAME)

corruption-exp:
	python corruption_exp.py olidv1 \
		--exp_name olid \
		--batch_size $(BATCH_SIZE) \
		--seed $(SEED) \
		--augmentation_type $(AUGMENTATION_TYPE) \
		--pretrained_bert_name $(PRETRAINED_NAME)

	python corruption_exp.py convabuse \
		--exp_name convabuse \
		--batch_size $(BATCH_SIZE) \
		--seed $(SEED) \
		--augmentation_type $(AUGMENTATION_TYPE) \
		--pretrained_bert_name $(PRETRAINED_NAME)

	python corruption_exp.py waseem \
		--exp_name waseem \
		--batch_size $(BATCH_SIZE) \
		--seed $(SEED) \
		--augmentation_type $(AUGMENTATION_TYPE) \
		--pretrained_bert_name $(PRETRAINED_NAME)