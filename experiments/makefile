download-datasets:
	gdown https://drive.google.com/drive/u/3/folders/1rGblqA0Wh0vhDFrjasMGvOYpyDiD3jVW -O ./ --folder 

olid-default:
	python run.py olidv1 \
		--min_confidence_threshold 0.9 \
		--exp_name olidv1-default

olid-ns:
	python run.py olidv1 \
		--min_confidence_threshold 0.9 \
		--use_augmentation \
		--classifier_dropout 0.10 \
		--attention_dropout 0.10 \
		--increase_attention_dropout_amount 0.15 \
		--increase_classifier_dropout_amount 0.15 \
		--exp_name olidv1-ns

convabuse-default:
	python run.py convabuse \
		--min_confidence_threshold 0.8 \
		--exp_name convabuse-default

convabuse-ns:
	python run.py convabuse \
		--min_confidence_threshold 0.8 \
		--use_augmentation \
		--classifier_dropout 0.10 \
		--attention_dropout 0.10 \
		--increase_attention_dropout_amount 0.15
		--increase_classifier_dropout_amount 0.15
		--exp_name convabuse-ns

mhs-default:
	python run.py mhs \
		--min_confidence_threshold 0.8 \
		--exp_name mhs-default

mhs-ns:
	python run.py mhs \
		--min_confidence_threshold 0.8 \
		--use_augmentation \
		--classifier_dropout 0.10 \
		--attention_dropout 0.10 \
		--increase_attention_dropout_amount 0.15
		--increase_classifier_dropout_amount 0.15
		--exp_name mhs-ns