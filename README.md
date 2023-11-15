# Offense-Self-Training 
## Abstract
Online social media is rife with offensive and hateful comments, prompting the need for their automatic detection given the sheer amount of posts created every second. Creating high-quality human-labelled datasets for this task is difficult and costly, especially because non-offensive posts are significantly more frequent than offensive ones. However, unlabelled data is abundant, easier, and cheaper to obtain. In this scenario, self-training methods make use of weakly-labelled examples to increase the amount of training data. Recent "noisy" self-training approaches incorporate data augmentation techniques to ensure prediction consistency and increase robustness against noisy data and adversarial attacks. In this work, we experiment with default and noisy self-training using three different textual data augmentation techniques across five different pretrained BERT architectures varying in size. We evaluate our experiments on two offensive/hate-speech datasets and demonstrate that (i) self-training consistently improves performance regardless of model size, resulting in up to +1.5% F1-macro on both datasets, and (ii) noisy self-training with textual data augmentations, despite being successfully applied in similar settings, decreases performance on offensive and hate-speech domains when compared to the default self-train method, even with state-of-the-art augmentations such as backtranslation. Finally, we discuss future research ideas to mitigate the issues found with this work.



## How to reproduce the experiments:
* Make sure to use Python 3.10.4. 
* Using a decent GPU is heavily encouraged.
0. (Optional) Installing dependencies with conda:
    >conda create -n selftrain python==3.10.4

    >conda activate selftrain
1. Install python dependencies.
    >pip install -r requirements.txt
2. Move your current directory to the experiments folder.
    >cd experiments
3. Download the data sets.
    >make download-datasets
4. Run one of the experiments available at the makefile with its correspoding parameters.


## Citing
@inproceedings{leite-etal-2023-noisy,
    title = "Noisy Self-Training with Data Augmentations for Offensive and Hate Speech Detection Tasks",
    author = "Leite, Jo{\~a}o  and
      Scarton, Carolina  and
      Silva, Diego",
    editor = "Mitkov, Ruslan  and
      Angelova, Galia",
    booktitle = "Proceedings of the 14th International Conference on Recent Advances in Natural Language Processing",
    month = sep,
    year = "2023",
    address = "Varna, Bulgaria",
    publisher = "INCOMA Ltd., Shoumen, Bulgaria",
    url = "https://aclanthology.org/2023.ranlp-1.68",
    pages = "631--640",
    abstract = "Online social media is rife with offensive and hateful comments, prompting the need for their automatic detection given the sheer amount of posts created every second. Creating high-quality human-labelled datasets for this task is difficult and costly, especially because non-offensive posts are significantly more frequent than offensive ones. However, unlabelled data is abundant, easier, and cheaper to obtain. In this scenario, self-training methods, using weakly-labelled examples to increase the amount of training data, can be employed. Recent {``}noisy{''} self-training approaches incorporate data augmentation techniques to ensure prediction consistency and increase robustness against noisy data and adversarial attacks. In this paper, we experiment with default and noisy self-training using three different textual data augmentation techniques across five different pre-trained BERT architectures varying in size. We evaluate our experiments on two offensive/hate-speech datasets and demonstrate that (i) self-training consistently improves performance regardless of model size, resulting in up to +1.5{\%} F1-macro on both datasets, and (ii) noisy self-training with textual data augmentations, despite being successfully applied in similar settings, decreases performance on offensive and hate-speech domains when compared to the default method, even with state-of-the-art augmentations such as backtranslation.",
}
