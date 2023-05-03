#!/bin.bash

BATCH_SIZE=128

sbatch run.slurm 42 $BATCH_SIZE backtranslation distilbert-base-cased
sbatch run.slurm 43 $BATCH_SIZE backtranslation distilbert-base-cased
sbatch run.slurm 44 $BATCH_SIZE backtranslation distilbert-base-cased 

sbatch run.slurm 42 $BATCH_SIZE "" distilbert-base-cased
sbatch run.slurm 43 $BATCH_SIZE "" distilbert-base-cased
sbatch run.slurm 44 $BATCH_SIZE "" distilbert-base-cased 

sbatch run.slurm 42 $BATCH_SIZE word_swap distilbert-base-cased
sbatch run.slurm 43 $BATCH_SIZE word_swap distilbert-base-cased
sbatch run.slurm 44 $BATCH_SIZE word_swap distilbert-base-cased 

sbatch run.slurm 42 $BATCH_SIZE synonym_substitution distilbert-base-cased
sbatch run.slurm 43 $BATCH_SIZE synonym_substitution distilbert-base-cased
sbatch run.slurm 44 $BATCH_SIZE synonym_substitution distilbert-base-cased 

sbatch run.slurm 42 $BATCH_SIZE backtranslation bert-base-cased 
sbatch run.slurm 43 $BATCH_SIZE backtranslation bert-base-cased 
sbatch run.slurm 44 $BATCH_SIZE backtranslation bert-base-cased 

sbatch run.slurm 42 $BATCH_SIZE word_swap bert-base-cased 
sbatch run.slurm 43 $BATCH_SIZE word_swap bert-base-cased 
sbatch run.slurm 44 $BATCH_SIZE word_swap bert-base-cased 

sbatch run.slurm 42 $BATCH_SIZE "" bert-base-cased 
sbatch run.slurm 43 $BATCH_SIZE "" bert-base-cased 
sbatch run.slurm 44 $BATCH_SIZE "" bert-base-cased 

sbatch run.slurm 42 $BATCH_SIZE synonym_substitution bert-base-cased 
sbatch run.slurm 43 $BATCH_SIZE synonym_substitution bert-base-cased 
sbatch run.slurm 44 $BATCH_SIZE synonym_substitution bert-base-cased 

sbatch run.slurm 42 $BATCH_SIZE "" roberta-base
sbatch run.slurm 43 $BATCH_SIZE "" roberta-base
sbatch run.slurm 44 $BATCH_SIZE "" roberta-base

sbatch run.slurm 42 $BATCH_SIZE word_swap roberta-base
sbatch run.slurm 43 $BATCH_SIZE word_swap roberta-base
sbatch run.slurm 44 $BATCH_SIZE word_swap roberta-base

sbatch run.slurm 42 $BATCH_SIZE synonym_substitution roberta-base
sbatch run.slurm 43 $BATCH_SIZE synonym_substitution roberta-base
sbatch run.slurm 44 $BATCH_SIZE synonym_substitution roberta-base

sbatch run.slurm 42 $BATCH_SIZE backtranslation roberta-base
sbatch run.slurm 43 $BATCH_SIZE backtranslation roberta-base
sbatch run.slurm 44 $BATCH_SIZE backtranslation roberta-base

sbatch run.slurm 42 $BATCH_SIZE "" roberta-large
sbatch run.slurm 43 $BATCH_SIZE "" roberta-large
sbatch run.slurm 44 $BATCH_SIZE "" roberta-large

sbatch run.slurm 42 $BATCH_SIZE word_swap roberta-large
sbatch run.slurm 43 $BATCH_SIZE word_swap roberta-large
sbatch run.slurm 44 $BATCH_SIZE word_swap roberta-large

sbatch run.slurm 42 $BATCH_SIZE synonym_substitution roberta-large
sbatch run.slurm 43 $BATCH_SIZE synonym_substitution roberta-large
sbatch run.slurm 44 $BATCH_SIZE synonym_substitution roberta-large

sbatch run.slurm 42 $BATCH_SIZE backtranslation roberta-large
sbatch run.slurm 43 $BATCH_SIZE backtranslation roberta-large
sbatch run.slurm 44 $BATCH_SIZE backtranslation roberta-large
