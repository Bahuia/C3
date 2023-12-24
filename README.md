Codes for NeurIPS 2023 Paper [**_Parameterizing Context: Unleashing the Power of Parameter-Efficient Fine-Tuning and In-Context Tuning for Continual Table Semantic Parsing_**](https://openreview.net/forum?id=B01uiWhjpc)

In this paper, we introduce a novel approach for continual table semantic parsing, integrating parameter-efficient fine-tuning (PEFT) and in-context tuning (ICT). We address the common challenges of overfitting and catastrophic forgetting in training parsers with limited examples. Our method employs a task-adaptive PEFT framework, freezing the model backbone and fine-tuning prompts to prevent forgetting. Additionally, we use a teacher-student framework with ICT to assist in few-shot learning and contextual understanding. Our experimental results on benchmarks demonstrate the superiority of our method over existing few-shot and continual learning approaches.

![image](https://github.com/Bahuia/C3/assets/71223470/796e61b0-815c-4d17-a25b-4f1224d627e4)

# Quick Start

## 1. Dataset Preparation

Our task splits are based on the *Spider* and *WikiSQL* dataset, and you will need to download them cause the experiments' evaluation requires the database file.

### 1.1 Dataset Download

You can get the original Spider dataset [here](https://drive.google.com/uc?export=download&id=1TqleXec_OykOYFREKKtschzY29dUcVAQ), and the original WikiSQL dataset [here](https://github.com/salesforce/WikiSQL/blob/master/data.tar.bz2).

### 1.2 Dataset Placement

After unzipping the above files, you may place them in `c3/datasets`, and the directory should look like:

```
c3
├─datasets
|	├─spider
|	|	├─database
|	|	├─dev.json
|	|	├─dev_gold.sql
|	|	├─README.txt
|	|	...
|	└─wikisql
|		├─dev.db
|       ├─dev.jsonl
|       ├─dev.tables.jsonl
|       ...
├─train
|
...
```

## 2. PLM Preparation

The backbone model of the proposed method is T5. Specifically,  you can get the [t5-large-lm-adapt](https://huggingface.co/google/t5-large-lm-adapt) and [t5-base-lm-adapt](https://huggingface.co/google/t5-base-lm-adapt) checkpoint from Hugging Face Hub. To achieve best performance, we recommend to use the large version of T5.

The checkpoint should be placed in `c2/train/plms`, the directory should look like:

```
c3
├─ datasets
|
├─train
|	├─ckpt
|	├─data_train
|	├─plms
|	|	└─t5-large-lm-adapt
|	|		├─config.json
|	|		├─tokenizer.json
|	|		├─pytorch_model.bin
|	|		...
|	├─train_utils
|	├─trainer
|	...
|
...
```

## 3. Environment Setup

You may install the packages mentioned in c2/requirements.txt to run the experiments. 

Additionally, you will need to download punkt for nltk to run evaluation:

```shell
python
>>> import nltk
>>> nltk.download('punkt')
```

  ## 4. Training & Evaluation

**We‘ve provided the training logs and evaluation logs of the below experiments in `c3/train/ckpt/logs` for reference.**

Our experiments can be done on one single GPU with 24GB memory or higher (e.g. RTX 3090/RTX 4090/A100).

Noted that the full training process includes the training of the teacher model (GPT-3.5 dose not require training) to get the output distribution (logits) from the teacher model. To make it easier for you to recover our results, we've provided the processed logits from T5 and GPT-3.5 (in `c3/train/ckpt/spider_perm_1/teacher_logits` & `c3/train/ckpt/combine1_perm_1/teacher_logits`) so that you don't have to train the teacher model. If you'd like to get the logits by yourself, you may need to run the shell command provided in 5 to train a teacher and process the logits.

### 4.1 T5-Large as student & teacher

To achieve the beast performance mentioned in the paper, which takes T5-Large as both the student and the teacher, you can run

```shell
# Spider-Stream
python train/main.py --cuda_visible_devices 0 --do_train true --do_cl_eval true --predictor_backbone_plm t5-large-lm-adapt --predictor_batch_size 1 --predictor_gradient_accumulation_steps 12 --dataset_type spider --dataset spider_perm_1 --first_task_id 0 --last_task_id 10 --task_num 11 --teacher true --teacher_plm t5-large-lm-adapt --teacher_with_context True --task_adaptation_with_teacher_logits false
# Combined-Stream
python train/main.py --cuda_visible_devices 0 --do_train true --do_cl_eval true --predictor_backbone_plm t5-large-lm-adapt --predictor_batch_size 1 --predictor_gradient_accumulation_steps 12 --dataset_type combine --dataset combine1_perm_1 --first_task_id 0 --last_task_id 6 --task_num 7 --teacher true --teacher_plm t5-large-lm-adapt --teacher_with_context True --task_adaptation_with_teacher_logits false
```

and you should get the results in Table 2 (evaluation is automatically done after training):

| Stream          | TA (%) | EA (%) |
| --------------- | ------ | ------ |
| Spider-Stream   | 70.7   | 68.9   |
| Combined-Stream | 69.0   | 71.2   |

### 4.2 T5-Large as student & text-davinci-003 as teacher

The raw output from text-davinci-003 we get through OpenAI's API has been placed in `c3/train/data_train/spider/spider_gpt_perm_1/raw` &  `c3/train/data_train/combine/combine1_gpt_perm_1/raw`, including the output sequences and logits, check it if you like.

You can also try to take T5-Large as the student and GPT-3.5 (text-davinci-003) as the teacher by running

```shell
# Spider-Stream
python train/main.py --cuda_visible_devices 0 --do_train true --do_cl_eval true --predictor_backbone_plm t5-large-lm-adapt --predictor_batch_size 1 --predictor_gradient_accumulation_steps 12 --dataset_type spider --dataset spider_perm_1 --first_task_id 0 --last_task_id 10 --task_num 11 --teacher true --teacher_plm gpt --teacher_with_context true --task_adaptation_with_teacher_logits false
# Combine-Stream
python train/main.py --cuda_visible_devices 0 --do_train true --do_cl_eval true --predictor_backbone_plm t5-large-lm-adapt --predictor_batch_size 1 --predictor_gradient_accumulation_steps 12 --dataset_type combine --dataset combine1_perm_1 --first_task_id 0 --last_task_id 6 --task_num 7 --teacher true --teacher_plm gpt --teacher_with_context true --task_adaptation_with_teacher_logits false
```

and you should get the results in Table 3, which is:

| Stream          | TA (%) | EA (%) |
| --------------- | ------ | ------ |
| Spider-Stream   | 71.3   | 69.6   |
| Combined-Stream | 67.6   | 70.0   |

## 5. All Experiments

Shell commands for all experiments.

### 5.1 Finetune T5

+ T5 fine-tuning experiments (bare finetune & multi-task)

```shell
# combine1_perm_1
python train/finetune_main.py --cuda_visible_devices 0 --backbone_plm t5-base-lm-adapt --per_device_train_batch_size 6 --per_device_eval_batch_size 6 --gradient_accumulation_steps 2 --dataset_type combine --dataset combine1_perm_1 --task_num 7 --first_task_id=0 --last_task_id=6 --do_cl_eval true
# combine1_perm_2
python train/finetune_main.py --cuda_visible_devices 2 --backbone_plm t5-base-lm-adapt --per_device_train_batch_size 6 --per_device_eval_batch_size 6 --gradient_accumulation_steps 2 --dataset_type combine --dataset combine1_perm_2 --task_num 7 --first_task_id=0 --last_task_id=6 --do_cl_eval true
# combine1_perm_3
python train/finetune_main.py --cuda_visible_devices 0 --backbone_plm t5-base-lm-adapt --per_device_train_batch_size 6 --per_device_eval_batch_size 6 --gradient_accumulation_steps 2 --dataset_type combine --dataset combine1_perm_3 --task_num 7 --first_task_id=0 --last_task_id=6 --do_cl_eval true

# spider_perm_1
python train/finetune_main.py --cuda_visible_devices 0 --backbone_plm t5-base-lm-adapt --per_device_train_batch_size 6 --per_device_eval_batch_size 6 --gradient_accumulation_steps 2 --dataset_type spider --dataset spider_perm_1 --task_num 11 --first_task_id=0 --last_task_id=10 --do_cl_eval true
# spider_perm_2
python train/finetune_main.py --cuda_visible_devices 2 --backbone_plm t5-base-lm-adapt --per_device_train_batch_size 6 --per_device_eval_batch_size 6 --gradient_accumulation_steps 2 --dataset_type spider --dataset spider_perm_2 --task_num 11 --first_task_id=0 --last_task_id=10 --do_cl_eval true
# spider_perm_3
python train/finetune_main.py --cuda_visible_devices 2 --backbone_plm t5-base-lm-adapt --per_device_train_batch_size 6 --per_device_eval_batch_size 6 --gradient_accumulation_steps 2 --dataset_type spider --dataset spider_perm_3 --task_num 11 --first_task_id=0 --last_task_id=10 --do_cl_eval true

# combine1_multi_perm_1
python train/finetune_main.py --cuda_visible_devices 0 --backbone_plm t5-base-lm-adapt --per_device_train_batch_size 6 --per_device_eval_batch_size 6 --gradient_accumulation_steps 2 --dataset_type combine --dataset combine1_multi_perm_1 --task_num 7 --first_task_id=0 --last_task_id=6 --do_cl_eval true
# combine1_multi_perm_2
python train/finetune_main.py --cuda_visible_devices 2 --backbone_plm t5-base-lm-adapt --per_device_train_batch_size 6 --per_device_eval_batch_size 6 --gradient_accumulation_steps 2 --dataset_type combine --dataset combine1_multi_perm_2 --task_num 7 --first_task_id=0 --last_task_id=6 --do_cl_eval true
# combine1_multi_perm_3
python train/finetune_main.py --cuda_visible_devices 0 --backbone_plm t5-base-lm-adapt --per_device_train_batch_size 6 --per_device_eval_batch_size 6 --gradient_accumulation_steps 2 --dataset_type combine --dataset combine1_multi_perm_3 --task_num 7 --first_task_id=0 --last_task_id=6 --do_cl_eval true

# spider_multi_perm_1
python train/finetune_main.py --cuda_visible_devices 0 --backbone_plm t5-base-lm-adapt --per_device_train_batch_size 6 --per_device_eval_batch_size 6 --gradient_accumulation_steps 2 --dataset_type spider --dataset spider_multi_perm_1 --task_num 11 --first_task_id=0 --last_task_id=10 --do_cl_eval true
# spider_multi_perm_2
python train/finetune_main.py --cuda_visible_devices 2 --backbone_plm t5-base-lm-adapt --per_device_train_batch_size 6 --per_device_eval_batch_size 6 --gradient_accumulation_steps 2 --dataset_type spider --dataset spider_multi_perm_2 --task_num 11 --first_task_id=0 --last_task_id=10 --do_cl_eval true
# spider_multi_perm_3
python train/finetune_main.py --cuda_visible_devices 2 --backbone_plm t5-base-lm-adapt --per_device_train_batch_size 6 --per_device_eval_batch_size 6 --gradient_accumulation_steps 2 --dataset_type spider --dataset spider_multi_perm_3 --task_num 11 --first_task_id=0 --last_task_id=10 --do_cl_eval true
```

### 5.2 Ablation Study

+ w/o teacher

```shell
# combine1_perm_1
python train/main.py --cuda_visible_devices 0 --do_train true --predictor_backbone_plm t5-base-lm-adapt --predictor_batch_size 4 --predictor_gradient_accumulation_steps 3 --dataset_type combine --dataset combine1_perm_1 --first_task_id 0 --last_task_id 6 --task_num 7 --teacher false
# spider_perm_1
python train/main.py --cuda_visible_devices 0 --do_train true --predictor_backbone_plm t5-base-lm-adapt --predictor_batch_size 4 --predictor_gradient_accumulation_steps 3 --dataset_type combine --dataset combine1_perm_1 --first_task_id 0 --last_task_id 10 --task_num 11 --teacher false
```

+ teacher w/o contexts

```shell
# spider_perm_1
python train/main.py --cuda_visible_devices 0 --predictor_backbone_plm t5-base-lm-adapt --predictor_batch_size 4 --predictor_gradient_accumulation_steps 3 --dataset_type spider --dataset spider_perm_1 --first_task_id 0 --last_task_id 10 --task_num 11 --teacher true --teacher_plm t5-large-lm-adapt --teacher_with_context false --task_adaptation_with_teacher_logits false --do_cl_eval true
# combine1_perm_1
python train/main.py --cuda_visible_devices 0 --predictor_backbone_plm t5-base-lm-adapt --predictor_batch_size 4 --predictor_gradient_accumulation_steps 3 --dataset_type combine --dataset combine1_perm_1 --first_task_id 0--last_task_id 6 --task_num 7 --teacher true --teacher_plm t5-large-lm-adapt --teacher_with_context false --task_adaptation_with_teacher_logits false --do_cl_eval true
```

+ w/o task adaptation

```shell
# spider_perm_1
python train/main.py --cuda_visible_devices 0 --do_train true --task_adaptation false --predictor_backbone_plm t5-base-lm-adapt --predictor_batch_size 4 --predictor_gradient_accumulation_steps 3 --dataset_type spider --dataset spider_perm_1 --first_task_id 0 --last_task_id 10 --task_num 11 --teacher true --teacher_plm t5-large-lm-adapt --teacher_with_context True --task_adaptation_with_teacher_logits false
# combine1_perm_1
python train/main.py --cuda_visible_devices 0 --do_train true --task_adaptation false --predictor_backbone_plm t5-base-lm-adapt --predictor_batch_size 4 --predictor_gradient_accumulation_steps 3 --dataset_type combine --dataset combine1_perm_1 --first_task_id 0 --last_task_id 6 --task_num 7 --teacher true --teacher_plm t5-large-lm-adapt --teacher_with_context True --task_adaptation_with_teacher_logits false
```

### 5.3 Hyperparameter Study

+ prompt length

```shell
# spider_perm_1
for i in 1 5 20 75
do
python train/main.py --cuda_visible_devices 1 --soft_token_num $i --predictor_backbone_plm t5-small-lm-adapt --predictor_batch_size 12 --predictor_gradient_accumulation_steps 1 --dataset_type spider --dataset spider_perm_1 --first_task_id 10 --last_task_id 10 --task_num 11 --teacher true --teacher_plm t5-large-lm-adapt --teacher_with_context True --task_adaptation_with_teacher_logits false ;
done

for i in 1 5 20 75
do
python train/main.py --cuda_visible_devices 3 --soft_token_num $i --predictor_backbone_plm t5-base-lm-adapt --predictor_batch_size 4 --predictor_gradient_accumulation_steps 3 --dataset_type spider --dataset spider_perm_1 --first_task_id 4 --last_task_id 10 --task_num 11 --teacher true --teacher_plm t5-large-lm-adapt --teacher_with_context True --task_adaptation_with_teacher_logits false ;
done


# combine1_perm_1
for i in 1 5 20 75
do
python train/main.py --cuda_visible_devices 0 --soft_token_num $i --predictor_backbone_plm t5-base-lm-adapt --predictor_batch_size 4 --predictor_gradient_accumulation_steps 3 --dataset_type combine --dataset combine1_perm_1 --first_task_id 4 --last_task_id 6 --task_num 7 --teacher true --teacher_plm t5-large-lm-adapt --teacher_with_context True --task_adaptation_with_teacher_logits false ;
done

for i in 1 5 20 75
do
python train/main.py --cuda_visible_devices 0 --soft_token_num $i --predictor_backbone_plm t5-small-lm-adapt --predictor_batch_size 12 --predictor_gradient_accumulation_steps 1 --dataset_type combine --dataset combine1_perm_1 --first_task_id 0 --last_task_id 6 --task_num 7 --teacher true --teacher_plm t5-large-lm-adapt --teacher_with_context True --task_adaptation_with_teacher_logits false ;
done
```

+ continual_initialization

```shell
# spider_perm_1
python train/main.py --cuda_visible_devices 0 --continual_initialization true --predictor_backbone_plm t5-base-lm-adapt --predictor_batch_size 4 --predictor_gradient_accumulation_steps 3 --dataset_type spider --dataset spider_perm_1 --first_task_id 4 --last_task_id 11 --task_num 11 --teacher true --teacher_plm t5-large-lm-adapt --teacher_with_context True --task_adaptation_with_teacher_logits false --do_cl_eval true
# combine1_perm_1
python train/main.py --cuda_visible_devices 0 --continual_initialization true --predictor_backbone_plm t5-base-lm-adapt --predictor_batch_size 4 --predictor_gradient_accumulation_steps 3 --dataset_type combine --dataset combine1_perm_1 --first_task_id 6 --last_task_id 6 --task_num 7 --teacher true --teacher_plm t5-large-lm-adapt --teacher_with_context True --task_adaptation_with_teacher_logits false --do_cl_eval true
```

+ demonstration example number

```shell
# combine1_cxt_1_4.0_ctco_perm_1
python train/finetune_main.py --cuda_visible_devices 0 --backbone_plm t5-large-lm-adapt --per_device_train_batch_size 3 --per_device_eval_batch_size 3 --gradient_accumulation_steps 4 --dataset_type combine --dataset combine1_cxt_1_4.0_ctco_perm_1 --task_num 7 --first_task_id=0 --last_task_id=6 ;
# spider_cxt_1_4.0_ctco_perm_1
python train/finetune_main.py --cuda_visible_devices 0 --backbone_plm t5-large-lm-adapt --per_device_train_batch_size 3 --per_device_eval_batch_size 3 --gradient_accumulation_steps 4 --dataset_type spider --dataset spider_cxt_1_4.0_ctco_perm_1 --task_num 11 --first_task_id=0 --last_task_id=10 ;
# combine1_cxt_1_3.0_mix_ctco_perm_1
python train/finetune_main.py --cuda_visible_devices 0 --backbone_plm t5-large-lm-adapt --per_device_train_batch_size 3 --per_device_eval_batch_size 3 --gradient_accumulation_steps 4 --dataset_type combine --dataset combine1_cxt_1_3.0_mix_ctco_perm_1 --task_num 7 --first_task_id=0 --last_task_id=6 ;
# spider_cxt_1_3.0_mix_ctco_perm_1
python train/finetune_main.py --cuda_visible_devices 0 --backbone_plm t5-large-lm-adapt --per_device_train_batch_size 3 --per_device_eval_batch_size 3 --gradient_accumulation_steps 4 --dataset_type spider --dataset spider_cxt_1_3.0_mix_ctco_perm_1 --task_num 11 --first_task_id=0 --last_task_id=10 ;
# combine1_cxt_1_2.0_mix_ctco_perm_1
python train/finetune_main.py --cuda_visible_devices 0 --backbone_plm t5-large-lm-adapt --per_device_train_batch_size 3 --per_device_eval_batch_size 3 --gradient_accumulation_steps 4 --dataset_type combine --dataset combine1_cxt_1_2.0_mix_ctco_perm_1 --task_num 7 --first_task_id=0 --last_task_id=6 ;
# spider_cxt_1_2.0_mix_ctco_perm_1
python train/finetune_main.py --cuda_visible_devices 0 --backbone_plm t5-large-lm-adapt --per_device_train_batch_size 3 --per_device_eval_batch_size 3 --gradient_accumulation_steps 4 --dataset_type spider --dataset spider_cxt_1_2.0_mix_ctco_perm_1 --task_num 11 --first_task_id=0 --last_task_id=10 ;





# combine1_cxt_2_4.0_mix_ctco_perm_1
python train/finetune_main.py --cuda_visible_devices 0 --backbone_plm t5-large-lm-adapt --per_device_train_batch_size 3 --per_device_eval_batch_size 3 --gradient_accumulation_steps 4 --dataset_type combine --dataset combine1_cxt_2_4.0_mix_ctco_perm_1 --task_num 7 --first_task_id=0 --last_task_id=6 ;
# spider_cxt_2_4.0_mix_ctco_perm_1
python train/finetune_main.py --cuda_visible_devices 0 --backbone_plm t5-large-lm-adapt --per_device_train_batch_size 3 --per_device_eval_batch_size 3 --gradient_accumulation_steps 4 --dataset_type spider --dataset spider_cxt_2_4.0_mix_ctco_perm_1 --task_num 11 --first_task_id=0 --last_task_id=10 ;
# combine1_cxt_2_3.0_mix_ctco_perm_1
python train/finetune_main.py --cuda_visible_devices 0 --backbone_plm t5-large-lm-adapt --per_device_train_batch_size 3 --per_device_eval_batch_size 3 --gradient_accumulation_steps 4 --dataset_type combine --dataset combine1_cxt_2_3.0_mix_ctco_perm_1 --task_num 7 --first_task_id=0 --last_task_id=6 ;
# spider_cxt_2_3.0_mix_ctco_perm_1
python train/finetune_main.py --cuda_visible_devices 0 --backbone_plm t5-large-lm-adapt --per_device_train_batch_size 3 --per_device_eval_batch_size 3 --gradient_accumulation_steps 4 --dataset_type spider --dataset spider_cxt_2_3.0_mix_ctco_perm_1 --task_num 11 --first_task_id=0 --last_task_id=10 ;
# combine1_cxt_2_2.0_mix_ctco_perm_1
python train/finetune_main.py --cuda_visible_devices 0 --backbone_plm t5-large-lm-adapt --per_device_train_batch_size 3 --per_device_eval_batch_size 3 --gradient_accumulation_steps 4 --dataset_type combine --dataset combine1_cxt_2_2.0_mix_ctco_perm_1 --task_num 7 --first_task_id=0 --last_task_id=6 ;
# spider_cxt_2_2.0_mix_ctco_perm_1
python train/finetune_main.py --cuda_visible_devices 0 --backbone_plm t5-large-lm-adapt --per_device_train_batch_size 3 --per_device_eval_batch_size 3 --gradient_accumulation_steps 4 --dataset_type spider --dataset spider_cxt_2_2.0_mix_ctco_perm_1 --task_num 11 --first_task_id=0 --last_task_id=10 ;

# combine1_cxt_3_4.0_mix_ctco_perm_1
python train/finetune_main.py --cuda_visible_devices 0 --backbone_plm t5-large-lm-adapt --per_device_train_batch_size 3 --per_device_eval_batch_size 3 --gradient_accumulation_steps 4 --dataset_type combine --dataset combine1_cxt_3_4.0_mix_ctco_perm_1 --task_num 7 --first_task_id=0 --last_task_id=6 ;
# spider_cxt_3_4.0_mix_ctco_perm_1
python train/finetune_main.py --cuda_visible_devices 0 --backbone_plm t5-large-lm-adapt --per_device_train_batch_size 3 --per_device_eval_batch_size 3 --gradient_accumulation_steps 4 --dataset_type spider --dataset spider_cxt_3_4.0_mix_ctco_perm_1 --task_num 11 --first_task_id=0 --last_task_id=10 ;
# combine1_cxt_3_3.0_mix_ctco_perm_1
python train/finetune_main.py --cuda_visible_devices 0 --backbone_plm t5-large-lm-adapt --per_device_train_batch_size 3 --per_device_eval_batch_size 3 --gradient_accumulation_steps 4 --dataset_type combine --dataset combine1_cxt_3_3.0_mix_ctco_perm_1 --task_num 7 --first_task_id=0 --last_task_id=6 ;
# spider_cxt_3_3.0_mix_ctco_perm_1
python train/finetune_main.py --cuda_visible_devices 3 --backbone_plm t5-large-lm-adapt --per_device_train_batch_size 3 --per_device_eval_batch_size 3 --gradient_accumulation_steps 4 --dataset_type spider --dataset spider_cxt_3_3.0_mix_ctco_perm_1 --task_num 11 --first_task_id=1 --last_task_id=10 ;
# combine1_cxt_3_2.0_mix_ctco_perm_1
python train/finetune_main.py --cuda_visible_devices 0 --backbone_plm t5-large-lm-adapt --per_device_train_batch_size 3 --per_device_eval_batch_size 3 --gradient_accumulation_steps 4 --dataset_type combine --dataset combine1_cxt_3_2.0_mix_ctco_perm_1 --task_num 7 --first_task_id=0 --last_task_id=6 ;
# spider_cxt_3_2.0_mix_ctco_perm_1
python train/finetune_main.py --cuda_visible_devices 0 --backbone_plm t5-large-lm-adapt --per_device_train_batch_size 3 --per_device_eval_batch_size 3 --gradient_accumulation_steps 4 --dataset_type spider --dataset spider_cxt_3_2.0_mix_ctco_perm_1 --task_num 11 --first_task_id=0 --last_task_id=10 ;
```

### 5.4 C3

+ training teacher

```shell
# combine1_perm_1
python train/finetune_main.py --cuda_visible_devices 0 --backbone_plm t5-large-lm-adapt --per_device_train_batch_size 3 --per_device_eval_batch_size 3 --gradient_accumulation_steps 4 --dataset_type combine --dataset combine1_cxt_1_4.0_mix_ctco_perm_1 --task_num 7 --first_task_id=0 --last_task_id=6
# combine1_perm_2
python train/finetune_main.py --cuda_visible_devices 2 --backbone_plm t5-large-lm-adapt --per_device_train_batch_size 3 --per_device_eval_batch_size 3 --gradient_accumulation_steps 4 --dataset_type combine --dataset combine1_cxt_1_4.0_mix_ctco_perm_2 --task_num 7 --first_task_id=6 --last_task_id=6
# combine1_perm_3
python train/finetune_main.py --cuda_visible_devices 0 --backbone_plm t5-large-lm-adapt --per_device_train_batch_size 3 --per_device_eval_batch_size 3 --gradient_accumulation_steps 4 --dataset_type combine --dataset combine1_cxt_1_4.0_mix_ctco_perm_3 --task_num 7 --first_task_id=2 --last_task_id=6

# spider_perm_1
python train/finetune_main.py --cuda_visible_devices 3 --backbone_plm t5-large-lm-adapt --per_device_train_batch_size 3 --per_device_eval_batch_size 3 --gradient_accumulation_steps 4 --dataset_type spider --dataset spider_cxt_1_4.0_mix_ctco_perm_1 --task_num 11 --first_task_id=0 --last_task_id=10
# spider_perm_2
python train/finetune_main.py --cuda_visible_devices 3 --backbone_plm t5-large-lm-adapt --per_device_train_batch_size 3 --per_device_eval_batch_size 3 --gradient_accumulation_steps 4 --dataset_type spider --dataset spider_cxt_1_4.0_mix_ctco_perm_2 --task_num 11 --first_task_id=1 --last_task_id=10
# spider_perm_3
python train/finetune_main.py --cuda_visible_devices 3 --backbone_plm t5-large-lm-adapt --per_device_train_batch_size 3 --per_device_eval_batch_size 3 --gradient_accumulation_steps 4 --dataset_type spider --dataset spider_cxt_1_4.0_mix_ctco_perm_3 --task_num 11 --first_task_id=1 --last_task_id=10
```

+ training student with teacher logits (student: t5-base, teacher: t5-large)

```shell
# combine1_perm_1
python train/main.py --cuda_visible_devices 0 --do_train true --do_cl_eval true --predictor_backbone_plm t5-base-lm-adapt --predictor_batch_size 4 --predictor_gradient_accumulation_steps 3 --dataset_type combine --dataset combine1_perm_1 --first_task_id 0 --last_task_id 6 --task_num 7 --teacher true --teacher_plm t5-large-lm-adapt --teacher_with_context True --task_adaptation_with_teacher_logits false
# combine1_perm_2
python train/main.py --cuda_visible_devices 3 --do_train true --do_cl_eval true --predictor_backbone_plm t5-base-lm-adapt --predictor_batch_size 4 --predictor_gradient_accumulation_steps 3 --dataset_type combine --dataset combine1_perm_2 --first_task_id 0 --last_task_id 6 --task_num 7 --teacher true --teacher_plm t5-large-lm-adapt --teacher_with_context True --task_adaptation_with_teacher_logits false
# combine1_perm_3
python train/main.py --cuda_visible_devices 0 --do_train true --do_cl_eval true --predictor_backbone_plm t5-base-lm-adapt --predictor_batch_size 4 --predictor_gradient_accumulation_steps 3 --dataset_type combine --dataset combine1_perm_3 --first_task_id 0 --last_task_id 6 --task_num 7 --teacher true --teacher_plm t5-large-lm-adapt --teacher_with_context True --task_adaptation_with_teacher_logits false 
# spider_perm_1
python train/main.py --cuda_visible_devices 2 --do_train true --do_cl_eval true --predictor_backbone_plm t5-base-lm-adapt --predictor_batch_size 4 --predictor_gradient_accumulation_steps 3 --dataset_type spider --dataset spider_perm_1 --first_task_id 0 --last_task_id 10 --task_num 11 --teacher true --teacher_plm t5-large-lm-adapt --teacher_with_context True --task_adaptation_with_teacher_logits false
# spider_perm_2
python train/main.py --cuda_visible_devices 3 --do_train true --do_cl_eval true --predictor_backbone_plm t5-base-lm-adapt --predictor_batch_size 4 --predictor_gradient_accumulation_steps 3 --dataset_type spider --dataset spider_perm_2 --first_task_id 0 --last_task_id 10 --task_num 11 --teacher true --teacher_plm t5-large-lm-adapt --teacher_with_context True --task_adaptation_with_teacher_logits false
# spider_perm_3
python train/main.py --cuda_visible_devices 3 --do_train true --do_cl_eval true --predictor_backbone_plm t5-base-lm-adapt --predictor_batch_size 4 --predictor_gradient_accumulation_steps 3 --dataset_type spider --dataset spider_perm_3 --first_task_id 0 --last_task_id 10 --task_num 11 --teacher true --teacher_plm t5-large-lm-adapt --teacher_with_context True --task_adaptation_with_teacher_logits false
```

+ training student with teacher logits (student: t5-large, teacher: t5-large)

```shell
# combine1_perm_1
python train/main.py --cuda_visible_devices 0 --do_train true --do_cl_eval true --predictor_backbone_plm t5-large-lm-adapt --predictor_batch_size 1 --predictor_gradient_accumulation_steps 12 --dataset_type combine --dataset combine1_perm_1 --first_task_id 0 --last_task_id 6 --task_num 7 --teacher true --teacher_plm t5-large-lm-adapt --teacher_with_context True --task_adaptation_with_teacher_logits false
# combine1_perm_2
python train/main.py --cuda_visible_devices 0 --do_train true --do_cl_eval true --predictor_backbone_plm t5-large-lm-adapt --predictor_batch_size 1 --predictor_gradient_accumulation_steps 12 --dataset_type combine --dataset combine1_perm_2 --first_task_id 0 --last_task_id 6 --task_num 7 --teacher true --teacher_plm t5-large-lm-adapt --teacher_with_context True --task_adaptation_with_teacher_logits false
# combine1_perm_3
python train/main.py --cuda_visible_devices 0 --do_train true --do_cl_eval true --predictor_backbone_plm t5-large-lm-adapt --predictor_batch_size 1 --predictor_gradient_accumulation_steps 12--dataset_type combine --dataset combine1_perm_3 --first_task_id 0 --last_task_id 6 --task_num 7 --teacher true --teacher_plm t5-large-lm-adapt --teacher_with_context True --task_adaptation_with_teacher_logits false 
# spider_perm_1
python train/main.py --cuda_visible_devices 0 --do_train true --do_cl_eval true --predictor_backbone_plm t5-large-lm-adapt --predictor_batch_size 1 --predictor_gradient_accumulation_steps 12 --dataset_type spider --dataset spider_perm_1 --first_task_id 0 --last_task_id 10 --task_num 11 --teacher true --teacher_plm t5-large-lm-adapt --teacher_with_context True --task_adaptation_with_teacher_logits false
# spider_perm_2
python train/main.py --cuda_visible_devices 0 --do_train true --do_cl_eval true --predictor_backbone_plm t5-large-lm-adapt --predictor_batch_size 1 --predictor_gradient_accumulation_steps 12--dataset_type spider --dataset spider_perm_2 --first_task_id 0 --last_task_id 10 --task_num 11 --teacher true --teacher_plm t5-large-lm-adapt --teacher_with_context True --task_adaptation_with_teacher_logits false
# spider_perm_3
python train/main.py --cuda_visible_devices 0 --do_train true --do_cl_eval true --predictor_backbone_plm t5-large-lm-adapt --predictor_batch_size 1 --predictor_gradient_accumulation_steps 12 --dataset_type spider --dataset spider_perm_3 --first_task_id 0 --last_task_id 10 --task_num 11 --teacher true --teacher_plm t5-large-lm-adapt --teacher_with_context True --task_adaptation_with_teacher_logits false
```

+ training student with teacher logits (student: t5-base, teacher: gpt)

```shell
# spider_perm_1
python train/main.py --cuda_visible_devices 1 --do_train true --do_cl_eval true --predictor_backbone_plm t5-base-lm-adapt --predictor_batch_size 4 --predictor_gradient_accumulation_steps 3 --dataset_type spider --dataset spider_perm_1 --first_task_id 2 --last_task_id 3 --task_num 11 --teacher true --teacher_plm gpt --teacher_with_context true --task_adaptation_with_teacher_logits false

# combine1_perm_1
python train/main.py --cuda_visible_devices 3 --do_train true --do_cl_eval true --predictor_backbone_plm t5-base-lm-adapt --predictor_batch_size 4 --predictor_gradient_accumulation_steps 3 --dataset_type combine --dataset combine1_perm_1 --first_task_id 1 --last_task_id 2 --task_num 11 --teacher true --teacher_plm gpt --teacher_with_context true --task_adaptation_with_teacher_logits false
```

+ training student with teacher logits (student: t5-large, teacher: gpt)

```shell
# spider_perm_1
python train/main.py --cuda_visible_devices 0 --do_train true --do_cl_eval true --predictor_backbone_plm t5-large-lm-adapt --predictor_batch_size 1 --predictor_gradient_accumulation_steps 12 --dataset_type spider --dataset spider_perm_1 --first_task_id 4 --last_task_id 6 --task_num 11 --teacher true --teacher_plm gpt --teacher_with_context true --task_adaptation_with_teacher_logits false

# commbine1_perm_1
python train/main.py --cuda_visible_devices 0 --do_train true --do_cl_eval true --predictor_backbone_plm t5-large-lm-adapt --predictor_batch_size 1 --predictor_gradient_accumulation_steps 12 --dataset_type combine --dataset combine1_perm_1 --first_task_id 1 --last_task_id 6 --task_num 7 --teacher true --teacher_plm gpt --teacher_with_context true --task_adaptation_with_teacher_logits false
```

