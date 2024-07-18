import sys

sys.path.append('.')

from train.trainer.predictor.predictor_prompt_tuning import PromptTuningPredictorTrainer

import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser('')
parser.add_argument('--use_cuda', type=str2bool, default=True)
parser.add_argument('--cuda_visible_devices', type=str, default='1')  # openprompt currently supports single gpu only
parser.add_argument('--seed', type=int, default=2022)

parser.add_argument('--soft_token_num', type=int, default=150)
parser.add_argument('--task_adaptation', type=str2bool, default=True)
parser.add_argument('--continual_initialization', type=str2bool, default=False)
parser.add_argument('--do_train', type=str2bool, default=True)
parser.add_argument('--do_cl_eval', type=str2bool, default=True)

parser.add_argument('--predictor_backbone_plm', type=str, default='t5-large-lm-adapt')
parser.add_argument('--predictor_batch_size', type=int, default=4)
parser.add_argument('--predictor_gradient_accumulation_steps', type=int, default=3)
parser.add_argument('--predictor_max_input_length', type=int, default=768)

parser.add_argument('--dataset_type', type=str, default='spider')
parser.add_argument('--dataset', type=str, default='spider_perm_1')
parser.add_argument('--first_task_id', type=int, default=0)
parser.add_argument('--last_task_id', type=int, default=10)
parser.add_argument('--task_num', type=int, default=7)
parser.add_argument('--dataset_dir', type=str, default=None)
parser.add_argument('--output_dir', type=str, default=None)

parser.add_argument('--teacher', type=str2bool, default=True)
parser.add_argument('--teacher_plm', type=str, default='gpt')  # t5-large-lm-adapt/gpt
parser.add_argument('--teacher_output_dir', type=str, default=None)
parser.add_argument('--teacher_dataset_dir', type=str, default=None)
parser.add_argument('--logits_dir', type=str, default=None)
parser.add_argument('--teacher_with_context', type=str2bool, default=True)
parser.add_argument('--task_adaptation_with_teacher_logits', type=str2bool, default=False)

args = parser.parse_args()

if args.predictor_backbone_plm.startswith('t5-large'):
    dir_name = 'prompt_t5-large'
elif args.predictor_backbone_plm.startswith('t5-base'):
    dir_name = 'prompt_t5-base'
elif args.predictor_backbone_plm.startswith('t5-small'):
    dir_name = 'prompt_t5-small'
else:
    raise NotImplementedError

if args.teacher:
    perm_id = args.dataset[-1]
    if args.dataset_type == 'spider':
        if args.teacher_with_context:
            args.teacher_output_dir = f'train/ckpt/spider/spider_cxt_1_4.0_mix_ctco_perm_{perm_id}/finetune_t5-large'
            args.teacher_dataset_dir = f'train/data_train/spider/spider_cxt_1_4.0_mix_ctco_perm_{perm_id}'
            args.output_dir = f'train/ckpt/{args.dataset_type}/{args.dataset}/{dir_name}_teacher'
        else:
            args.teacher_output_dir = f'train/ckpt/spider/spider_perm_{perm_id}/finetune_t5-large'
            args.teacher_dataset_dir = f'train/data_train/spider/spider_perm_{perm_id}'
            args.output_dir = f'train/ckpt/{args.dataset_type}/{args.dataset}/{dir_name}_teacher-wo-cxt'

    elif args.dataset_type == 'combine':
        if args.teacher_with_context:
            args.teacher_output_dir = f'train/ckpt/combine/combine1_cxt_1_4.0_mix_ctco_perm_{perm_id}/finetune_t5-large'
            args.teacher_dataset_dir = f'train/data_train/combine/combine1_cxt_1_4.0_mix_ctco_perm_{perm_id}'
            args.output_dir = f'train/ckpt/{args.dataset_type}/{args.dataset}/{dir_name}_teacher'
        else:
            args.teacher_output_dir = 'train/ckpt/combine/combine1_perm_1/finetune_t5-large'
            args.teacher_dataset_dir = 'train/data_train/combine/combine1_perm_1'
            args.output_dir = f'train/ckpt/{args.dataset_type}/{args.dataset}/{dir_name}_teacher-wo-cxt'
    else:
        raise NotImplementedError

    args.dataset_dir = f'train/data_train/{args.dataset_type}/{args.dataset}'
    args.logits_dir = f'train/ckpt/{args.dataset_type}/{args.dataset}/teacher_logits'

else:
    args.dataset_dir = f'train/data_train/{args.dataset_type}/{args.dataset}'
    args.output_dir = f'train/ckpt/{args.dataset_type}/{args.dataset}/{dir_name}'

if args.continual_initialization:
    args.output_dir += '_ci'

if not args.task_adaptation:
    args.output_dir += f'_no-task-adapt'

if not args.soft_token_num == 150:
    args.output_dir += f'_{args.soft_token_num}'


if __name__ == '__main__':
    predictor_trainer = PromptTuningPredictorTrainer(args)
    # retriever_trainer = PromptTuningRetrieverTrainer(args)

    if args.teacher:
        for task_id in range(args.task_num):
            predictor_trainer.prepare_teacher_logits(task_id, batch_size=12)


    if args.do_train:
        for task_id in range(args.first_task_id, args.last_task_id + 1):
            # predictor_trainer.predict(task_id, freeze_plm=True, batch_size=16)
            predictor_trainer.train(task_id)

            # predictor_trainer.align_tokenizers(task_id=task_id)

    if args.do_cl_eval:
        predictor_trainer.evaluate_continual_learning_metrics(task_range=range(args.task_num))
