import sys

sys.path.append('.')

from train.trainer.predictor.predictor_finetune import FinetunePredictorTrainer

import argparse



parser = argparse.ArgumentParser('')
parser.add_argument('--cuda_visible_devices', type=str, default='3')
parser.add_argument('--backbone_plm', type=str, default='t5-large-lm-adapt')
parser.add_argument('--per_device_train_batch_size', type=int, default=3)
parser.add_argument('--per_device_eval_batch_size', type=int, default=3)
parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
parser.add_argument('--model_name_or_path', type=str, default='')
parser.add_argument('--total_output_dir', type=str, default='train/ckpt/predictor/finetune')
parser.add_argument('--total_dataset_dir', type=str, default='train/data_train/predictor')

parser.add_argument('--dataset_type', type=str, default='combine')
parser.add_argument('--dataset', type=str, default='combine1_cxt_1_4.0_ctco_perm_1')
parser.add_argument('--task_num', type=int, default=7)
parser.add_argument('--first_task_id', type=int, default=0)
parser.add_argument('--last_task_id', type=int, default=6)
parser.add_argument('--do_cl_eval', type=bool, default=True)
parser.add_argument('--do_wo_stu_ablation', type=bool, default=False)
parser.add_argument('--do_first_seen_eval', type=bool, default=False)
parser.add_argument('--seed', type=int, default=2022)

parser.add_argument('--metric_for_best_model', type=str, default='exact_match')
parser.add_argument('--greater_is_better', type=bool, default=True)
parser.add_argument('--max_source_length', type=int, default=512)
parser.add_argument('--max_target_length', type=int, default=256)
parser.add_argument('--overwrite_output_dir', type=bool, default=False)
parser.add_argument('--resume_from_checkpoint', type=bool, default=None)
parser.add_argument('--do_train', type=bool, default=True)
parser.add_argument('--do_eval', type=bool, default=True)
parser.add_argument('--do_predict', type=bool, default=True)
parser.add_argument('--predict_with_generate', type=bool, default=True)
parser.add_argument('--lr_scheduler_type', type=str, default='linear')
parser.add_argument('--label_smoothing_factor', type=float, default=0.0)
parser.add_argument('--warmup_ratio', type=float, default=0.0)
parser.add_argument('--warmup_steps', type=int, default=0)
parser.add_argument('--num_beams', type=int, default=4)
parser.add_argument('--optim', type=str, default='adafactor')
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--adam_epsilon', type=float, default=1e-06)
parser.add_argument('--load_best_model_at_end', type=bool, default=True)
parser.add_argument('--num_train_epochs', type=int, default=1000)
parser.add_argument('--save_strategy', type=str, default='steps')
parser.add_argument('--save_total_limit', type=int, default=1)
parser.add_argument('--evaluation_strategy', type=str, default='steps')

args = parser.parse_args()

if args.backbone_plm.startswith('t5-large'):
    dir_name = 'finetune_t5-large'
elif args.backbone_plm.startswith('t5-base'):
    dir_name = 'finetune_t5-base'
elif args.backbone_plm.startswith('t5-small'):
    dir_name = 'finetune_t5-small'
else:
    raise NotImplementedError


args.dataset_dir = f'train/data_train/{args.dataset_type}/{args.dataset}'
args.output_dir = f'train/ckpt/{args.dataset_type}/{args.dataset}/{dir_name}'

if __name__ == '__main__':
    predictor_trainer = FinetunePredictorTrainer(args)

    for task_id in range(args.first_task_id, args.last_task_id + 1):
         predictor_trainer.train(task_id)

    if args.do_cl_eval:
        predictor_trainer.evaluate_continual_learning_metrics(do_wo_stu_ablation=args.do_wo_stu_ablation)

    # predictor_trainer.predict(0, 0)