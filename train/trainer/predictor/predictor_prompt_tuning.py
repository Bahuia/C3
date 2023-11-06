import json
import math
import os
import shutil
import time
import re

import torch
from torch.nn import KLDivLoss, MSELoss, CrossEntropyLoss
from torch.nn.functional import log_softmax, softmax
import tiktoken

from openprompt import PromptForGeneration, PromptDataLoader
from openprompt.plms import T5TokenizerWrapper
from openprompt.prompts import SoftTemplate
from openprompt.data_utils import InputExample
from tqdm import tqdm
from transformers import Adafactor, set_seed, T5ForConditionalGeneration, AutoTokenizer

from train.train_utils.spider.evaluation import evaluate as spider_evaluate
from train.train_utils.wikisql.evaluation import evaluate as wikisql_evaluate
from train.train_utils.spider.evaluation_single_example import evaluate_single_example as spider_eval_single_example
from train.train_utils.wikisql.evaluation_single_example import evaluate_single_example as wikisql_eval_single_example
from train.train_utils.wikisql.evaluation_single_example import wrap_pred_query

from train.train_utils.prompt_utils import PromptForPredictorGeneration

from train.trainer.predictor.predictor_base import PredictorTrainer


class PromptTuningPredictorTrainer(PredictorTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.generation_arguments = {
            'max_length': 256,
            'max_new_tokens': None,
            'min_length': 5,
            'temperature': 1.0,
            'do_sample': False,
            'top_k': 0,
            'top_p': 0.9,
            'repetition_penalty': 1.0,
            'num_beams': 4,
            'bad_words_ids': [[628], [198]]
        }
        self.output_dir = self.args.output_dir
        self.dataset_dir = self.args.dataset_dir
        # self.loss_fct = MSELoss() if self.args.teacher else None
        # self.loss_fct = (log_KLDivLosstarget=True) if self.args.teacher else None
        self.loss_fct = CrossEntropyLoss(reduction='none') if self.args.teacher else None
        if self.args.teacher:
            if self.args.teacher_plm.startswith('t5'):
                self.output_dir += '_T5'
            elif self.args.teacher_plm.startswith('gpt'):
                self.output_dir += '_GPT'
            else:
                raise NotImplementedError

            if not self.args.task_adaptation_with_teacher_logits:
                self.output_dir += '_h0'
        '''
        if self.args.supervised_retriever:
            self.output_dir += '/predictor'
            self.dataset_dir += '/predictor'
        '''
        set_seed(self.args.seed)

    def prepare_raw_dataset(self, seq2seq_jsonl_file: str) -> list:
        """
        prepare raw datasets for the dataloader
        :param seq2seq_jsonl_file:
        :return: raw datasets for dataloader
        """
        raw_dataset = []
        with open(seq2seq_jsonl_file, 'r', encoding='utf-8') as reader:
            for line in reader.readlines():
                ex = json.loads(line)
                raw_dataset.append(
                    InputExample(guid=ex['example']['guid'], text_a='', text_b=ex['text'],
                                 tgt_text=ex['sql'], meta=ex['example'])
                )

        return raw_dataset

    def load_plm_and_prompt(self, model_name_or_path: str, freeze_plm: bool, template_path: str = None,
                            with_teacher: bool = False):
        """
        load PLM for training
        :param task_id: task id in the continual learning task stream
        :param model_name_or_path: plm path
        :param freeze_plm: whether to freeze the plm's parameters
        :return: prompt_model, tokenizer, wrapper_class, template
        """
        print(f'Loading PLM from {model_name_or_path} (Parameter Frozen: {freeze_plm}) ... ', flush=True, end='')
        plm = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        tokenizer.add_tokens(['<', '<='])
        print('Done', flush=True)

        # prepare the soft prompt template for training
        template = SoftTemplate(model=plm, tokenizer=tokenizer, num_tokens=self.args.soft_token_num,
                                text='{"placeholder":"text_a", "shortenable": False}'
                                     '{"placeholder":"text_b"}{"soft": None}{"mask"}')

        if template_path is not None:
            print(f'\nLoading soft prompt from {template_path} ... ', flush=True, end='')
            state_dict = torch.load(template_path)
            template.soft_embeds = torch.nn.Parameter(state_dict['soft_embeds'], requires_grad=True)
            print('Done', flush=True)

        if with_teacher:
            prompt_model = PromptForPredictorGeneration(plm=plm, template=template, freeze_plm=freeze_plm,
                                                        tokenizer=tokenizer)
        else:
            prompt_model = PromptForGeneration(plm=plm, template=template, freeze_plm=freeze_plm, tokenizer=tokenizer)

        print('Moving all model parameters and buffering to the GPU ... ', flush=True, end='')
        if self.args.use_cuda:
            prompt_model = prompt_model.cuda()
        print('Done', flush=True)

        return prompt_model, tokenizer, template

    def prepare_optimizers(self, task_id: int, prompt_model):
        """
        prepare optimizers for the soft prompt & PLM
        :param task_id: task id in the continual learning task stream
        :param prompt_model: loaded PLM & soft prompt template
        :return: soft prompt optimizer, PLM optimizer (if the PLM's parameters aren't frozen)
        """
        prompt_optimizer, plm_optimizer = None, None

        # if the soft prompt's parameters are tuned together with the PLM's,both of their learning rates are set as 1e-4
        if task_id == 0 and self.args.task_adaptation:

            plm_learning_rate = 1e-4
            no_decay = ['bias', 'LayerNorm.weight']
            plm_optimizer_grouped_parameters = [
                {'params': [p for n, p in prompt_model.plm.named_parameters() if
                            not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]
            plm_optimizer = Adafactor(plm_optimizer_grouped_parameters, lr=plm_learning_rate, relative_step=False,
                                      scale_parameter=False, warmup_init=False)

            prompt_learning_rate = 1e-4
        # else if soft prompt's parameters are tuned while the PLM's are frozen, the learning rate is set as 0.3
        else:
            prompt_learning_rate = 0.3

        prompt_optimizer_grouped_parameters = [{'params': [p for name, p in prompt_model.template.named_parameters() if
                                                           'raw_embedding' not in name]}]
        prompt_optimizer = Adafactor(prompt_optimizer_grouped_parameters, lr=prompt_learning_rate, relative_step=False,
                                     scale_parameter=False, warmup_init=False)

        return prompt_optimizer, plm_optimizer

    def get_meta(self, dataloader) -> (list, list):
        """
        get lists of db_ids, gold SQL queries and interaction_ids (CoSQL) for examples in the dataloader
        :param dataloader: dataloader for evaluation/prediction
        :return: a list of db_ids, a list of gold SQL queries
        """
        db_ids, sqls, interaction_ids = [], [], []
        guid_to_ex = {}
        for ex in dataloader.raw_dataset:
            guid_to_ex[ex.guid] = ex
        for inputs in dataloader:
            for guid in inputs['guid']:
                guid = guid.item()
                db_ids.append(guid_to_ex[guid].meta['db_id'])
                sqls.append(guid_to_ex[guid].tgt_text)
                if self.args.dataset.startswith('cosql'):
                    interaction_ids.append(guid_to_ex[guid].meta['interaction_id'])

        return db_ids, sqls, interaction_ids

    def evaluate(self, prompt_model, dataloader):
        """
        evaluate generated SQL queries' match/execution accuracies
        :param prompt_model: loaded PLM & soft prompt template
        :param dataloader: dataloader for evaluation
        :return: lists of execution scores, match scores, evaluation results, generated SQL queries
        """
        generated_sqls = []
        gt_dbs, gt_sqls, interaction_ids = self.get_meta(dataloader)
        exec_score, match_score = None, None
        with torch.no_grad():
            prompt_model.eval()

            for step, inputs in enumerate(tqdm(dataloader, desc='evaluation')):
                if self.args.use_cuda:
                    inputs = inputs.cuda()
                _, output_sentence = prompt_model.generate(inputs, **self.generation_arguments)
                generated_sqls.extend(output_sentence)

            if self.args.dataset.startswith('spider'):
                exec_score, match_score, eval_results = spider_evaluate(gt_sqls, gt_dbs, generated_sqls, 'all')

            elif self.args.dataset.startswith('combine'):
                dataset_name = dataloader.raw_dataset[0].meta['dataset_name']
                if dataset_name == 'spider':
                    match_score, eval_results = spider_evaluate(gt_sqls, gt_dbs, generated_sqls, 'match')
                elif dataset_name == 'wikisql':
                    match_score, eval_results = wikisql_evaluate(gt_sqls, gt_dbs, generated_sqls, 'match')

            elif self.args.dataset.startswith('cosql'):
                pre_ia_id, g_ia_db = None, None
                g_ia_one, p_ia_one = [], []
                gold_interactions, pred_interactions, gold_interaction_dbs = [], [], []
                for i, ia_id in enumerate(interaction_ids):
                    if ia_id != pre_ia_id and pre_ia_id is not None:
                        gold_interactions.append(g_ia_one)
                        pred_interactions.append(p_ia_one)
                        gold_interaction_dbs.append(g_ia_db)
                        g_ia_one, p_ia_one = [gt_sqls[i]], [generated_sqls[i]]
                        g_ia_db = gt_dbs[i]
                    else:
                        g_ia_one.append(gt_sqls[i])
                        p_ia_one.append(generated_sqls[i])
                        g_ia_db = gt_dbs[i]
                    pre_ia_id = ia_id
                gold_interactions.append(g_ia_one)
                pred_interactions.append(p_ia_one)
                gold_interaction_dbs.append(g_ia_db)

                gt_sqls = gold_interactions
                gt_dbs = gold_interaction_dbs

                exec_score, match_score, eval_results = cosql_evaluate(gt_sqls, gt_dbs, pred_interactions, 'all')
            else:
                raise NotImplementedError

        return exec_score, match_score, eval_results, generated_sqls

    def predict(self, task_id: int, freeze_plm, batch_size=None):
        print(f'\n-------- Running Prediction on Task {task_id}--------\n', flush=True)
        torch.cuda.empty_cache()
        template_path = f'{self.output_dir}/task_{task_id}/template.pth'
        print(f'Loading soft prompt from {template_path} ... ', flush=True, end='')
        template_state_dict = torch.load(f'{self.output_dir}/task_{task_id}/template.pth')
        print('Done')
        if freeze_plm:
            if self.args.task_adaptation:
                plm_path = f'{self.output_dir}/task_0/model'
            else:
                plm_path = f'train/plms/{self.args.predictor_backbone_plm}'
        else:
            # load the best checkpoint
            plm_path = f'{self.output_dir}/task_{task_id}/model'
        print(f'Loading PLM from {plm_path} ... ', flush=True, end='')
        plm = T5ForConditionalGeneration.from_pretrained(plm_path)
        tokenizer = AutoTokenizer.from_pretrained(plm_path)
        template = SoftTemplate(model=plm, tokenizer=tokenizer, num_tokens=self.args.soft_token_num,
                                text='{"placeholder":"text_a", "shortenable": False}'
                                     '{"placeholder":"text_b"}{"soft": None}{"mask"}')
        template.soft_embeds = torch.nn.Parameter(template_state_dict['soft_embeds'])
        prompt_model = PromptForGeneration(plm=plm, template=template, freeze_plm=True, tokenizer=tokenizer)
        if self.args.use_cuda:
            prompt_model = prompt_model.cuda()
        print('Done', flush=True)

        dataloader_args = {
            'template': prompt_model.template,
            'tokenizer': tokenizer,
            'tokenizer_wrapper_class': T5TokenizerWrapper,
            'decoder_max_length': 256,
            'batch_size': batch_size if batch_size is not None else self.args.predictor_batch_size,
            'predict_eos_token': True,
            'max_seq_length': self.args.predictor_max_input_length,
            'truncate_method': 'head',
        }

        generated_sqls_file = f'{self.output_dir}/task_{task_id}/generated_predictions.txt'
        eval_results_file = f'{self.output_dir}/task_{task_id}/eval_results.txt'
        seq2seq_jsonl_file = f'{self.dataset_dir}/task_{task_id}/test_seq2seq.jsonl'

        print('Preparing test dataloader ...', flush=True)

        raw_dataset = self.prepare_raw_dataset(seq2seq_jsonl_file)
        test_dataloader = PromptDataLoader(dataset=raw_dataset, **dataloader_args)
        print('Done')

        _, _, eval_results, generated_sqls = self.evaluate(prompt_model, test_dataloader)

        with open(generated_sqls_file, 'w') as writer:
            writer.write('\n'.join(generated_sqls))
        with open(eval_results_file, 'w') as writer:
            writer.write('\n'.join(eval_results))

    def train(self, task_id: int):
        """
        main training process
        :return:
        """
        print(f'\n------------Performing Task {task_id}------------\n', flush=True)
        os.makedirs(f'{self.output_dir}/task_{task_id}', exist_ok=True)

        if task_id == 0 and self.args.teacher and self.args.task_adaptation and not self.args.task_adaptation_with_teacher_logits and self.args.soft_token_num == 150:
            trained_ckpt = self.output_dir.split('_teacher')[0] + '/task_0'
            if os.path.exists(f'{trained_ckpt}/model/pytorch_model.bin'):
                print(f'Detected trained checkpoint, copying from {trained_ckpt} ... ', flush=True, end='')
                shutil.copytree(trained_ckpt, f'{self.output_dir}/task_{task_id}', dirs_exist_ok=True)
                print('Done')
                return

        if task_id == 0 or not self.args.task_adaptation:
            model_name_or_path = f'train/plms/{self.args.predictor_backbone_plm}'
        else:
            model_name_or_path = f'{self.output_dir}/task_0/model'

        if task_id == 0 and self.args.task_adaptation:
            freeze_plm = False
        else:
            freeze_plm = True

        if task_id > 0:
            if not self.args.continual_initialization:
                template_path = f'{self.output_dir}/task_0/template.pth'
            else:
                template_path = f'{self.output_dir}/task_{task_id - 1}/template.pth'
        else:
            template_path = None

        if self.args.teacher:
            if task_id == 0 and not self.args.task_adaptation_with_teacher_logits:
                with_teacher = False
            else:
                with_teacher = True
        else:
            with_teacher = False

        prompt_model, tokenizer, template = \
            self.load_plm_and_prompt(model_name_or_path, freeze_plm, template_path, with_teacher=with_teacher)

        print('Preparing dataloader for training ...', flush=True)
        raw_datasets = {}
        for ds_name in ['train', 'dev']:
            if with_teacher and self.args.teacher_plm == 'gpt' and ds_name == 'train':
                seq2seq_jsonl_file = f'{self.dataset_dir}/task_{task_id}/{ds_name}_gpt_seq2seq.jsonl'
            else:
                seq2seq_jsonl_file = f'{self.dataset_dir}/task_{task_id}/{ds_name}_seq2seq.jsonl'
            raw_datasets[ds_name] = self.prepare_raw_dataset(seq2seq_jsonl_file)

        dataloader_args = {
            'template': template,
            'tokenizer': tokenizer,
            'tokenizer_wrapper_class': T5TokenizerWrapper,
            'decoder_max_length': 256,
            'batch_size': self.args.predictor_batch_size,
            'predict_eos_token': True,
            'max_seq_length': self.args.predictor_max_input_length,
            'truncate_method': 'head',
        }
        print('Preparing train dataloader ...', flush=True)
        train_dataloader = PromptDataLoader(dataset=raw_datasets['train'], teacher_forcing=True, **dataloader_args)
        print('Preparing validation dataloader ...', flush=True)
        validation_dataloader = PromptDataLoader(dataset=raw_datasets['dev'], **dataloader_args)

        # training and generation
        #
        tot_loss, log_loss, actual_step, max_match_score = 0, 0, 0, -1
        num_examples = len(raw_datasets['train'])

        # set max training epochs
        num_epochs = 10000 if not freeze_plm else 20000
        if not self.args.task_adaptation:
            num_epochs = 2000

        actual_batch_size = self.args.predictor_batch_size * self.args.predictor_gradient_accumulation_steps

        # set the number of epochs or steps to wait for before the first evaluation can be performed
        # and the number of epochs between two evaluations
        if not freeze_plm:
            eval_delay = 50
            eval_epochs = 5 if self.args.predictor_backbone_plm == 't5-large-lm-adapt' else 10
        else:
            if self.args.teacher:
                if self.args.task_adaptation_with_teacher_logits or not self.args.teacher_with_context:
                    eval_delay, eval_epochs = 1000, 50
                else:
                    eval_delay, eval_epochs = 2000, 50
                if self.args.predictor_backbone_plm == 't5-large-lm-adapt':
                    eval_delay, eval_epochs = 500, 25
            else:
                if self.args.predictor_backbone_plm == 't5-large-lm-adapt':
                    eval_delay, eval_epochs = 500, 25
                else:
                    eval_delay, eval_epochs = 1000, 50

            if not self.args.soft_token_num == 150:
                if self.args.soft_token_num <= 50:
                    eval_delay, eval_epochs = 100, 25
                else:
                    eval_delay, eval_epochs = 500, 25

            if self.args.predictor_backbone_plm == 't5-small-lm-adapt' and self.args.soft_token_num == 150:
                eval_delay, eval_epochs = 500, 25

            if not self.args.task_adaptation:
                eval_delay, eval_epochs = 500, 25

        # set up early stopping
        early_stop_patience = 5 if self.args.predictor_backbone_plm == 't5-large-lm-adapt' else 10
        early_stop_patience_counter = 0

        def remove_last_ckpt():
            """
            remove template checkpoint
            :return:
            """
            output_dir = f'{self.output_dir}/task_{task_id}'
            for file_name in os.listdir(output_dir):
                if file_name.startswith('template-') and file_name.endswith('.pth'):
                    os.remove(f'{output_dir}/{file_name}')

        prompt_optimizer, plm_optimizer = self.prepare_optimizers(task_id, prompt_model)

        teachers = None
        if with_teacher:
            if self.args.teacher_plm.startswith('t5'):
                logit_file_name = 't5-large_logits.pth' if self.args.teacher_with_context else 't5-large-wo-cxt_logits.pth'
                teacher_logits_file = f'{self.args.logits_dir}/task_{task_id}/{logit_file_name}'
            elif self.args.teacher_plm.startswith('gpt'):
                teacher_logits_file = f'{self.args.logits_dir}/task_{task_id}/gpt_logits.pth'
            else:
                raise NotImplementedError
            print(f'Loading teacher logits from {teacher_logits_file} ... ', end='', flush=True)
            teachers = torch.load(teacher_logits_file)
            print('Done')

        # start training
        print('\n**********Train**********', flush=True)
        print(f'soft token num:\t{self.args.soft_token_num}', flush=True)
        print(f'num examples:\t{num_examples}', flush=True)
        print(f'num epochs:\t{num_epochs}', flush=True)
        print(f'batch size:\t{self.args.predictor_batch_size}', flush=True)
        print(f'gradient accumulation steps:\t{self.args.predictor_gradient_accumulation_steps}', flush=True)
        print(f'actual batch size:\t{actual_batch_size}', flush=True)
        print(f'eval delay:\t{eval_delay}')
        print(f'eval intervals:\t{eval_epochs} epoch(s)', flush=True)
        print(f'train with teacher logits:\t{with_teacher}\n', flush=True)

        for epoch in tqdm(range(num_epochs), desc=f'task_{task_id}'):
            start_time = time.time()
            prompt_model.train()
            for step, inputs in enumerate(tqdm(train_dataloader, desc=f'task_{task_id} epoch_{epoch}')):
                if self.args.use_cuda:
                    inputs = inputs.cuda()

                if with_teacher:
                    logits = prompt_model(inputs)
                    guids = inputs['guid'].cpu().tolist()
                    batch_teachers = [teachers[guid] for guid in guids]
                    batch_logits = torch.split(logits, 1, dim=0)
                    processed_batch_logits, processed_batch_teacher_logits, indices = [], [], []
                    start_index, end_index = 0, 0
                    for single_logits, single_teacher in zip(batch_logits, batch_teachers):
                        top_k_indices = single_teacher['top_k_indices'].to('cuda')
                        top_k_logits = single_teacher['top_k_logits'].to('cuda')
                        length = top_k_indices.size(1)
                        end_index = start_index + length
                        indices.append([start_index, end_index])
                        start_index = end_index
                        single_logits = single_logits[:, 1:length + 1, :]
                        # CrossEntropyLoss with class probabilities

                        if self.args.teacher_plm.startswith('t5'):
                            top_k_logits = top_k_logits.softmax(dim=-1)

                        single_teacher_logits = torch.zeros_like(single_logits).scatter_(dim=-1, index=top_k_indices,
                                                                                         src=top_k_logits).to('cuda')
                        processed_batch_logits.append(single_logits.squeeze(0))
                        processed_batch_teacher_logits.append(single_teacher_logits.squeeze(0))

                        '''
                        # KLDivLoss
                        single_teacher_logits = torch.zeros_like(single_logits).scatter_(dim=-1, index=top_k_indices,
                                                                                         src=top_k_logits).to('cuda')
                        single_logits = log_softmax(single_logits, dim=-1)
                        single_teacher_logits = log_softmax(single_teacher_logits, dim=-1)      
                        processed_batch_logits.append(single_logits.squeeze(0))
                        # processed_batch_teacher_logits.append(top_k_logits.squeeze(0))
                        processed_batch_teacher_logits.append(single_teacher_logits.squeeze(0))
                        '''

                    processed_logits = torch.cat(processed_batch_logits, dim=0)
                    processed_teacher_logits = torch.cat(processed_batch_teacher_logits, dim=0)
                    loss = self.loss_fct(processed_logits, processed_teacher_logits)
                    batch_losses = []
                    for s_idx, e_idx in indices:
                        single_ex_loss = loss[s_idx: e_idx].sum()
                        batch_losses.append(single_ex_loss)

                    loss = torch.stack(batch_losses).mean()  # batch_mean_loss

                    # loss = self.loss_fct(processed_logits, processed_teacher_logits)

                else:
                    loss = prompt_model(inputs)

                loss.backward()

                tot_loss += loss.item()
                actual_step += 1

                if actual_step % self.args.predictor_gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(template.parameters(), 1.0)
                    prompt_optimizer.step()
                    prompt_optimizer.zero_grad()

                    if not freeze_plm:
                        plm_optimizer.step()
                        plm_optimizer.zero_grad()

            duration = time.time() - start_time
            mean_loss = (tot_loss - log_loss) / len(train_dataloader)

            print(f'epoch {epoch}, average loss: {mean_loss}')
            with open(f'{self.output_dir}/task_{task_id}/train_log.jsonl', 'a') as writer:
                writer.write(json.dumps({'task': task_id, 'epoch': epoch, 'loss': mean_loss,
                                         'duration': duration}) + '\n')
            log_loss = tot_loss

            # do evaluation
            if (epoch + 1) % eval_epochs == 0 and (epoch + 1) >= eval_delay:
                print('\n**********Evaluation**********', flush=True)
                exec_score, match_score, _, _ = self.evaluate(prompt_model, validation_dataloader)

                print(f'Epoch {epoch}:\texec score: {exec_score}\tmatch score: {match_score}\n', flush=True)
                with open(f'{self.output_dir}/task_{task_id}/eval_log.jsonl', 'a') as writer:
                    writer.write(json.dumps({'task': task_id, 'epoch': epoch, 'exec score:': exec_score,
                                             'match score': match_score}) + '\n')

                if match_score > max_match_score:
                    early_stop_patience_counter = 0
                    max_match_score = match_score
                    # preserve the current best soft prompt template
                    model_state_dict = prompt_model.template.state_dict()
                    best_state_dict = {'soft_embeds': model_state_dict['soft_embeds']}
                    torch.save(best_state_dict, f'{self.output_dir}/task_{task_id}/template.pth')
                    if not freeze_plm:
                        # overwrite the former best plm checkpoint
                        prompt_model.plm.save_pretrained(f'{self.output_dir}/task_{task_id}/model')
                        prompt_model.tokenizer.save_pretrained(f'{self.output_dir}/task_{task_id}/model')

                else:
                    early_stop_patience_counter += 1

                # save latest checkpoints
                model_state_dict = prompt_model.template.state_dict()
                state_dict = {}
                state_dict['soft_embeds'] = model_state_dict['soft_embeds']
                remove_last_ckpt()
                torch.save(state_dict, f'{self.output_dir}/task_{task_id}/template-{epoch}.pth')
                torch.save(early_stop_patience_counter,
                           f'{self.output_dir}/task_{task_id}/early_stop_patience_counter.pth')

            # lose early stop patience
            if early_stop_patience_counter >= early_stop_patience or epoch == num_epochs - 1:
                remove_last_ckpt()
                os.remove(f'{self.output_dir}/task_{task_id}/early_stop_patience_counter.pth')
                print(f'Lose early stop patience at epoch {epoch}.\n', flush=True)
                break

        # do prediction
        if not freeze_plm:
            prompt_model = None
        self.predict(task_id, freeze_plm)

    # def save_logits(self, batch_logits: torch.Tensor, k: int):

    def evaluate_continual_learning_metrics(self, task_range):
        """
        compute continual learning metrics after training
        :return:
        """
        acc_dct = {}
        acc_a, acc_w, counter_a, counter_w = 0, 0, 0, 0
        for task_id in task_range:
            if task_id > 0:
                eval_result_file = f'{self.output_dir}/task_{task_id}/eval_results.txt'
            else:
                eval_result_file = f'{self.output_dir}/task_{task_id}/eval_results.txt'
            with open(eval_result_file, 'r') as reader:
                for line in reader.readlines():
                    if line.startswith('count'):
                        task_size = int(line.strip()[-3:])
                    if line.startswith('exact match'):
                        acc = float(line.strip()[-5:])
                        break
                acc_a += acc
                counter_a += 1
                acc_w += acc * task_size
                counter_w += task_size
            acc_dct[f'task_{task_id}'] = {'acc_a': acc_a / counter_a, 'acc_w': acc_w / counter_w}
        json.dump(acc_dct, open(f'{self.output_dir}/metrics.json', 'w'), indent=4)

    def prepare_teacher_logits(self, task_id: int, batch_size: int = 12):
        print(f'\n----------Preparing Teacher Logits for Task {task_id}----------', flush=True)
        os.makedirs(f'{self.args.logits_dir}/task_{task_id}/', exist_ok=True)
        logit_file_name = 't5-large_logits.pth' if self.args.teacher_with_context else 't5-large-wo-cxt_logits.pth'

        if os.path.exists(f'{self.args.logits_dir}/task_{task_id}/{logit_file_name}'):
            print('Logits file already exists, skip inference procedure.', flush=True)
            return

        teacher_train_file = f'{self.args.teacher_dataset_dir}/task_{task_id}/train_seq2seq.jsonl'
        train_examples = {}
        with open(teacher_train_file, 'r', encoding='utf-8') as reader:
            for line in reader.readlines():
                ex = json.loads(line)
                guid = ex['example']['guid']
                text, sql = ex['text'], ex['sql']
                if guid not in train_examples.keys():
                    train_examples[guid] = {'text': text, 'sql': sql, 'example': {'guid': guid}}
                else:
                    if ' || ' in text:
                        train_examples[guid]['text'] = text
        raw_inputs = [train_examples[k] for k in train_examples.keys()]

        model_name_or_path = f'{self.args.teacher_output_dir}/task_{task_id}'
        print(f'Loading teacher plm from {model_name_or_path} ...', end='', flush=True)
        plm = T5ForConditionalGeneration.from_pretrained(model_name_or_path).to('cuda')
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        print('Done')
        tokenizer.truncation_side = 'left'

        # print(tokenizer.get_added_vocab())

        def batch(iterable):
            l = len(iterable)
            for ndx in range(0, l, batch_size):
                yield iterable[ndx: min(ndx + batch_size, l)]

        with torch.no_grad():
            num_examples = len(raw_inputs)
            print(f'Getting teacher logits for {num_examples} items in {math.ceil(num_examples / batch_size)} steps')
            to_save_logits = {}
            for inputs in tqdm(batch(raw_inputs), desc='Getting teacher logits'):
                labels = [x['sql'] for x in inputs]
                texts = [x['text'] for x in inputs]
                guids = [x['example']['guid'] for x in inputs]
                input_ids = tokenizer(texts, return_tensors='pt', truncation=True, padding=True,
                                      max_length=512).input_ids.to('cuda')
                tokenized_labels = tokenizer(labels, return_tensors='pt', truncation=True, padding=True,
                                             max_length=256)
                # print(tokenized_labels.keys())
                label_ids = tokenized_labels.input_ids.to('cuda')
                attention_mask = tokenized_labels.attention_mask.to('cuda')

                lengths = torch.count_nonzero(attention_mask, dim=-1)
                # print(lengths)
                # print(label_ids)

                output = plm(input_ids=input_ids, labels=label_ids)

                output_logits = output.logits.to('cuda')
                batch_logits = torch.split(output_logits, 1, dim=0)
                batch_labels = torch.split(label_ids, 1, dim=0)
                batch_lengths = torch.split(lengths, 1, dim=0)

                for logits, lab_ids, length, guid in zip(batch_logits, batch_labels, batch_lengths, guids):
                    # print(logits.size())
                    # print(lab_ids.size())
                    clipped_logits = logits[:, :length.item(), :]
                    clipped_labels = lab_ids[:, :length.item()]
                    # print('clipped_logits_size:', clipped_logits.size())

                    top_k_indices, top_k_logits = self.get_top_k_indices_and_logits(clipped_logits, k=5)
                    to_save_logits[guid] = {'top_k_indices': top_k_indices.cpu(), 'top_k_logits': top_k_logits.cpu()}

            # print(to_save_logits)
            torch.save(to_save_logits, f'{self.args.logits_dir}/task_{task_id}/{logit_file_name}')

    def get_top_k_indices_and_logits(self, logits: torch.tensor, k: int = 5):
        sorted_logits = torch.argsort(logits, dim=-1, descending=True)
        top_k_indices = sorted_logits[:, :, :k].contiguous()
        top_k_logits = torch.gather(logits, dim=-1, index=top_k_indices).contiguous()

        return top_k_indices, top_k_logits

    def align_tokenizers(self, task_id: int):
        if self.args.dataset_type == 'combine' and task_id in [1, 3, 5]:
            dataset_name = 'wikisql'
            evaluation_method = wikisql_eval_single_example
        else:
            dataset_name = 'spider'
            evaluation_method = spider_eval_single_example


        print(f'------- Aligning tokenizers for task {task_id} --------', flush=True)
        logits_output_dir = f'{self.args.logits_dir}/task_{task_id}'
        os.makedirs(logits_output_dir, exist_ok=True)

        t5_tokenizer = AutoTokenizer.from_pretrained(f'train/plms/{self.args.predictor_backbone_plm}')
        t5_tokenizer.add_tokens(['<', '<='])
        t5_vocab = t5_tokenizer.get_vocab()

        special_token_mapping = {
            '<|endoftext|>': '</s>'
        }

        def get_best_tok(candidates: dict):
            return list(candidates.keys())[0]

        def rebuild_sql_from_logits(logtis: list):
            toks = [get_best_tok(x) for x in logtis]
            return ''.join(toks)

        def refine_gpt_logits(logits: list):
            space_pattern = re.compile(r'\s\s+')

            for idx in range(len(logits) - 1):
                # process `Q:` `DB:`
                best_tok = get_best_tok(logits[idx])
                flag = False
                if best_tok in ['DB', ' DB', 'Q', ' Q'] and idx + 1 < len(logits):
                    next_best_tok = get_best_tok(logits[idx + 1])
                    if next_best_tok == ':':
                        logits = logits[:idx]
                        flag = True
                if flag:
                    break

            # remove bad first token.
            if get_best_tok(logits[0]).lower() not in ['select', ' select']:
                logits = logits[1:]

            # set first token
            if get_best_tok(logits[0]).lower() in ['select', ' select']:
                logits[0] = {'SELECT': 0}

            # replace `\n`, `  `, `   ` with ` `
            for idx in range(len(logits)):
                best_tok = get_best_tok(logits[idx])
                if best_tok in ['\n', '\n\n', '\t', '\xa0'] or space_pattern.match(best_tok):
                    logits[idx] = {' ': 0}

            # remove duplicate ` ` token
            to_remove_indices = []
            for idx in range(len(logits) - 1):
                best_tok = get_best_tok(logits[idx])
                if best_tok == ' ':
                    next_best_tok = get_best_tok(logits[idx + 1])
                    if next_best_tok.startswith(' '):
                        to_remove_indices.append(idx)
            logits = [x for i, x in enumerate(logits) if i not in to_remove_indices]

            # merge special tokens
            for idx, candidates in enumerate(logits):
                refined_candidates = {}
                for candidate in candidates.keys():
                    if candidate not in ['\n', '\n\n', '\t', '\xa0'] and not space_pattern.match(candidate):
                        refined_candidates[candidate] = candidates[candidate]

                logits[idx] = refined_candidates

            if get_best_tok(logits[-1]) == '<|endoftext|>':
                logits = logits[:-1]

            return logits

        for ds_name in ['train']:
            raw_file = f'{self.args.dataset_dir}/task_{task_id}/{ds_name}_seq2seq.jsonl'
            raw_examples = {}
            with open(raw_file, 'r', encoding='utf-8') as reader:
                for line in reader.readlines():
                    ex = json.loads(line)
                    guid = ex['example']['guid']
                    raw_examples[guid] = ex

            if self.args.dataset_type == 'spider':
                gpt_result_dir = f'train/data_train/spider/spider_gpt_perm_1/raw/task_{task_id}/{ds_name}'
            elif self.args.dataset_type == 'combine':
                gpt_result_dir = f'train/data_train/combine/combine1_gpt_perm_1/raw/task_{task_id}/{ds_name}'
            else:
                raise NotImplementedError
            with open(f'{gpt_result_dir}/db_id.txt', 'r', encoding='utf-8') as db_ids_file:
                db_id_lines = list(db_ids_file.readlines())
            with open(f'{gpt_result_dir}/gpt_logits.jsonl', 'r', encoding='utf-8') as logits_file:
                logits_lines = list(logits_file.readlines())
            with open(f'{gpt_result_dir}/gold_sql.txt', 'r', encoding='utf-8') as gold_sqls_file:
                gold_sql_lines = list(gold_sqls_file.readlines())

            guids = json.load(open(f'{gpt_result_dir}/prompt.json', 'r', encoding='utf-8'))
            guids = [x[0] for x in guids]

            to_process_elements = zip(guids, db_id_lines, logits_lines, gold_sql_lines)

            to_save_logits = {}
            refined_examples = []
            counter = 0
            for guid, db_id, logits, gold_sql in tqdm(to_process_elements):
                db_id, gold_sql = db_id.strip(), gold_sql.strip()
                logits = json.loads(logits)
                logits = refine_gpt_logits(logits)
                pred_sql = rebuild_sql_from_logits(logits)

                if evaluation_method(gold_sql, db_id, pred_sql):
                    counter += 1
                    raw_examples[guid]['sql'] = pred_sql
                    refined_examples.append(json.dumps(raw_examples[guid]))

                    t5_toks = t5_tokenizer.tokenize(pred_sql)
                    t5_tok_ids = t5_tokenizer(pred_sql, return_tensors='pt').input_ids[0]

                    gpt_toks = [get_best_tok(x) for x in logits]

                    t5_tok_indices, gpt_tok_indices = [], []  # indices in the original sql
                    t5_start_index, t5_end_index = 0, 0
                    reconstructed_t5_str, reconstructed_gpt_str = '', ''
                    for idx, tok in enumerate(t5_toks):
                        if idx == 0 and tok == '▁':
                            t5_tok_indices.append((0, 0))
                            continue
                        if idx == 0 and '▁' in tok:
                            tok = tok.replace('▁', '')
                        tok = tok.replace('▁', ' ')
                        reconstructed_t5_str += tok
                        t5_end_index = t5_start_index + len(tok)
                        t5_tok_indices.append((t5_start_index, t5_end_index))
                        t5_start_index = t5_end_index

                    gpt_start_index, gpt_end_index = 0, 0
                    for tok in gpt_toks:
                        reconstructed_gpt_str += tok
                        gpt_end_index = gpt_start_index + len(tok)
                        gpt_tok_indices.append((gpt_start_index, gpt_end_index))
                        gpt_start_index = gpt_end_index

                    assert t5_end_index == gpt_end_index
                    assert reconstructed_t5_str == reconstructed_gpt_str

                    same_indices = list(set(t5_tok_indices) & set(gpt_tok_indices))
                    t5_to_gpt = {}
                    for index in same_indices:
                        t5_idx = t5_tok_indices.index(index)
                        gpt_idx = gpt_tok_indices.index(index)
                        t5_to_gpt[t5_idx] = gpt_idx

                    seq_length = len(t5_tok_ids)
                    top_k_indices = torch.zeros((1, seq_length, 5), dtype=torch.int64)
                    top_k_logits = torch.zeros((1, seq_length, 5), dtype=torch.float32)

                    for idx, tok_id in enumerate(t5_tok_ids):
                        if idx in t5_to_gpt.keys():
                            gpt_idx = t5_to_gpt[idx]
                            candidates = logits[gpt_idx]
                            t5_indices, probs = [], []
                            for candidate, log_prob in candidates.items():
                                if candidate in special_token_mapping.keys():
                                    candidate = special_token_mapping[candidate]
                                candidate = candidate.replace(' ', '▁')
                                if candidate in t5_vocab:
                                    candidate_t5_tok_id = t5_tokenizer.convert_tokens_to_ids(candidate)
                                    # print(candidate, candidate_t5_tok_id)
                                    t5_indices.append(candidate_t5_tok_id)
                                    probs.append(math.exp(log_prob))

                            if len(t5_indices) > 0:
                                top_k_indices[0, idx, :len(t5_indices)] = torch.tensor(t5_indices)
                                top_k_logits[0, idx, :len(probs)] = torch.tensor(probs)
                            else:
                                # use hard label
                                top_k_indices[0, idx, 0] = tok_id
                                top_k_logits[0, idx, 0] = 1.0


                        else:
                            top_k_indices[0, idx, 0] = tok_id
                            top_k_logits[0, idx, 0] = 1.0

                else:
                    raw_examples[guid]['sql'] = gold_sql
                    refined_examples.append(json.dumps(raw_examples[guid]))

                    t5_tok_ids = t5_tokenizer(gold_sql, return_tensors='pt').input_ids[0]
                    seq_length = len(t5_tok_ids)
                    top_k_indices = torch.zeros((1, seq_length, 5), dtype=torch.int64)
                    top_k_logits = torch.zeros((1, seq_length, 5), dtype=torch.float32)
                    for idx, tok_id in enumerate(t5_tok_ids):
                        top_k_indices[0, idx, 0] = tok_id
                        top_k_logits[0, idx, 0] = 1.0

                to_save_logits[guid] = {'top_k_indices': top_k_indices, 'top_k_logits': top_k_logits}

            print(counter)

            with open(f'{self.args.dataset_dir}/task_{task_id}/{ds_name}_gpt_seq2seq.jsonl', 'w',
                      encoding='utf-8') as writer:
                writer.write('\n'.join(refined_examples))
            torch.save(to_save_logits, f'{logits_output_dir}/gpt_logits.pth')
