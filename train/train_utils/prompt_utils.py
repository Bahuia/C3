# modify some classes from OpenPrompt
from typing import Tuple, Any

import torch.nn
from openprompt.pipeline_base import *
from torch.nn import MarginRankingLoss, CrossEntropyLoss, KLDivLoss


# class PromptForRetrieverGeneration(PromptForGeneration):
#     def __init__(self, plm: PreTrainedModel,
#                  template: Template,
#                  freeze_plm: bool = False,
#                  plm_eval_mode: bool = False,
#                  gen_config: Optional[CfgNode] = None,
#                  tokenizer: Optional[PreTrainedTokenizer] = None, ):
#         super().__init__(plm, template, freeze_plm, plm_eval_mode, gen_config, tokenizer)
#
#         # self.loss_fct = MarginRankingLoss(margin=0.1)
#         self.loss_fct = nn.CrossEntropyLoss(reduction='none')
#
#     def _forward(self, batch: Union[Dict, InputFeatures]) -> tuple[Any, Any]:
#         r"""
#         This is the forward method of the training of generation in prompt-learning framework.
#
#         Args:
#             batch (:obj:`Union[Dict, InputFeatures]`): The input features of batchified data sequences.
#
#         Returns:
#             loss(:obj:torch.Tensor): The loss of the current generation procedure.
#         """
#         outputs = self.prompt_model(batch)
#         logits = outputs.logits
#         if isinstance(self.loss_fct, MarginRankingLoss):
#             special_token_index = 32089  # corresponding to the special token <extra_id_10>
#             special_token_logits = logits[:, 1, special_token_index]
#             best_context_score = special_token_logits[0]
#             # bad_context_scores = torch.sigmoid(special_token_logits[1:])
#             bad_context_scores = special_token_logits[1:]
#             num_pairs = bad_context_scores.size(0)
#             # best_context_scores = torch.sigmoid(best_context_score.repeat(num_pairs))
#             best_context_scores = best_context_score.repeat(num_pairs)
#             labels = torch.ones(num_pairs).cuda()
#             loss = self.loss_fct(best_context_scores, bad_context_scores, labels)
#
#         elif isinstance(self.loss_fct, CrossEntropyLoss):
#             if self.config.is_encoder_decoder:
#                 reference_ids = batch['decoder_input_ids']
#             else:
#                 reference_ids = batch['input_ids']  # in case in some template, these field is dropped
#             logits, labels = self.shift_logits_and_labels(logits, batch['loss_ids'], reference_ids)
#             batch_size, seq_len, vocab_size = logits.shape
#             '''
#             true_token_id, false_token_id = 1176, 6136
#             true_logits, false_logits = logits[:, 1, true_token_id:true_token_id + 1], \
#                 logits[:, 1, false_token_id:false_token_id + 1]
#             eval_logits = torch.cat((true_logits, false_logits), dim=-1)
#             # print(eval_logits)
#             # print(labels)
#             labels = labels[:, 1]
#             labels[labels == true_token_id] = 0
#             labels[labels == false_token_id] = 1
#             # print(labels)
#             loss = self.loss_fct(eval_logits.view(-1, eval_logits.size(-1)), labels.view(-1))
#             '''
#             loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
#
#             loss = loss.view(batch_size, -1).sum(dim=-1)  # TODO support more objectives
#             loss = loss.mean()
#
#         else:
#             raise NotImplementedError
#
#         return logits, loss

class PromptForPredictorGeneration(PromptForGeneration):
    def __init__(self, plm: PreTrainedModel,
                 template: Template,
                 freeze_plm: bool = False,
                 plm_eval_mode: bool = False,
                 gen_config: Optional[CfgNode] = None,
                 tokenizer: Optional[PreTrainedTokenizer] = None, ):
        super().__init__(plm, template, freeze_plm, plm_eval_mode, gen_config, tokenizer)

        self.loss_fct = nn.CrossEntropyLoss(reduction='none')

    def _forward(self, batch: Union[Dict, InputFeatures]) -> torch.Tensor:
        r"""
        This is the forward method of the training of generation in prompt-learning framework.

        Args:
            batch (:obj:`Union[Dict, InputFeatures]`): The input features of batchified data sequences.

        Returns:
            loss(:obj:torch.Tensor): The loss of the current generation procedure.
        """
        outputs = self.prompt_model(batch)
        logits = outputs.logits
        if self.config.is_encoder_decoder:
            reference_ids = batch['decoder_input_ids']
        else:
            reference_ids = batch['input_ids']  # in case in some template, these field is dropped
        logits, labels = self.shift_logits_and_labels(logits, batch['loss_ids'], reference_ids)

        return logits
