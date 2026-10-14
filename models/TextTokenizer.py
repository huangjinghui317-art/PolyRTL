# coding=utf-8
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
import numpy as np
import random, json
from accelerate import init_empty_weights
from transformers import AutoTokenizer
from transformers import BertConfig, BertForMaskedLM


class TextTokenizer_NoCA(nn.Module):
    ''' without cross attention'''
    def __init__(self, config, device):
        super(TextTokenizer_NoCA, self).__init__()

        self.device = device
        self.config = config

        bert_model = config['bert_model_name']
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.summary_enc = BertForMaskedLM.from_pretrained(bert_model)
        # bert_config = BertConfig.from_json_file(config['bert_config'])
        # self.summary_enc = BertForMaskedLM.from_pretrained(bert_model, config=bert_config)

        self.embed_dim = config['embed_dim']
        summary_embed_dim = self.summary_enc.config.hidden_size
        self.sumamry_proj = nn.Linear(summary_embed_dim, self.embed_dim)
        self.mlm_probability = config['mlm_probability']

    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()

        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False

        if targets is not None:
            targets[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        if targets is not None:
            return input_ids, targets
        else:
            return input_ids

    def forward(self, data):
        # convert string(summary_data) to id_dict(input: input_ids, attention_mask)
        input = self.tokenizer(data, return_tensors="pt", padding='longest', truncation=True,
                               max_length=512).to(self.device)
        output = self.summary_enc.bert(input.input_ids, attention_mask=input.attention_mask, return_dict=True)
        # output.last_hidden_state: (batch_size, seq_length+1, hidden_size)
        # output.last_hidden_state[:, 0, :]-->[CLS]: (batch_size, hidden_size)
        # embeds: (batch_size, hidden_size)-->(batch_size, self.embed_dim)
        embeds = F.normalize(self.sumamry_proj(output.last_hidden_state[:, 0, :]), dim=-1)

        return embeds, input, output

    def mlm_forward(self, data):
        summary_in = self.tokenizer(data, return_tensors="pt", padding='longest', truncation=True,
                                    max_length=512).to(self.device)
        summary_ids = summary_in.input_ids.clone()
        labels = summary_ids.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        input_ids, labels = self.mask(summary_ids, self.summary_enc.config.vocab_size, self.device, targets=labels,
                                      probability_matrix=probability_matrix)

        mlm_output = self.summary_enc(input_ids, attention_mask=summary_in.attention_mask, return_dict=True, labels=labels)
        loss_mlm = mlm_output.loss

        return loss_mlm

