# -*- coding: utf-8 -*-
# file: dlcf_dca.py
# author: xumayi <xumayi@m.scnu.edu.cn>
# Copyright (C) 2021. All Rights Reserved.

import torch
import torch.nn as nn
import copy
import numpy as np
import math

from pytorch_transformers.modeling_bert import BertPooler, BertSelfAttention, BertConfig

def dependency_hidden(bert_local_out, depend, depended):
    depend_out = bert_local_out.clone()
    depended_out = bert_local_out.clone()
    for i in range(bert_local_out.size()[0]):
        for j in range(1,bert_local_out.size()[1]):
            if j-1 not in depend[i]:
                    depend_out[i][j] = depend_out[i][j] * 0
    for i in range(bert_local_out.size()[0]):
        for j in range(1,bert_local_out.size()[1]):
            if j-1 not in depended[i]:
                depended_out[i][j] = depended_out[i][j] * 0 
    return depend_out,depended_out

def weight_distrubute_local(bert_local_out,depend_weight,depended_weight,depend,depended,opt,no_connect):
    bert_local_out2 = torch.zeros_like(bert_local_out)
    for j in range(depend.size()[0]):
        bert_local_out2[j][0] = bert_local_out[j][0]  
    
    for j in range(depend.size()[0]):
        for i in range(depend.size()[1]):
            if depend[j][i] != -1 and (depend[j][i]+1) < opt.max_seq_len:
                bert_local_out2[j][depend[j][i]+1] = depend_weight[j].item() * bert_local_out[j][depend[j][i]+1]

    for j in range(depended.size()[0]):
        for i in range(depended.size()[1]):
            if depended[j][i] != -1 and (depended[j][i]+1) < opt.max_seq_len:
                bert_local_out2[j][depended[j][i]+1] = depended_weight[j].item() * bert_local_out[j][depended[j][i]+1]
    
    for j in range(no_connect.size()[0]):
        for i in range(no_connect.size()[1]):
            if no_connect[j][i] != -1 and (no_connect[j][i]+1) < opt.max_seq_len:
                bert_local_out2[j][no_connect[j][i]+1] = 0   
    
    return bert_local_out2


class PointwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''
    def __init__(self, d_hid, d_inner_hid=None,d_out=None, dropout=0):
        super(PointwiseFeedForward, self).__init__()
        if d_inner_hid is None:
            d_inner_hid = d_hid
        if d_out is None:
            d_out = d_inner_hid
        self.w_1 = nn.Conv1d(d_hid, d_inner_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_inner_hid, d_out, 1)  # position-wise
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.relu(self.w_1(x.transpose(1, 2)))
        output = self.w_2(output).transpose(2, 1)
        output = self.dropout(output)
        return output

class SelfAttention(nn.Module):
    def __init__(self, config,opt):
        super(SelfAttention, self).__init__()
        self.opt = opt
        self.config = config
        self.SA = BertSelfAttention(config)
        self.tanh = torch.nn.Tanh()

    def forward(self, inputs):
        zero_tensor = torch.tensor(np.zeros((inputs.size(0), 1, 1, self.opt.max_seq_len),
                                            dtype=np.float32), dtype=torch.float32).to(self.opt.device)
        SA_out,att = self.SA(inputs, zero_tensor)

        SA_out = self.tanh(SA_out)
        return SA_out,att

class DLCF_DCA(nn.Module):
    def __init__(self, model, opt):
        super(DLCF_DCA, self).__init__()
        if 'bert' in opt.pretrained_bert_name:
            hidden = model.config.hidden_size
        elif 'xlnet' in opt.pretrained_bert_name:
            hidden = model.config.d_model
        
        self.hidden = hidden
        sa_config = BertConfig(hidden_size=self.hidden,output_attentions=True)

        self.bert_spc = model

        self.opt = opt
        if opt.use_single_bert:
            self.bert_local = model
        else:
            self.bert_local = copy.deepcopy(model)


        self.dropout = nn.Dropout(opt.dropout)
        self.bert_sa = SelfAttention(sa_config,opt)


        self.mean_pooling_double = PointwiseFeedForward(hidden * 2, hidden,hidden)
        self.bert_pooler = BertPooler(sa_config)
        self.dense = nn.Linear(hidden, opt.polarities_dim)
        
        if opt.layer >= 1:
            self.bert_d_sa1 = SelfAttention(sa_config,opt)
            self.bert_d_pooler1 = BertPooler(sa_config)
            self.lin1 = nn.Sequential(
                nn.Linear(opt.bert_dim, opt.bert_dim * 2),
                nn.GELU(),
                nn.Linear(opt.bert_dim * 2, 1),
                nn.Sigmoid(),
            )
        if opt.layer >= 2:
            self.bert_d_sa2 = SelfAttention(sa_config,opt)
            self.bert_d_pooler2 = BertPooler(sa_config)
            self.lin2 = nn.Sequential(
                nn.Linear(opt.bert_dim, opt.bert_dim * 2),
                nn.GELU(),
                nn.Linear(opt.bert_dim * 2, 1),
                nn.Sigmoid(),
            )
        if opt.layer >= 3:
            self.bert_d_sa3 = SelfAttention(sa_config,opt)
            self.bert_d_pooler3 = BertPooler(sa_config)
            self.lin3 = nn.Sequential(
                nn.Linear(opt.bert_dim, opt.bert_dim * 2),
                nn.GELU(),
                nn.Linear(opt.bert_dim * 2, 1),
                nn.Sigmoid(),
            )
        if opt.layer >= 4:
            self.bert_d_sa4 = SelfAttention(sa_config,opt)
            self.bert_d_pooler4 = BertPooler(sa_config)
            self.lin4 = nn.Sequential(
                nn.Linear(opt.bert_dim, opt.bert_dim * 2),
                nn.GELU(),
                nn.Linear(opt.bert_dim * 2, 1),
                nn.Sigmoid(),
            )
        if opt.layer >= 5:
            self.bert_d_sa5 = SelfAttention(sa_config,opt)
            self.bert_d_pooler5 = BertPooler(sa_config)
            self.lin5 = nn.Sequential(
                nn.Linear(opt.bert_dim, opt.bert_dim * 2),
                nn.GELU(),
                nn.Linear(opt.bert_dim * 2, 1),
                nn.Sigmoid(),
            )
        if opt.layer >= 6:
            self.bert_d_sa6 = SelfAttention(sa_config,opt)
            self.bert_d_pooler6 = BertPooler(sa_config)
            self.lin6 = nn.Sequential(
                nn.Linear(opt.bert_dim, opt.bert_dim * 2),
                nn.GELU(),
                nn.Linear(opt.bert_dim * 2, 1),
                nn.Sigmoid(),
            )
        if opt.layer >= 7:
            self.bert_d_sa7 = SelfAttention(sa_config,opt)
            self.bert_d_pooler7 = BertPooler(sa_config)
            self.lin7 = nn.Sequential(
                nn.Linear(opt.bert_dim, opt.bert_dim * 2),
                nn.GELU(),
                nn.Linear(opt.bert_dim * 2, 1),
                nn.Sigmoid(),
            )

    def feature_dynamic_mask(self, text_local_indices, aspect_indices, max_dist, distances_input=None):
        texts = text_local_indices.cpu().numpy() # batch_size x seq_len
        asps = aspect_indices.cpu().numpy() # batch_size x aspect_len
        if distances_input is not None:
            distances_input = distances_input.cpu().numpy()
            
        mask_len = np.zeros(len(max_dist))
        for i in range(len(mask_len)):
            if max_dist[i].item() > 0:
                mask_len[i] = math.log(max_dist[i].item(),self.opt.a) + self.opt.a - 1
            else:
                mask_len[i] = 3
            
        masked_text_raw_indices = np.ones((text_local_indices.size(0), self.opt.max_seq_len, self.hidden),
                                          dtype=np.float32) # batch_size x seq_len x hidden size
        for text_i, asp_i in zip(range(len(texts)), range(len(asps))): # For each sample
            if distances_input is None:
                asp_len = np.count_nonzero(asps[asp_i]) # Calculate aspect length
                try:
                    asp_begin = np.argwhere(texts[text_i] == asps[asp_i][0])[0][0]
                except:
                    continue
                # Mask begin -> Relative position of an aspect vs the mask
                if asp_begin >= mask_len:
                    mask_begin = asp_begin - mask_len
                else:
                    mask_begin = 0
                for i in range(mask_begin): # Masking to the left
                    masked_text_raw_indices[text_i][i] = np.zeros((self.hidden), dtype=np.float)
                for j in range(asp_begin + asp_len + mask_len, self.opt.max_seq_len): # Masking to the right
                    masked_text_raw_indices[text_i][j] = np.zeros((self.hidden), dtype=np.float)
            else:
                distances_i = distances_input[text_i]
                for i,dist in enumerate(distances_i):
                    if dist > mask_len[text_i].item():
                        masked_text_raw_indices[text_i][i] = np.zeros((self.hidden), dtype=np.float)

        masked_text_raw_indices = torch.from_numpy(masked_text_raw_indices)
        return masked_text_raw_indices.to(self.opt.device)

    def feature_dynamic_weighted(self, text_local_indices, aspect_indices, max_dist, distances_input=None):
        texts = text_local_indices.cpu().numpy()
        asps = aspect_indices.cpu().numpy()
        if distances_input is not None:
            distances_input = distances_input.cpu().numpy()
            distances_inputs = np.asarray(distances_input,dtype=np.float32)
        masked_text_raw_indices = np.ones((text_local_indices.size(0), self.opt.max_seq_len, self.opt.bert_dim),
                                          dtype=np.float32) # batch x seq x dim

        mask_len = np.zeros(len(max_dist))
        for i in range(len(mask_len)):
            if max_dist[i].item() > 0:
                mask_len[i] = math.log(max_dist[i].item(),self.opt.a) + self.opt.a - 1
            else:
                mask_len[i] = 3
            
        for text_i, asp_i in zip(range(len(texts)), range(len(asps))):
            if distances_input is None:
                asp_len = np.count_nonzero(asps[asp_i]) - 2
                try:
                    asp_begin = np.argwhere(texts[text_i] == asps[asp_i][2])[0][0]
                    asp_avg_index = (asp_begin * 2 + asp_len) / 2 # central position
                except:
                    continue
                distances = np.zeros(np.count_nonzero(texts[text_i]), dtype=np.float32)
                for i in range(1, np.count_nonzero(texts[text_i])-1):
                    srd = abs(i - asp_avg_index) + asp_len / 2
                    if srd > self.opt.SRD:
                        distances[i] = 1 - (srd - self.opt.SRD)/np.count_nonzero(texts[text_i])
                    else:
                        distances[i] = 1
                for i in range(len(distances)):
                    masked_text_raw_indices[text_i][i] = masked_text_raw_indices[text_i][i] * distances[i]
            else:
                distances_i = distances_inputs[text_i] # distances of batch i-th
                for i,dist in enumerate(distances_i):
                    if dist > mask_len[text_i].item():
                        try:
                            if max_dist[text_i] >= dist:
                                distances_i[i] = 1 - dist/max_dist[text_i]
                            else:
                                distances_i[i] = 0 
                        except:
                            distances_i[i] = 1                         
                    else:
                        distances_i[i] = 1
                for i in range(len(distances_i)):
                    masked_text_raw_indices[text_i][i] = masked_text_raw_indices[text_i][i] * distances_i[i]

        masked_text_raw_indices = torch.from_numpy(masked_text_raw_indices)
        return masked_text_raw_indices.to(self.opt.device)
    
    def weight_calculate1(self, d_w, ded_w, depend_out, depended_out):     
        depend_sa_out,_ =  self.bert_d_sa1(depend_out)
        depend_sa_out = self.dropout(depend_sa_out)
        depended_sa_out,_ =  self.bert_d_sa1(depended_out)
        depended_sa_out = self.dropout(depended_sa_out)

        depend_pool_out = self.bert_d_pooler1(depend_sa_out)
        depend_pool_out = self.dropout(depend_pool_out)
        depended_pool_out = self.bert_d_pooler1(depended_sa_out)
        depended_pool_out = self.dropout(depended_pool_out)

        depend_weight = self.lin1(depend_pool_out)
        depend_weight = self.dropout(depend_weight)
        depended_weight = self.lin1(depended_pool_out)
        depended_weight = self.dropout(depended_weight)
    
        for i in range(depend_weight.size()[0]):
            depend_weight[i] = depend_weight[i].item() * d_w[i].item()
            depended_weight[i] = depended_weight[i].item() * ded_w[i].item()
            weight_sum = depend_weight[i].item() + depended_weight[i].item()
            if weight_sum != 0:
                depend_weight[i] = (2 * depend_weight[i] / weight_sum) ** self.opt.power
                if depend_weight[i] > 2:
                    depend_weight[i] = 2
                depended_weight[i] = (2 * depended_weight[i] / weight_sum) ** self.opt.power
                if depended_weight[i] > 2:
                    depended_weight[i] = 2
            else:
                depend_weight[i] = 1
                depended_weight[i] = 1
        return depend_weight, depended_weight
    
    def weight_calculate2(self, d_w, ded_w, depend_out, depended_out):     
        depend_sa_out,_ =  self.bert_d_sa2(depend_out)
        depend_sa_out = self.dropout(depend_sa_out)
        depended_sa_out,_ =  self.bert_d_sa2(depended_out)
        depended_sa_out = self.dropout(depended_sa_out)

        depend_pool_out = self.bert_d_pooler2(depend_sa_out)
        depend_pool_out = self.dropout(depend_pool_out)
        depended_pool_out = self.bert_d_pooler2(depended_sa_out)
        depended_pool_out = self.dropout(depended_pool_out)

        depend_weight = self.lin2(depend_pool_out)
        depend_weight = self.dropout(depend_weight)
        depended_weight = self.lin2(depended_pool_out)
        depended_weight = self.dropout(depended_weight)
    
        for i in range(depend_weight.size()[0]):
            depend_weight[i] = depend_weight[i].item() * d_w[i].item()
            depended_weight[i] = depended_weight[i].item() * ded_w[i].item()
            weight_sum = depend_weight[i].item() + depended_weight[i].item()
            if weight_sum != 0:
                depend_weight[i] = (2 * depend_weight[i] / weight_sum) ** self.opt.power
                if depend_weight[i] > 2:
                    depend_weight[i] = 2
                depended_weight[i] = (2 * depended_weight[i] / weight_sum) ** self.opt.power
                if depended_weight[i] > 2:
                    depended_weight[i] = 2
            else:
                depend_weight[i] = 1
                depended_weight[i] = 1
        return depend_weight, depended_weight
    
    def weight_calculate3(self, d_w, ded_w, depend_out, depended_out):     
        depend_sa_out,_ =  self.bert_d_sa3(depend_out)
        depend_sa_out = self.dropout(depend_sa_out)
        depended_sa_out,_ =  self.bert_d_sa3(depended_out)
        depended_sa_out = self.dropout(depended_sa_out)

        depend_pool_out = self.bert_d_pooler3(depend_sa_out)
        depend_pool_out = self.dropout(depend_pool_out)
        depended_pool_out = self.bert_d_pooler3(depended_sa_out)
        depended_pool_out = self.dropout(depended_pool_out)

        depend_weight = self.lin3(depend_pool_out)
        depend_weight = self.dropout(depend_weight)
        depended_weight = self.lin3(depended_pool_out)
        depended_weight = self.dropout(depended_weight)
    
        for i in range(depend_weight.size()[0]):
            depend_weight[i] = depend_weight[i].item() * d_w[i].item()
            depended_weight[i] = depended_weight[i].item() * ded_w[i].item()
            weight_sum = depend_weight[i].item() + depended_weight[i].item()
            if weight_sum != 0:
                depend_weight[i] = (2 * depend_weight[i] / weight_sum) ** self.opt.power
                if depend_weight[i] > 2:
                    depend_weight[i] = 2
                depended_weight[i] = (2 * depended_weight[i] / weight_sum) ** self.opt.power
                if depended_weight[i] > 2:
                    depended_weight[i] = 2
            else:
                depend_weight[i] = 1
                depended_weight[i] = 1
        return depend_weight, depended_weight

    def weight_calculate4(self, d_w, ded_w, depend_out, depended_out):     
        depend_sa_out,_ =  self.bert_d_sa4(depend_out)
        depend_sa_out = self.dropout(depend_sa_out)
        depended_sa_out,_ =  self.bert_d_sa4(depended_out)
        depended_sa_out = self.dropout(depended_sa_out)

        depend_pool_out = self.bert_d_pooler4(depend_sa_out)
        depend_pool_out = self.dropout(depend_pool_out)
        depended_pool_out = self.bert_d_pooler4(depended_sa_out)
        depended_pool_out = self.dropout(depended_pool_out)

        depend_weight = self.lin4(depend_pool_out)
        depend_weight = self.dropout(depend_weight)
        depended_weight = self.lin4(depended_pool_out)
        depended_weight = self.dropout(depended_weight)
    
        for i in range(depend_weight.size()[0]):
            depend_weight[i] = depend_weight[i].item() * d_w[i].item()
            depended_weight[i] = depended_weight[i].item() * ded_w[i].item()
            weight_sum = depend_weight[i].item() + depended_weight[i].item()
            if weight_sum != 0:
                depend_weight[i] = (2 * depend_weight[i] / weight_sum) ** self.opt.power
                if depend_weight[i] > 2:
                    depend_weight[i] = 2
                depended_weight[i] = (2 * depended_weight[i] / weight_sum) ** self.opt.power
                if depended_weight[i] > 2:
                    depended_weight[i] = 2
            else:
                depend_weight[i] = 1
                depended_weight[i] = 1
        return depend_weight, depended_weight
    
    def weight_calculate5(self, d_w, ded_w, depend_out, depended_out):     
        depend_sa_out,_ =  self.bert_d_sa5(depend_out)
        depend_sa_out = self.dropout(depend_sa_out)
        depended_sa_out,_ =  self.bert_d_sa5(depended_out)
        depended_sa_out = self.dropout(depended_sa_out)

        depend_pool_out = self.bert_d_pooler5(depend_sa_out)
        depend_pool_out = self.dropout(depend_pool_out)
        depended_pool_out = self.bert_d_pooler5(depended_sa_out)
        depended_pool_out = self.dropout(depended_pool_out)

        depend_weight = self.lin5(depend_pool_out)
        depend_weight = self.dropout(depend_weight)
        depended_weight = self.lin5(depended_pool_out)
        depended_weight = self.dropout(depended_weight)
    
        for i in range(depend_weight.size()[0]):
            depend_weight[i] = depend_weight[i].item() * d_w[i].item()
            depended_weight[i] = depended_weight[i].item() * ded_w[i].item()
            weight_sum = depend_weight[i].item() + depended_weight[i].item()
            if weight_sum != 0:
                depend_weight[i] = (2 * depend_weight[i] / weight_sum) ** self.opt.power
                if depend_weight[i] > 2:
                    depend_weight[i] = 2
                depended_weight[i] = (2 * depended_weight[i] / weight_sum) ** self.opt.power
                if depended_weight[i] > 2:
                    depended_weight[i] = 2
            else:
                depend_weight[i] = 1
                depended_weight[i] = 1
        return depend_weight, depended_weight

    def weight_calculate6(self, d_w, ded_w, depend_out, depended_out):     
        depend_sa_out,_ =  self.bert_d_sa6(depend_out)
        depend_sa_out = self.dropout(depend_sa_out)
        depended_sa_out,_ =  self.bert_d_sa6(depended_out)
        depended_sa_out = self.dropout(depended_sa_out)

        depend_pool_out = self.bert_d_pooler6(depend_sa_out)
        depend_pool_out = self.dropout(depend_pool_out)
        depended_pool_out = self.bert_d_pooler6(depended_sa_out)
        depended_pool_out = self.dropout(depended_pool_out)

        depend_weight = self.lin6(depend_pool_out)
        depend_weight = self.dropout(depend_weight)
        depended_weight = self.lin6(depended_pool_out)
        depended_weight = self.dropout(depended_weight)
    
        for i in range(depend_weight.size()[0]):
            depend_weight[i] = depend_weight[i].item() * d_w[i].item()
            depended_weight[i] = depended_weight[i].item() * ded_w[i].item()
            weight_sum = depend_weight[i].item() + depended_weight[i].item()
            if weight_sum != 0:
                depend_weight[i] = (2 * depend_weight[i] / weight_sum) ** self.opt.power
                if depend_weight[i] > 2:
                    depend_weight[i] = 2
                depended_weight[i] = (2 * depended_weight[i] / weight_sum) ** self.opt.power
                if depended_weight[i] > 2:
                    depended_weight[i] = 2
            else:
                depend_weight[i] = 1
                depended_weight[i] = 1
        return depend_weight, depended_weight
    
    def weight_calculate7(self, d_w, ded_w, depend_out, depended_out):     
        depend_sa_out,_ =  self.bert_d_sa7(depend_out)
        depend_sa_out = self.dropout(depend_sa_out)
        depended_sa_out,_ =  self.bert_d_sa7(depended_out)
        depended_sa_out = self.dropout(depended_sa_out)

        depend_pool_out = self.bert_d_pooler7(depend_sa_out)
        depend_pool_out = self.dropout(depend_pool_out)
        depended_pool_out = self.bert_d_pooler7(depended_sa_out)
        depended_pool_out = self.dropout(depended_pool_out)

        depend_weight = self.lin7(depend_pool_out)
        depend_weight = self.dropout(depend_weight)
        depended_weight = self.lin7(depended_pool_out)
        depended_weight = self.dropout(depended_weight)
    
        for i in range(depend_weight.size()[0]):
            depend_weight[i] = depend_weight[i].item() * d_w[i].item()
            depended_weight[i] = depended_weight[i].item() * ded_w[i].item()
            weight_sum = depend_weight[i].item() + depended_weight[i].item()
            if weight_sum != 0:
                depend_weight[i] = (2 * depend_weight[i] / weight_sum) ** self.opt.power
                if depend_weight[i] > 2:
                    depend_weight[i] = 2
                depended_weight[i] = (2 * depended_weight[i] / weight_sum) ** self.opt.power
                if depended_weight[i] > 2:
                    depended_weight[i] = 2
            else:
                depend_weight[i] = 1
                depended_weight[i] = 1
        return depend_weight, depended_weight
    
    def forward(self, inputs, output_attentions = False):
        text_bert_indices = inputs[0]
        bert_segments_ids = inputs[1]
        text_local_indices = inputs[2] # Raw text without adding aspect term
        aspect_indices = inputs[3] # Raw text of aspect
        distances = inputs[4]
        depend = inputs[5]
        depended = inputs[6]
        no_connect = inputs[7]
        max_dist = inputs[8]

        spc_out = self.bert_spc(text_bert_indices, bert_segments_ids)
        bert_spc_out = spc_out[0]
        spc_att = spc_out[-1][-1]

        bert_local_out = self.bert_local(text_local_indices)[0]
        
        if self.opt.local_context_focus == 'cdm':
            masked_local_text_vec = self.feature_dynamic_mask(text_local_indices, aspect_indices, max_dist ,distances)
            bert_local_out = torch.mul(bert_local_out, masked_local_text_vec)

        elif self.opt.local_context_focus == 'cdw':
            weighted_text_local_features = self.feature_dynamic_weighted(text_local_indices, aspect_indices, max_dist, distances)
            bert_local_out = torch.mul(bert_local_out, weighted_text_local_features)    
        
        
        depend_weight = torch.ones(bert_local_out.size()[0])
        depended_weight = torch.ones(bert_local_out.size()[0])
        
        if self.opt.layer ==1 :
            depend_out,depended_out = dependency_hidden(bert_local_out, depend, depended)
            depend_weight, depended_weight = self.weight_calculate1(depend_weight, depended_weight, depend_out, depended_out)
            bert_local_out = weight_distrubute_local(bert_local_out,depend_weight,depended_weight,depend,depended,self.opt,no_connect)
        elif self.opt.layer ==2 :
            depend_out,depended_out = dependency_hidden(bert_local_out, depend, depended)
            depend_weight, depended_weight = self.weight_calculate1(depend_weight, depended_weight, depend_out, depended_out)
            bert_local_out = weight_distrubute_local(bert_local_out,depend_weight,depended_weight,depend,depended,self.opt,no_connect)
            depend_out,depended_out = dependency_hidden(bert_local_out, depend, depended)
            depend_weight, depended_weight = self.weight_calculate2(depend_weight, depended_weight, depend_out, depended_out)
            bert_local_out = weight_distrubute_local(bert_local_out,depend_weight,depended_weight,depend,depended,self.opt,no_connect)
        elif self.opt.layer ==3 :
            depend_out,depended_out = dependency_hidden(bert_local_out, depend, depended)
            depend_weight, depended_weight = self.weight_calculate1(depend_weight, depended_weight, depend_out, depended_out)
            bert_local_out = weight_distrubute_local(bert_local_out,depend_weight,depended_weight,depend,depended,self.opt,no_connect)
            depend_out,depended_out = dependency_hidden(bert_local_out, depend, depended)
            depend_weight, depended_weight = self.weight_calculate2(depend_weight, depended_weight, depend_out, depended_out)
            bert_local_out = weight_distrubute_local(bert_local_out,depend_weight,depended_weight,depend,depended,self.opt,no_connect)
            depend_out,depended_out = dependency_hidden(bert_local_out, depend, depended)
            depend_weight, depended_weight = self.weight_calculate3(depend_weight, depended_weight, depend_out, depended_out)
            bert_local_out = weight_distrubute_local(bert_local_out,depend_weight,depended_weight,depend,depended,self.opt,no_connect)
        elif self.opt.layer ==4 :
            depend_out,depended_out = dependency_hidden(bert_local_out, depend, depended)
            depend_weight, depended_weight = self.weight_calculate1(depend_weight, depended_weight, depend_out, depended_out)
            bert_local_out = weight_distrubute_local(bert_local_out,depend_weight,depended_weight,depend,depended,self.opt,no_connect)
            depend_out,depended_out = dependency_hidden(bert_local_out, depend, depended)
            depend_weight, depended_weight = self.weight_calculate2(depend_weight, depended_weight, depend_out, depended_out)
            bert_local_out = weight_distrubute_local(bert_local_out,depend_weight,depended_weight,depend,depended,self.opt,no_connect)
            depend_out,depended_out = dependency_hidden(bert_local_out, depend, depended)
            depend_weight, depended_weight = self.weight_calculate3(depend_weight, depended_weight, depend_out, depended_out)
            bert_local_out = weight_distrubute_local(bert_local_out,depend_weight,depended_weight,depend,depended,self.opt,no_connect)
            depend_out,depended_out = dependency_hidden(bert_local_out, depend, depended)
            depend_weight, depended_weight = self.weight_calculate4(depend_weight, depended_weight, depend_out, depended_out)
            bert_local_out = weight_distrubute_local(bert_local_out,depend_weight,depended_weight,depend,depended,self.opt,no_connect)
        elif self.opt.layer ==5 :
            depend_out,depended_out = dependency_hidden(bert_local_out, depend, depended)
            depend_weight, depended_weight = self.weight_calculate1(depend_weight, depended_weight, depend_out, depended_out)
            bert_local_out = weight_distrubute_local(bert_local_out,depend_weight,depended_weight,depend,depended,self.opt,no_connect)
            depend_out,depended_out = dependency_hidden(bert_local_out, depend, depended)
            depend_weight, depended_weight = self.weight_calculate2(depend_weight, depended_weight, depend_out, depended_out)
            bert_local_out = weight_distrubute_local(bert_local_out,depend_weight,depended_weight,depend,depended,self.opt,no_connect)
            depend_out,depended_out = dependency_hidden(bert_local_out, depend, depended)
            depend_weight, depended_weight = self.weight_calculate3(depend_weight, depended_weight, depend_out, depended_out)
            bert_local_out = weight_distrubute_local(bert_local_out,depend_weight,depended_weight,depend,depended,self.opt,no_connect)
            depend_out,depended_out = dependency_hidden(bert_local_out, depend, depended)
            depend_weight, depended_weight = self.weight_calculate4(depend_weight, depended_weight, depend_out, depended_out)
            bert_local_out = weight_distrubute_local(bert_local_out,depend_weight,depended_weight,depend,depended,self.opt,no_connect)
            depend_out,depended_out = dependency_hidden(bert_local_out, depend, depended)
            depend_weight, depended_weight = self.weight_calculate5(depend_weight, depended_weight, depend_out, depended_out)
            bert_local_out = weight_distrubute_local(bert_local_out,depend_weight,depended_weight,depend,depended,self.opt,no_connect)
        elif self.opt.layer ==6 :
            depend_out,depended_out = dependency_hidden(bert_local_out, depend, depended)
            depend_weight, depended_weight = self.weight_calculate1(depend_weight, depended_weight, depend_out, depended_out)
            bert_local_out = weight_distrubute_local(bert_local_out,depend_weight,depended_weight,depend,depended,self.opt,no_connect)
            depend_out,depended_out = dependency_hidden(bert_local_out, depend, depended)
            depend_weight, depended_weight = self.weight_calculate2(depend_weight, depended_weight, depend_out, depended_out)
            bert_local_out = weight_distrubute_local(bert_local_out,depend_weight,depended_weight,depend,depended,self.opt,no_connect)
            depend_out,depended_out = dependency_hidden(bert_local_out, depend, depended)
            depend_weight, depended_weight = self.weight_calculate3(depend_weight, depended_weight, depend_out, depended_out)
            bert_local_out = weight_distrubute_local(bert_local_out,depend_weight,depended_weight,depend,depended,self.opt,no_connect)
            depend_out,depended_out = dependency_hidden(bert_local_out, depend, depended)
            depend_weight, depended_weight = self.weight_calculate4(depend_weight, depended_weight, depend_out, depended_out)
            bert_local_out = weight_distrubute_local(bert_local_out,depend_weight,depended_weight,depend,depended,self.opt,no_connect)
            depend_out,depended_out = dependency_hidden(bert_local_out, depend, depended)
            depend_weight, depended_weight = self.weight_calculate5(depend_weight, depended_weight, depend_out, depended_out)
            bert_local_out = weight_distrubute_local(bert_local_out,depend_weight,depended_weight,depend,depended,self.opt,no_connect)
            depend_out,depended_out = dependency_hidden(bert_local_out, depend, depended)
            depend_weight, depended_weight = self.weight_calculate6(depend_weight, depended_weight, depend_out, depended_out)
            bert_local_out = weight_distrubute_local(bert_local_out,depend_weight,depended_weight,depend,depended,self.opt,no_connect)
        elif self.opt.layer ==7 :
            depend_out,depended_out = dependency_hidden(bert_local_out, depend, depended)
            depend_weight, depended_weight = self.weight_calculate1(depend_weight, depended_weight, depend_out, depended_out)
            bert_local_out = weight_distrubute_local(bert_local_out,depend_weight,depended_weight,depend,depended,self.opt,no_connect)
            depend_out,depended_out = dependency_hidden(bert_local_out, depend, depended)
            depend_weight, depended_weight = self.weight_calculate2(depend_weight, depended_weight, depend_out, depended_out)
            bert_local_out = weight_distrubute_local(bert_local_out,depend_weight,depended_weight,depend,depended,self.opt,no_connect)
            depend_out,depended_out = dependency_hidden(bert_local_out, depend, depended)
            depend_weight, depended_weight = self.weight_calculate3(depend_weight, depended_weight, depend_out, depended_out)
            bert_local_out = weight_distrubute_local(bert_local_out,depend_weight,depended_weight,depend,depended,self.opt,no_connect)
            depend_out,depended_out = dependency_hidden(bert_local_out, depend, depended)
            depend_weight, depended_weight = self.weight_calculate4(depend_weight, depended_weight, depend_out, depended_out)
            bert_local_out = weight_distrubute_local(bert_local_out,depend_weight,depended_weight,depend,depended,self.opt,no_connect)
            depend_out,depended_out = dependency_hidden(bert_local_out, depend, depended)
            depend_weight, depended_weight = self.weight_calculate5(depend_weight, depended_weight, depend_out, depended_out)
            bert_local_out = weight_distrubute_local(bert_local_out,depend_weight,depended_weight,depend,depended,self.opt,no_connect)
            depend_out,depended_out = dependency_hidden(bert_local_out, depend, depended)
            depend_weight, depended_weight = self.weight_calculate6(depend_weight, depended_weight, depend_out, depended_out)
            bert_local_out = weight_distrubute_local(bert_local_out,depend_weight,depended_weight,depend,depended,self.opt,no_connect)
            depend_out,depended_out = dependency_hidden(bert_local_out, depend, depended)
            depend_weight, depended_weight = self.weight_calculate7(depend_weight, depended_weight, depend_out, depended_out)
            bert_local_out = weight_distrubute_local(bert_local_out,depend_weight,depended_weight,depend,depended,self.opt,no_connect)
        
        out_cat = torch.cat((bert_local_out, bert_spc_out), dim=-1)
        mean_pool = self.mean_pooling_double(out_cat)
        self_attention_out, local_att = self.bert_sa(mean_pool)
        pooled_out = self.bert_pooler(self_attention_out)
        dense_out = self.dense(pooled_out)
        if output_attentions:
            return (dense_out,spc_att,local_att)
        return dense_out