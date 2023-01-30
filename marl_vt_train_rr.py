import os
from random import choice
import sys
import math
import time
import copy
import logging
import datetime
from turtle import st
import numpy as np
import multiprocessing
from pprint import pformat
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain
from typing_extensions import TypeAlias
from ignite.distributed.utils import device
from ignite.engine.events import RemovableEventHandle

import torch
from torch._C import AnyType
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.modules.loss import TripletMarginLoss
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from transformers import GPT2Tokenizer, AdamW, BartTokenizer
from transformers.file_utils import (CONFIG_NAME, WEIGHTS_NAME)
from VideoGPT2 import VideoGPT2LMHeadModel
import pickle as pkl

from du_rl.Graph import Knowledge_graph
from du_rl.Baseline import ReactiveBaseline
from du_rl.vcls_data import Video_data_loader
from du_rl.text_data import Text_data_loader
from du_rl.Agent import Agent
from du_rl.video_options import read_options
from du_rl.text_options import read_text_options
from du_rl.vrd_utils import calc_cum_discounted_reward, calc_cum_discounted_reward_credit, calc_reinforce_loss, rouge_n, bleu_corpus
from du_rl.randomreward import RR, RR_p


SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>","<cap>", "<video>", "<pad>", "<vrd>", "<trd>", "<start>", "<equal>"]
SPECIAL_TOKENS_DICT = {'bos_token': "<bos>", 'eos_token': "<eos>", 'additional_special_tokens': ["<speaker1>", "<speaker2>", "<video>", "<cap>", "<vrd>", "<trd>", "<start>","<equal>"], 'pad_token': "<pad>"}
MODEL_INPUTS = ["input_ids", "token_type_ids","lm_labels"]
PADDED_INPUTS = ["input_ids", "token_type_ids","lm_labels"]

logger = logging.getLogger(__file__)

def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()

def padding(seq, pad_token):
    max_len = max([i.size(0) for i in seq])
    if len(seq[0].size()) == 1:
        result = torch.ones((len(seq), max_len)).long() * pad_token
    else:
        result = torch.ones((len(seq), max_len, seq[0].size(-1))).float()
    for i in range(len(seq)):
        result[i, :seq[i].size(0)] = seq[i]
    return result

def get_data_loaders_new(args, tokenizer):
    from marl_vt_dataset import get_dataset, AVSDDataSet, collate_fn

    if args.drop_rate == 1:
        train_data_path = 'pkls/marl_con_vrd_train_data.pkl'
    else:
        train_data_path = 'pkls/marl_cap_con_vrd_train_data.pkl'

    if not os.path.exists(train_data_path):
        if args.drop_rate == 1:
            train_data = get_dataset(tokenizer, args.train_path, args.fea_path, args.context_traj_path, args.video_traj_path, n_history=args.max_history)
        else:
            train_data = get_dataset(tokenizer, args.train_path, args.fea_path, args.caption_traj_path, args.video_traj_path, n_history=args.max_history)
        """
        train_data[0] dialog_list: num of dialogs * num of turns [{'vid':'','history':max 3 turns [[q],[a],...],'answer':[a],'caption':[[caption list], [summary list]]}]
        train_data[1] all_feature dict
        """
        with open(train_data_path, 'wb') as f:
            pkl.dump(train_data, f)
    else:
        with open(train_data_path, 'rb') as f:
            train_data = pkl.load(f)

    if args.video: 
        train_dataset = AVSDDataSet(train_data[0], tokenizer, (train_data[1], None), drop_rate=args.drop_rate, train=True, model=args.model)
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=args.train_batch_size, shuffle=(not args.distributed), collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id, features=True))
    else:
        train_dataset = AVSDDataSet(train_data[0], tokenizer, None, drop_rate=args.drop_rate, train=True, model=args.model)
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=args.train_batch_size, shuffle=(not args.distributed), collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id, features=None))
    return train_loader

def train():
    parser = ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/avsd/train_set4DSTC7-AVSD.json", help="Path of the trainset")
    parser.add_argument("--fea_path", type=str, default="data/avsd/", help="Path of the trainset")
    parser.add_argument("--context_traj_path", type=str, default="data/mm-graph/semantic_graph/avsd_context/con_trip.json")
    parser.add_argument("--caption_traj_path", type=str, default="data/mm-graph/semantic_graph/avsd_cap_con/cap_con_trip.json")
    parser.add_argument("--video_traj_path", type=str, default="data/mm-graph/semantic_graph/avsd_video/cls_trip.json")
    parser.add_argument("--valid_path", type=str, default="data/avsd/valid_set4DSTC7-AVSD.json", help="Path of the validset")
    parser.add_argument("--max_history", type=int, default=3, help="Number of previous exchanges to keep in history")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=4, help="Batch size for validation")
    parser.add_argument("--drop_rate", type=float, default=1, help="drop rate for caption")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--eval_before_start", action='store_true', help="If true start with a first evaluation before training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="", help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--gpuid", type=str, default='', help='select proper gpu id')
    parser.add_argument("--model", type=str, default='gpt2', help='Pretrained Model name')
    parser.add_argument('--video', type=int, default=1, help='if use video: 1 use 0 not')
    parser.add_argument('--exp_set', type=str, default='test')
    parser.add_argument('--top_n', type=int, default=1)
    parser.add_argument('--reward', type=str, default='rouge', choices=['rouge', 'gpt'])
    parser.add_argument('--anneal', type=int, default=0, choices=[0, 1], help='simulated annealing')
    parser.add_argument('--finetune', type=int, default=0, choices=[0, 1])
    args = parser.parse_args()

    args.valid_batch_size = args.train_batch_size
    if not args.video:
        args.vasm = 0
        args.fea_path = None
    exp_set = args.exp_set
    args.exp = args.model + exp_set
    args.log_path = 'log/' + args.exp + '/'
    args.tb_path = 'tb_logs/' + args.exp + '/'

    if args.device == 'cuda':
        args.gpuid = 'cuda:' + args.gpuid

    # select model
    if args.model == 'gpt2':
        args.model_checkpoint = "prev_trained_model/gpt2"
    else:
        raise ValueError('NO MODEL IMPLEMENTED!')

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    if not os.path.exists(args.tb_path):
        os.makedirs(args.tb_path)
    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only, logger.warning => log all processes
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Running process %d", args.local_rank)  # This is a logger.warning: it will be printed by all distributed processes
    logger.info("Arguments: %s", pformat(args))

    # Initialize distributed training if needed
    args.distributed = (args.local_rank != -1)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    logger.info("Prepare tokenizer, pretrained model and optimizer - add special tokens for fine-tuning")
    tokenizer_class = GPT2Tokenizer
    model_class = VideoGPT2LMHeadModel
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    model = model_class.from_pretrained(args.model_checkpoint)
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    args.eos = model.config.decoder_start_token_id

    # Prepare model for FP16 and distributed training if needed (order is important, distributed should be the last)
    if args.fp16:
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    logger.info("Prepare datasets")
    train_loader = get_data_loaders_new(args, tokenizer)
    # load vrd_agent and dataloader
    logger.info("Prepare datasets for video")
    vrd_option = read_options()
    vrd_option.device = args.device
    video_loader = Video_data_loader(vrd_option)
    vrd_option.num_entity = video_loader.num_entity
    vrd_option.num_relation = video_loader.num_relation
    vrd_agent = Agent(vrd_option, video_loader)
    vrd_option.decaying_beta_init = vrd_option.beta
    vrd_agent.to(args.device)
    vrd_baseline = ReactiveBaseline(vrd_option, vrd_option.Lambda)
    vrd_type_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("<vrd>"))[0]
    vrd_id2ent = video_loader.num2entity
    vrd_id2rel = video_loader.num2relation
    vrd_option.log_path = 'vrd_log/' + args.exp + '/'
    
    # load text_agent and dataloader
    logger.info("Prepare datasets for text")
    text_option = read_text_options()
    text_option.device = args.device
    if args.drop_rate:
        text_loader = Text_data_loader(text_option, args.context_traj_path)
    else:
        text_loader = Text_data_loader(text_option, args.caption_traj_path)
    text_option.num_entity = text_loader.num_entity
    text_option.num_relation = text_loader.num_relation
    text_agent = Agent(text_option, text_loader)
    text_agent.to(args.device)
    text_baseline = ReactiveBaseline(text_option, text_option.Lambda)
    text_type_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("<trd>"))[0]
    text_id2ent = text_loader.num2entity
    text_id2rel = text_loader.num2relation
    text_option.log_path = 'text_log/' + args.exp + '/'

    # optimizer
    rl_optimizer = torch.optim.Adam(list(vrd_agent.parameters()) + list(text_agent.parameters()), lr=text_option.learning_rate)

    # communicate setting
    assert vrd_option.train_rollouts == text_option.train_rollouts
    assert vrd_option.state_embed_size == text_option.state_embed_size
    assert vrd_option.max_step_length == text_option.max_step_length
    import json
    with open('data/mm-graph/semantic_graph/avsd_json/train_subid.json') as jh:
        ref_dct = json.load(jh)
    

    def update(engine, batch):
        
        cap_lst, cap_type_lst, his_lst, his_type_lst, query_lst, query_type_lst, ans_lst, ans_type_lst, \
        i3d, vid_lst, text_start_lsts, vrd_start_lsts, ref_lsts, subid_lst = batch

        i3d = i3d.to(args.device)

        # input for training
        input_id_lst = []
        input_type_lst = []
        label_lst = []
        label_type_lst = []

        # optimize graph
        vrd_all_loss = []
        vrd_all_logits = []
        vrd_reward = []
        vrd_all_traj = []

        text_all_loss = []
        text_all_logits = []
        text_reward = []
        text_all_traj = []
        
        ref_str_lst = []
        ref_trip_lst = []

        
        vrd_agent_param_lst = []
        text_agent_param_lst = []
        optimizer_param_lst = []
        rr_reward_lst = []
        vrd_rr_traj_lsts = []
        text_rr_traj_lsts = []
        vrd_rr_baseline_lsts = []
        text_rr_baseline_lsts = []
        ori_vrd_agent_param = copy.deepcopy(vrd_agent.state_dict())
        ori_text_agent_param = copy.deepcopy(text_agent.state_dict())
        ori_optimizer_param = copy.deepcopy(rl_optimizer.state_dict())
        
        itr_lst = [0, 1, 2]
        for rr_itr in itr_lst:
            vrd_agent_param, text_agent_param, optimizer_param, eval_reward, vrd_traj_rr, text_traj_rr, vrd_baseline_rr, text_baseline_rr = RR(vrd_agent, text_agent, rl_optimizer, tokenizer, vrd_baseline, text_baseline, 
                vid_lst, subid_lst, vrd_start_lsts, text_start_lsts, ref_dct, ref_lsts, video_loader, text_loader, Knowledge_graph, \
                    vrd_id2rel, text_id2rel, vrd_id2ent, text_id2ent, vrd_option, text_option, args, rr_itr)
            vrd_agent_param_lst.append(vrd_agent_param)
            text_agent_param_lst.append(text_agent_param)
            optimizer_param_lst.append(optimizer_param)
            rr_reward_lst.append(sum(eval_reward))
            vrd_rr_traj_lsts.append(vrd_traj_rr)
            text_rr_traj_lsts.append(text_traj_rr)
            vrd_rr_baseline_lsts.append(vrd_baseline_rr)
            text_rr_baseline_lsts.append(text_baseline_rr)
            vrd_agent.load_state_dict(ori_vrd_agent_param)
            text_agent.load_state_dict(ori_text_agent_param)
            rl_optimizer.load_state_dict(ori_optimizer_param)
        best_index = rr_reward_lst.index(max(rr_reward_lst))
        vrd_agent.load_state_dict(vrd_agent_param_lst[best_index])
        text_agent.load_state_dict(text_agent_param_lst[best_index])
        rl_optimizer.load_state_dict(optimizer_param_lst[best_index])
        vrd_baseline.update(vrd_rr_baseline_lsts[best_index])
        text_baseline.update(text_rr_baseline_lsts[best_index])
        vrd_rr_traj_lsts = vrd_rr_traj_lsts[best_index]
        text_rr_traj_lsts = text_rr_traj_lsts[best_index]
        # # debug
        # t1 = rl_optimizer.state_dict()

        # finetune
        if args.finetune:
            for i in range(len(vid_lst)):
                vid = vid_lst[i]
                subid = subid_lst[i]
                # ref = ref_lsts[i] # string 
                ori_bsz = min(len(vrd_start_lsts[i][:args.top_n]), len(text_start_lsts[i][:args.top_n]))
                bsz = ori_bsz * vrd_option.train_rollouts
                ref_str_lst.append(ref_dct[subid])

                # vrd_agent init
                # 1. init Graph
                vrd_trip_data = video_loader.get_graph_data(vid)
                vrd_train_graph = Knowledge_graph(vrd_option, video_loader, vrd_trip_data)
                vrd_agent.set_graph(vrd_train_graph)
                # 2. init start entities
                vrd_start_ent_str = [x[0] for x in vrd_start_lsts[i][:ori_bsz]] # [ent1, ent2, ...]
                vrd_start_rel_str = [x[1] for x in vrd_start_lsts[i][:ori_bsz]] # [rel1, rel2, ...]
                vrd_start_entity_id = [video_loader.entity2num[x] for x in vrd_start_ent_str]
                vrd_start_relation_id = [video_loader.relation2num[x] for x in vrd_start_rel_str]
                vrd_start_entity_id = np.repeat(vrd_start_entity_id, vrd_option.train_rollouts)
                vrd_start_relation_id = np.repeat(vrd_start_relation_id, vrd_option.train_rollouts)
                vrd_start_entity = torch.tensor(vrd_start_entity_id).to(args.device)
                vrd_relation = torch.tensor(vrd_start_relation_id).to(args.device)
                vrd_prev_relation = vrd_agent.get_dummy_start_relation(bsz).to(args.device)
                vrd_current_entity = vrd_start_entity
                vrd_traj = []
                for x in vrd_start_ent_str:
                    for j in range(vrd_option.train_rollouts):
                        vrd_traj.append([x])
                vrd_state = torch.zeros(1, 2, bsz, vrd_agent.m * vrd_option.state_embed_size).to(args.device)
                vrd_log_probs = np.zeros((bsz, )) * 1.0

                # text_agent init
                # 1. init graph
                text_trip_data = text_loader.get_graph_data(subid)
                text_train_graph = Knowledge_graph(text_option, text_loader, text_trip_data)
                text_agent.set_graph(text_train_graph)
                # 2. init start entities
                text_start_ent_str = [x[0] for x in text_start_lsts[i][:ori_bsz]] # [ent1, ent2, ...]
                text_start_rel_str = [x[1] for x in text_start_lsts[i][:ori_bsz]] # [rel1, rel2, ...]
                text_start_entity_id = [text_loader.entity2num[x] for x in text_start_ent_str]
                text_start_relation_id = [text_loader.relation2num[x] for x in text_start_rel_str]
                text_start_entity_id = np.repeat(text_start_entity_id, text_option.train_rollouts)
                text_start_relation_id = np.repeat(text_start_relation_id, text_option.train_rollouts)
                text_start_entity = torch.tensor(text_start_entity_id).to(args.device)
                text_relation = torch.tensor(text_start_relation_id).to(args.device)
                text_prev_relation = text_agent.get_dummy_start_relation(bsz).to(args.device)
                text_current_entity = text_start_entity
                text_traj = []
                for x in text_start_ent_str:
                    for j in range(vrd_option.train_rollouts):
                        text_traj.append([x])
                text_state = torch.zeros(1, 2, bsz, text_agent.m * text_option.state_embed_size).to(args.device)
                text_log_probs = np.zeros((bsz, )) * 1.0
                
                range_arr = torch.arange(bsz).to(args.device)
            
                # start grounding on graph 
                for step in range(vrd_option.max_step_length):
                    vrd_loss, vrd_state, vrd_logits, vrd_action_idx, vrd_next_entity, vrd_chosen_relation= \
                        vrd_agent.step(vrd_state, vrd_prev_relation, vrd_current_entity, vrd_relation, range_arr, text_state)
                    # loss [B] logits [B, max_out] action_id [B]
                    text_loss, text_state, text_logits, text_action_idx, text_next_entity, text_chosen_relation= \
                        text_agent.step(text_state, text_prev_relation, text_current_entity, text_relation, range_arr, vrd_state)
                    # loss [B] logits [B, max_out] action_id [B]

                    vrd_log_probs += vrd_logits.clone().detach().cpu().numpy()[np.arange(vrd_log_probs.shape[0]), vrd_action_idx.cpu().numpy()]
                    text_log_probs += text_logits.clone().detach().cpu().numpy()[np.arange(text_log_probs.shape[0]), text_action_idx.cpu().numpy()] # [B*num_rollouts]

                    if i == 0:
                        vrd_all_loss.append(vrd_loss) # [B, 1]
                        vrd_all_logits.append(vrd_logits) # [B, max_out]
                        text_all_loss.append(text_loss) # [B, 1]
                        text_all_logits.append(text_logits) # [B, max_out]
                    else:
                        vrd_all_loss[step] = torch.cat([vrd_all_loss[step], vrd_loss], dim=0)
                        vrd_all_logits[step] = torch.cat([vrd_all_logits[step], vrd_logits], dim=0)
                        text_all_loss[step] = torch.cat([text_all_loss[step], text_loss], dim=0)
                        text_all_logits[step] = torch.cat([text_all_logits[step], text_logits], dim=0)
                    

                    # all_action_id.append(action_id)
                    vrd_prev_relation = vrd_chosen_relation
                    vrd_current_entity = vrd_next_entity
                    for j in range(bsz):
                        vrd_traj[j].append(vrd_id2rel[(int)(vrd_chosen_relation[j])])
                        vrd_traj[j].append(vrd_id2ent[(int)(vrd_next_entity[j])])    

                    # all_action_id.append(action_id)
                    text_prev_relation = text_chosen_relation
                    text_current_entity = text_next_entity
                    for j in range(bsz):
                        text_traj[j].append(text_id2rel[(int)(text_chosen_relation[j])])
                        text_traj[j].append(text_id2ent[(int)(text_next_entity[j])])   

                # save all traj
                text_all_traj += text_traj
                vrd_all_traj += vrd_traj
                
                # select traj by log prob.
                vrd_log_probs = np.reshape(vrd_log_probs, (ori_bsz, vrd_option.train_rollouts))
                sorted_vrd_indx = np.argsort(-vrd_log_probs) # [ori_B, rollouts]
                vrd_traj_lst = []
                for j in range(ori_bsz):
                    vrd_traj_lst += vrd_traj[j*vrd_option.train_rollouts + sorted_vrd_indx[j][0]]
                if vrd_traj_lst == []:
                    vrd_traj_lst.append('<pad>')
                
                text_log_probs = np.reshape(text_log_probs, (ori_bsz, text_option.train_rollouts))
                sorted_text_indx = np.argsort(-text_log_probs) # [ori_B, rollouts]
                text_traj_lst = []
                for j in range(ori_bsz):
                    text_traj_lst += text_traj[j*text_option.train_rollouts + sorted_text_indx[j][0]]
                if text_traj_lst == []:
                    text_traj_lst.append('<pad>')
                

                # cosine anneal
                if ref_lsts[i] == []:
                    ref = ['<pad>', '<pad>', '<pad>']
                else:
                    ref = ref_lsts[i][np.random.choice(len(ref_lsts[i]), 1)[0]]
                # ref_lst.append(ref)
                ref_trip_lst.append(ref)
                if args.anneal:
                    text_cand_traj = [text_traj_lst, ref]
                    video_cand_traj = [vrd_traj_lst, ref]
                    x = engine.state.iteration
                    xs = engine.state.epoch_length * 0.0
                    xe = engine.state.epoch_length * 0.1
                    if x <= xs:
                        prob = [0, 1]
                    elif x > xs and x < xe:
                        prob_ref = 1/2 * (1 + math.cos(math.pi * (x-xs)/(xe-xs)))
                        prob = [1-prob_ref, prob_ref]
                    else:
                        prob = [1, 0]
                    traj_index = np.random.choice([0, 1], 1, p=prob)
                    text_traj_lst = text_cand_traj[traj_index[0]]
                    vrd_traj_lst = video_cand_traj[traj_index[0]]

                vrd_traj_id = [vrd_type_id] + list(chain(*[tokenizer.convert_tokens_to_ids(tokenizer.tokenize(d)) for d in vrd_traj_lst]))
                text_traj_id = [text_type_id] + list(chain(*[tokenizer.convert_tokens_to_ids(tokenizer.tokenize(d)) for d in text_traj_lst]))

                comm_traj_token_type = [vrd_type_id] * len(vrd_traj_id) + [text_type_id] * len(text_traj_id)
                comm_traj_id = vrd_traj_id + text_traj_id        
                comm_traj_id = torch.tensor(comm_traj_id).long()
                comm_traj_token_type = torch.tensor(comm_traj_token_type).long()

                if args.drop_rate == 0:
                    input_id_lst.append(torch.cat([cap_lst[i], comm_traj_id, his_lst[i], query_lst[i], ans_lst[i]]))
                    input_type_lst.append(torch.cat([cap_type_lst[i], comm_traj_token_type, his_type_lst[i], query_type_lst[i], ans_type_lst[i]]))
                    label_lst.append(torch.cat([torch.ones(cap_lst[i].size())*(-1), torch.ones(comm_traj_id.size())*(-1), torch.ones(his_lst[i].size())*(-1), \
                        torch.ones(query_lst[i].size())*(-1), ans_lst[i]]))
                    label_type_lst.append(torch.cat([torch.ones(cap_type_lst[i].size())*(-1), torch.ones(comm_traj_token_type.size())*(-1), torch.ones(his_type_lst[i].size())*(-1), \
                        torch.ones(query_type_lst[i].size())*(-1), ans_type_lst[i]]))
                elif args.drop_rate == 1:
                    input_id_lst.append(torch.cat([comm_traj_id, his_lst[i], query_lst[i], ans_lst[i]]))
                    input_type_lst.append(torch.cat([comm_traj_token_type, his_type_lst[i], query_type_lst[i], ans_type_lst[i]]))
                    label_lst.append(torch.cat([torch.ones(comm_traj_id.size())*(-1), torch.ones(his_lst[i].size())*(-1), torch.ones(query_lst[i].size())*(-1), ans_lst[i]]))
                    label_type_lst.append(torch.cat([torch.ones(comm_traj_token_type.size())*(-1), torch.ones(his_type_lst[i].size())*(-1), torch.ones(query_type_lst[i].size())*(-1), ans_type_lst[i]]))

            # optimize rl agent
            if args.reward == 'rouge':
                for j, traj in enumerate(vrd_all_traj):
                    # # sparse
                    # if j < len(vrd_all_traj)-bsz:
                    #     vrd_reward.append(0)
                    # else:
                    #     # # gold trip
                    #     # vrd_reward.append(rouge_n(tokenizer.tokenize(' '.join(ref_lst[j//bsz]).replace('<pad>', '')), tokenizer.tokenize(' '.join(traj).replace('<pad>', ''))))
                    #     # gold string
                    #     vrd_reward.append(rouge_n(tokenizer.tokenize(ref_lst[j//bsz]), tokenizer.tokenize(' '.join(traj).replace('<pad>', ''))))
                    
                    # # randomize
                    # if ref_trip_lst[-1] == ['<pad>'] * len(ref_trip_lst[j//bsz]) or traj == ['<pad>'] * len(traj):
                    # # if ref_str_lst[j//bsz] == '' or traj == ['<pad>'] * len(traj):
                    #     vrd_reward.append(0)
                    # else:
                    #     # gold trip
                    #     vrd_reward.append(rouge_n(tokenizer.tokenize(' '.join(ref_trip_lst[-1]).replace('<pad>', '')), tokenizer.tokenize(' '.join(traj).replace('<pad>', ''))))
                    #     # # gold string
                    #     # vrd_reward.append(rouge_n(tokenizer.tokenize(ref_str_lst[-1]), tokenizer.tokenize(' '.join(traj).replace('<pad>', ''))))

                    # normarl
                    if ref_trip_lst[j//bsz] == ['<pad>'] * len(ref_trip_lst[j//bsz]) or traj == ['<pad>'] * len(traj):
                        vrd_reward.append(0)
                    else:
                        # vrd_reward.append(bleu_corpus(' '.join(traj), ref))
                        # gold trip
                        vrd_reward.append(rouge_n(tokenizer.tokenize(' '.join(ref_trip_lst[j//bsz]).replace('<pad>', '')), tokenizer.tokenize(' '.join(traj).replace('<pad>', ''))))
                        # # gold string
                        # vrd_reward.append(rouge_n(tokenizer.tokenize(ref_str_lst[j//bsz]), tokenizer.tokenize(' '.join(traj).replace('<pad>', ''))))

                for j, traj in enumerate(text_all_traj):
                    # # sparse:
                    # if j < len(text_all_traj)-bsz:
                    #     text_reward.append(0)
                    # else:
                    #     # # gold trip
                    #     # text_reward.append(rouge_n(tokenizer.tokenize(' '.join(ref_lst[j//bsz]).replace('<pad>', '')), tokenizer.tokenize(' '.join(traj).replace('<pad>', ''))))
                    #     # gold string
                    #     text_reward.append(rouge_n(tokenizer.tokenize(ref_lst[j//bsz]), tokenizer.tokenize(' '.join(traj).replace('<pad>', ''))))
                    
                    # # randomize:
                    # if ref_trip_lst[-1] == ['<pad>'] * len(ref_trip_lst[j//bsz]) or traj == ['<pad>'] * len(traj):
                    # # if ref_str_lst[j//bsz] == '' or traj == ['<pad>'] * len(traj):
                    #     text_reward.append(0)
                    # else:
                    #     # # gold trip
                    #     text_reward.append(rouge_n(tokenizer.tokenize(' '.join(ref_trip_lst[-1]).replace('<pad>', '')), tokenizer.tokenize(' '.join(traj).replace('<pad>', ''))))
                    #     # # gold string
                    #     # text_reward.append(rouge_n(tokenizer.tokenize(ref_str_lst[-1]), tokenizer.tokenize(' '.join(traj).replace('<pad>', ''))))

                    # normal
                    if ref_trip_lst[j//bsz] == ['<pad>'] * len(ref_trip_lst[j//bsz]) or traj == ['<pad>'] * len(traj):
                        text_reward.append(0)
                    else:
                        # text_reward.append(bleu_corpus(' '.join(traj), ref))
                        # gold trip
                        text_reward.append(rouge_n(tokenizer.tokenize(' '.join(ref_trip_lst[j//bsz]).replace('<pad>', '')), tokenizer.tokenize(' '.join(traj).replace('<pad>', ''))))
                        # # gold string
                        # text_reward.append(rouge_n(tokenizer.tokenize(ref_trip_lst[j//bsz]), tokenizer.tokenize(' '.join(traj).replace('<pad>', ''))))
            else:
                raise ValueError('INVALID REWARD TYPE!')

            # select best traj for constructing input of generator
            vrd_reward = torch.tensor(vrd_reward).to(args.device)
            text_reward = torch.tensor(text_reward).to(args.device)
            vrd_cum_discounted_reward = calc_cum_discounted_reward_credit(vrd_option, vrd_reward, text_reward)
            vrd_reinforce_loss = calc_reinforce_loss(vrd_baseline, vrd_all_loss, vrd_all_logits, vrd_cum_discounted_reward, vrd_option.beta)
            vrd_baseline.update(torch.mean(vrd_cum_discounted_reward)) # baseline 使用线性插值
            text_cum_discounted_reward = calc_cum_discounted_reward_credit(text_option, text_reward, vrd_reward)
            text_reinforce_loss = calc_reinforce_loss(text_baseline, text_all_loss, text_all_logits, text_cum_discounted_reward, text_option.beta)
            text_baseline.update(torch.mean(text_cum_discounted_reward))
        
            rl_optimizer.zero_grad()
            reinforce_loss = vrd_reinforce_loss + text_reinforce_loss
            reinforce_loss.backward()
            torch.nn.utils.clip_grad_norm_(vrd_agent.parameters(), max_norm=vrd_option.grad_clip_norm, norm_type=2)
            torch.nn.utils.clip_grad_norm_(text_agent.parameters(), max_norm=text_option.grad_clip_norm, norm_type=2)
            rl_optimizer.step()
        else:
            for i, vid in enumerate(vid_lst):
                vrd_traj_lst = vrd_rr_traj_lsts[i]
                text_traj_lst = text_rr_traj_lsts[i]

                # cosine anneal
                if ref_lsts[i] == []:
                    ref = ['<pad>', '<pad>', '<pad>']
                else:
                    ref = ref_lsts[i][np.random.choice(len(ref_lsts[i]), 1)[0]]
                # ref_lst.append(ref)
                ref_trip_lst.append(ref)
                if args.anneal and engine.state.epoch <= 1:
                    text_cand_traj = [text_traj_lst, ref]
                    video_cand_traj = [vrd_traj_lst, ref]
                    x = engine.state.iteration
                    xs = engine.state.epoch_length * 0.0
                    xe = engine.state.epoch_length * 0.1
                    if x <= xs:
                        prob = [0, 1]
                    elif x > xs and x < xe:
                        prob_ref = 1/2 * (1 + math.cos(math.pi * (x-xs)/(xe-xs)))
                        prob = [1-prob_ref, prob_ref]
                    else:
                        prob = [1, 0]
                    traj_index = np.random.choice([0, 1], 1, p=prob)
                    text_traj_lst = text_cand_traj[traj_index[0]]
                    vrd_traj_lst = video_cand_traj[traj_index[0]]

                vrd_traj_id = [vrd_type_id] + list(chain(*[tokenizer.convert_tokens_to_ids(tokenizer.tokenize(d)) for d in vrd_traj_lst]))
                text_traj_id = [text_type_id] + list(chain(*[tokenizer.convert_tokens_to_ids(tokenizer.tokenize(d)) for d in text_traj_lst]))

                comm_traj_token_type = [vrd_type_id] * len(vrd_traj_id) + [text_type_id] * len(text_traj_id)
                comm_traj_id = vrd_traj_id + text_traj_id        
                comm_traj_id = torch.tensor(comm_traj_id).long()
                comm_traj_token_type = torch.tensor(comm_traj_token_type).long()

                if args.drop_rate == 0:
                    input_id_lst.append(torch.cat([cap_lst[i], comm_traj_id, his_lst[i], query_lst[i], ans_lst[i]]))
                    input_type_lst.append(torch.cat([cap_type_lst[i], comm_traj_token_type, his_type_lst[i], query_type_lst[i], ans_type_lst[i]]))
                    label_lst.append(torch.cat([torch.ones(cap_lst[i].size())*(-1), torch.ones(comm_traj_id.size())*(-1), torch.ones(his_lst[i].size())*(-1), \
                        torch.ones(query_lst[i].size())*(-1), ans_lst[i]]))
                    label_type_lst.append(torch.cat([torch.ones(cap_type_lst[i].size())*(-1), torch.ones(comm_traj_token_type.size())*(-1), torch.ones(his_type_lst[i].size())*(-1), \
                        torch.ones(query_type_lst[i].size())*(-1), ans_type_lst[i]]))
                elif args.drop_rate == 1:
                    input_id_lst.append(torch.cat([comm_traj_id, his_lst[i], query_lst[i], ans_lst[i]]))
                    input_type_lst.append(torch.cat([comm_traj_token_type, his_type_lst[i], query_type_lst[i], ans_type_lst[i]]))
                    label_lst.append(torch.cat([torch.ones(comm_traj_id.size())*(-1), torch.ones(his_lst[i].size())*(-1), torch.ones(query_lst[i].size())*(-1), ans_lst[i]]))
                    label_type_lst.append(torch.cat([torch.ones(comm_traj_token_type.size())*(-1), torch.ones(his_type_lst[i].size())*(-1), torch.ones(query_type_lst[i].size())*(-1), ans_type_lst[i]]))

            

        # padding input ids
        input_ids = padding(input_id_lst, tokenizer.pad_token_id).to(args.device)
        input_type_ids = padding(input_type_lst, tokenizer.pad_token_id).to(args.device)
        label_ids = padding(label_lst, -1).to(args.device)
        label_type_ids = padding(label_type_lst, -1).to(args.device)
        input_mask = input_ids != tokenizer.pad_token_id
        # video
        i3d_mask = torch.sum(i3d != 1, dim=2) != 0
        input_mask = torch.cat([i3d_mask, input_mask], dim=1)
        gpt_video_mask = torch.cat([torch.zeros((i3d.size(0), i3d.size(1))).long().to(args.device), torch.ones(label_ids.size()).long().to(args.device)], 1) # video_mask keep language
        reply_mask = torch.zeros(gpt_video_mask.size()).long().to(args.device) # reply_mask mask none
        i3d_labels = torch.ones((i3d.size(0), i3d.size(1))).long().to(args.device) * -1 # 
        label_ids = torch.cat([i3d_labels, label_ids], dim=1)
        input_type_ids = torch.cat([torch.ones((i3d.size(0), i3d.size(1))).long().to(args.device) * tokenizer.convert_tokens_to_ids('<video>'), input_type_ids], dim=1)
        label_type_ids = torch.cat([torch.ones((i3d.size(0), i3d.size(1))).long().to(args.device) * tokenizer.convert_tokens_to_ids('<video>'), label_type_ids], dim=1)
        # input embedding
        input_embs = model.transformer.wte(input_ids)
        video_embs = model.video_ff(i3d)
        input_embs = torch.cat([video_embs, input_embs], dim=1)


        model.train()
        video_loss = model(input_embs,token_type_ids=input_type_ids, labels=(label_ids, i3d), attention_mask=[gpt_video_mask, input_mask], mode="video")[0]
        reply_loss = model(input_embs,token_type_ids=input_type_ids, labels=(label_ids, i3d), attention_mask=[reply_mask, input_mask], mode="reply")[0]
        loss = (video_loss + reply_loss) / args.gradient_accumulation_steps
            
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()
    
    
    trainer = Engine(update)

    # # warmup
    # trainer.add_event_handler(Events.STARTED, lambda _:agent.run(train_loader))

    # Linearly decrease the learning rate from lr to zero
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Prepare metrics - note how we compute distributed metrics 
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")

    # On the main process: add progress bar, tensorboard, checkpoints and save model, configuration and tokenizer before we start to train
    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["loss"])
        

        tb_logger = TensorboardLogger(log_dir=args.tb_path)
        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]), event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
        

        checkpoint_handler = ModelCheckpoint(args.log_path, 'checkpoint', n_saved=args.n_epochs ,require_empty=False)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})  # "getattr" take care of distributed encapsulation
        
        torch.save(args, args.log_path + 'model_training_args.bin')
        getattr(model, 'module', model).config.to_json_file(os.path.join(args.log_path, CONFIG_NAME))
        tokenizer.save_vocabulary(args.log_path)

        vrd_checkpoint_handler = ModelCheckpoint('communicate_log/vrd/'+args.exp+'/', 'checkpoint', n_saved=args.n_epochs ,require_empty=False)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, vrd_checkpoint_handler, {'mymodel': getattr(vrd_agent, 'module', vrd_agent)})  # "getattr" take care of distributed encapsulation
        
        text_checkpoint_handler = ModelCheckpoint('communicate_log/text/'+args.exp+'/', 'checkpoint', n_saved=args.n_epochs ,require_empty=False)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, text_checkpoint_handler, {'mymodel': getattr(text_agent, 'module', text_agent)})  # "getattr" take care of distributed encapsulation

    

    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        tb_logger.close()

if __name__ == "__main__":
    train()
