import os
import json
import logging
import time
import copy
from argparse import ArgumentParser
from itertools import chain
import copy
import pickle as pkl
import numpy as np
import sys
sys.path.append('nltk_data')

import torch
import torch.nn.functional as F

from transformers import GPT2Tokenizer, GPT2Config, BartTokenizer, BartConfig
from VideoGPT2 import VideoGPT2LMHeadModel
from marl_vt_dataset import get_dataset, build_input_from_segments

from du_rl.Graph import Knowledge_graph
from du_rl.Baseline import ReactiveBaseline
from du_rl.vcls_data import Video_data_loader
from du_rl.text_data import Text_data_loader
from du_rl.Agent import Agent
from du_rl.video_options import read_options
from du_rl.text_options import read_text_options


SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>","<cap>", "<video>", "<pad>", "<vrd>", "<trd>", "<start>", "<equal>"]
SPECIAL_TOKENS_DICT = {'bos_token': "<bos>", 'eos_token': "<eos>", 'additional_special_tokens': ["<speaker1>", "<speaker2>", "<video>", "<cap>", "<vrd>", "<trd>", "<start>","<equal>"], 'pad_token': "<pad>"}
MODEL_INPUTS = ["input_ids", "token_type_ids","lm_labels"]
PADDED_INPUTS = ["input_ids", "token_type_ids","lm_labels"]

def top_k(scores, k, max_out):
    scores = scores.reshape(-1, k * max_out)  # [B, (k*max_num_actions)]
    idx = np.argsort(scores, axis=1)
    idx = idx[:, -k:]  # take the last k highest indices # [B , k]
    return idx.reshape((-1))

def beam_search(caption, history, tokenizer, model, args, traj_id, traj_token_type_id, current_output=None, video=None):
    if current_output is None:
        current_output = []
    hyplist = [([], 0., current_output)]
    best_state = None
    comp_hyplist = []

    for i in range(args.max_length):
        new_hyplist = []
        argmin = 0
        for out, lp, st in hyplist:
            
            if args.drop_rate == 1: 
                instance, _ = build_input_from_segments(caption, history, st, tokenizer, with_eos=False, drop_caption=True, model=args.model)
            elif args.drop_rate == 0: # train/validate: drop_rate = 0
                instance, _ = build_input_from_segments(caption, history, st, tokenizer, with_eos=False, drop_caption=False, model=args.model)
            else:
                raise ValueError('NO IMPLEMENTED DROP_RATE')

            his_ids = torch.tensor(instance["his_ids"], device=args.device).long()
            his_type_ids = torch.tensor(instance["his_type_ids"], device=args.device).long()
            query_ids = torch.tensor(instance["query_ids"], device=args.device).long()
            query_type_ids = torch.tensor(instance["query_type_ids"], device=args.device).long()
            answer_ids = torch.tensor(instance["answer_ids"], device=args.device).long()
            answer_type_ids = torch.tensor(instance["answer_type_ids"], device=args.device).long()

            if args.drop_rate == 0:
                cap_ids = torch.tensor(instance["cap_ids"], device=args.device).long()
                cap_type_ids = torch.tensor(instance["cap_type_ids"], device=args.device).long()
                input_ids = torch.cat([cap_ids, traj_id, his_ids, query_ids, answer_ids], dim=-1).unsqueeze(0)
                input_type_ids = torch.cat([cap_type_ids, traj_token_type_id, his_type_ids, query_type_ids, answer_type_ids], dim=-1).unsqueeze(0)
            elif args.drop_rate == 1:
                input_ids = torch.cat([traj_id, his_ids, query_ids, answer_ids], dim=-1).unsqueeze(0)
                input_type_ids = torch.cat([traj_token_type_id, his_type_ids, query_type_ids, answer_type_ids], dim=-1).unsqueeze(0)
            
            input_embs = model.transformer.wte(input_ids)
            # video = None
            if video is not None:
                input_embs = torch.cat([model.video_ff(video), input_embs], dim=1)
                token_type_ids = torch.cat([torch.ones((1, video.size(1))).long().to(args.device) * tokenizer.convert_tokens_to_ids('<video>'), input_type_ids], dim=1)
                i3d = video
            input_mask = input_ids != tokenizer.pad_token_id
            i3d_mask = torch.sum(i3d != 1, dim=2) != 0
            input_mask = torch.cat([i3d_mask, input_mask], dim=1).long().to(args.device)
            reply_mask = torch.zeros(input_mask.size()).long().to(args.device)
            logits = model(input_embs, token_type_ids=token_type_ids, attention_mask=[reply_mask, input_mask])
 
            # if "gpt2" == args.model:
            logits = logits[0] # (bz, seq_len, vocab_size)
            logp = F.log_softmax(logits, dim=-1)[:, -1, :] # (bz, 1, vocab_size)
            lp_vec = logp.cpu().data.numpy() + lp
            lp_vec = np.squeeze(lp_vec) # (vocab_size)
            if i >= args.min_length:
                new_lp = lp_vec[tokenizer.eos_token_id] + args.penalty * (len(out) + 1) # 结束的概率
                comp_hyplist.append((out, new_lp))
                if best_state is None or best_state < new_lp:
                    best_state = new_lp
            count = 1
            for o in np.argsort(lp_vec)[::-1]:
                if o == tokenizer.unk_token_id or o == tokenizer.eos_token_id:
                    continue
                new_lp = lp_vec[o]
                if len(new_hyplist) == args.beam_size:
                    if new_hyplist[argmin][1] < new_lp:
                        new_st = copy.deepcopy(st)
                        new_st.append(int(o))
                        new_hyplist[argmin] = (out + [o], new_lp, new_st) # 结束概率最小的 （out, lp, st）
                        argmin = min(enumerate(new_hyplist), key=lambda h: h[1][1])[0]
                    else:
                        break
                else:
                    new_st = copy.deepcopy(st)
                    new_st.append(int(o))
                    new_hyplist.append((out + [o], new_lp, new_st))
                    if len(new_hyplist) == args.beam_size:
                        argmin = min(enumerate(new_hyplist), key=lambda h: h[1][1])[0] # 结束概率最小的位置
                count += 1
        hyplist = new_hyplist 
    if len(comp_hyplist) > 0:
        maxhyps = sorted(comp_hyplist, key=lambda h: -h[1])[:1] # sorted 默认升序 选择结束概率最大的out
        return maxhyps
    else:
        return [([], 0)]

##################################
# main
if __name__ =="__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2", help="Model type (gpt or gpt2)")
    parser.add_argument("--model_checkpoint", type=str, default="log/gpt2/", help="Path, url or short name of the model")
    parser.add_argument("--max_history", type=int, default=3, help="Number of previous utterances to keep in history")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--beam_search", default=True, help="Set to use beam search instead of sampling")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam size")
    parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--penalty", type=float, default=0.3, help="elngth penalty")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    parser.add_argument("--task", type=str, default='7')
    parser.add_argument("--test_set", type=str, default="data/avsd/test_set4DSTC8-AVSD.json")
    parser.add_argument("--lbl_test_set", type=str, default="data/avsd/dstc7avsd_eval/data/lbl_undisclosedonly_test_set4DSTC7-AVSD.json")
    parser.add_argument("--context_traj_path", type=str, default="data/mm-graph/semantic_graph/avsd_context/con_trip.json")
    parser.add_argument("--caption_traj_path", type=str, default="data/mm-graph/semantic_graph/avsd_cap_con/cap_con_trip.json")
    parser.add_argument("--video_traj_path", type=str, default="data/mm-graph/semantic_graph/avsd_video/cls_trip.json")
    parser.add_argument("--output", type=str, default="result.json")
    parser.add_argument("--ckptid", type=str, help='ckpt selected for test')
    parser.add_argument("--gpuid", type=str, default='0', help='gpu id')
    parser.add_argument("--log", type=bool, default=False, help='if logging info')
    parser.add_argument('--exp_set', type=str, default='test')
    parser.add_argument('--log_set', type=str, default='', help='log file name')
    parser.add_argument('--video', type=int, default=1)
    parser.add_argument('--test', type=int, default=0)
    parser.add_argument('--drop_rate', type=int, default=1)
    parser.add_argument('--top_n', type=int, default=1)
    parser.add_argument('--step', type=int, default=2)
    parser.add_argument('--beam', type=int, default=0, choices=[0, 1], help='beam search in semantic graph')
    parser.add_argument('--visualize', type=int, default=1, choices=[0, 1])
    args = parser.parse_args()
    
    # test by hand params:
    # args.model = 'bart'
    # args.ckptid = '57444'
    # args.device = 'cpu'
    # args.log_set = '_marl_wocap_anneal'
    # args.step = 5
    # args.log_set = '_total_wv_wocap'
    # args.beam_search = False
    # args.penalty = 1

    exp_set = args.exp_set

    if args.task == '8':
        args.test_set = "data/avsd/test_set4DSTC8-AVSD.json"
    elif args.task == '7':
        args.test_set = "data/avsd/test_set4DSTC7-AVSD.json"
    args.model_checkpoint = 'log/' + args.model + args.log_set + '/'
    output_dir = 'results/' + args.model + exp_set
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    args.output = output_dir + '/task_' + args.task + '_result_' + args.ckptid  + '_' + str(args.beam_size) + '_' + str(args.min_length) + '_' + str(args.max_length) + '_' + str(args.penalty) + '.json'
    
    if args.device == 'cuda':
        args.device = 'cuda:' + args.gpuid
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    for arg in vars(args):
        print("{}={}".format(arg, getattr(args, arg)))

    logging.basicConfig(level=logging.INFO,
            format='%(asctime)s %(levelname)s: %(message)s')
 
    logging.info('Loading model params from ' + args.model_checkpoint)
    
    if 'gpt' in args.model:
        tokenizer_class = GPT2Tokenizer
        model_class = VideoGPT2LMHeadModel
        model_config = GPT2Config.from_pretrained(args.model_checkpoint)
    else:
        print('No pre-trained model: {}!'.format(args.model))
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
    model = model_class.from_pretrained(args.model_checkpoint+"checkpoint_mymodel_" + args.ckptid + ".pt", config=model_config)
    model.to(args.device)
    model.eval()

    if args.log:
        logging.info('Loading test data from ' + args.test_set)
    test_data = json.load(open(args.test_set,'r'))
    if args.drop_rate == 1:
        test_dataset_path = 'pkls/marl_con_vrd_test_{}_data.pkl'.format(args.task)
    else:
        test_dataset_path = 'pkls/marl_cap_con_vrd_test_{}_data.pkl'.format(args.task)
    if not os.path.exists(test_dataset_path):
        if args.drop_rate:
            test_dataset = get_dataset(tokenizer, args.test_set, text_traj_path=args.context_traj_path, video_traj_path=args.video_traj_path, undisclosed_only=True, n_history=args.max_history)
        else:
            test_dataset = get_dataset(tokenizer, args.test_set, text_traj_path=args.caption_traj_path, video_traj_path=args.video_traj_path, undisclosed_only=True, n_history=args.max_history)
        with open(test_dataset_path, 'wb') as f:
            pkl.dump(test_dataset, f)
    else:
        with open(test_dataset_path, 'rb') as f:
            test_dataset = pkl.load(f)
    # generate sentences
    if args.log:
        logging.info('-----------------------generate--------------------------')
    start_time = time.time()

    result_dialogs = []
    model.eval()

    # load vrd agent
    vrd_option = read_options()
    vrd_option.device = args.device
    video_loader = Video_data_loader(vrd_option)
    vrd_option.num_entity = video_loader.num_entity
    vrd_option.num_relation = video_loader.num_relation
    vrd_agent = Agent(vrd_option, video_loader)
    vrd_option.model_checkpoint = 'communicate_log/vrd/' + args.model + args.log_set + '/'
    vrd_option.decaying_beta_init = vrd_option.beta
    vrd_agent.load_state_dict(torch.load(vrd_option.model_checkpoint+"checkpoint_mymodel_" + args.ckptid + ".pt", map_location=args.device))
    vrd_agent.to(args.device)
    vrd_baseline = ReactiveBaseline(vrd_option, vrd_option.Lambda)
    vrd_type_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("<vrd>"))[0]
    vrd_id2ent = video_loader.num2entity
    vrd_id2rel = video_loader.num2relation
    vrd_option.max_step_length = args.step

    # load text agent
    text_option = read_text_options()
    text_option.device = args.device
    if args.drop_rate:
        text_loader = Text_data_loader(text_option, args.context_traj_path)
    else:
        text_loader = Text_data_loader(text_option, args.caption_traj_path)
    text_option.num_entity = text_loader.num_entity
    text_option.num_relation = text_loader.num_relation
    text_agent = Agent(text_option, text_loader)
    text_option.model_checkpoint = 'communicate_log/text/' + args.model + args.log_set + '/'
    text_agent.load_state_dict(torch.load(text_option.model_checkpoint+"checkpoint_mymodel_" + args.ckptid + ".pt", map_location=args.device))
    text_agent.to(args.device)
    text_baseline = ReactiveBaseline(text_option, text_option.Lambda)
    text_type_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("<trd>"))[0]
    text_id2ent = text_loader.num2entity
    text_id2rel = text_loader.num2relation
    vrd_option.max_step_length = args.step
    

    # visualize reference
    if args.visualize:
        with open('data/mm-graph/semantic_graph/avsd_json/test_set4DSTC7-AVSD_multiref.json') as jh:
            ref_dct = json.load(jh)
    

    with torch.no_grad():
        qa_id = 0
        for idx, dialog in enumerate(test_data['dialogs']):
            vid = dialog['image_id']
            out_dialog = dialog['dialog'][-1:]
            subid = len(dialog['dialog']) - 1
            pred_dialog = {'image_id': vid,
                           'dialog': copy.deepcopy(out_dialog)}
            result_dialogs.append(pred_dialog)
            
            if args.video:
                vgg = np.load("data/avsd/vggish_testset/"+vid+".npy")
                i3d_flow = np.load("data/avsd/i3d_flow_testset/"+vid+".npy")
                i3d_rgb = np.load("data/avsd/i3d_rgb_testset/"+vid+".npy")

                # sample_step = i3d_flow.shape[0] // vgg.shape[0]
                # if sample_step == 0:
                #     sample_step = 1
                
                sample_step = 1
                sample_i3d_flow = i3d_flow[range(1, i3d_flow.shape[0], sample_step)]
                sample_i3d_rgb = i3d_rgb[range(1, i3d_rgb.shape[0], sample_step)]

                vgg = torch.from_numpy(vgg).float().to(args.device)
                i3d_flow = torch.from_numpy(sample_i3d_flow).float().to(args.device)
                i3d_rgb = torch.from_numpy(sample_i3d_rgb).float().to(args.device)
                min_length = min([i3d_flow.size(0), i3d_rgb.size(0), vgg.size(0)])
                i3d = torch.cat([i3d_flow[:min_length], i3d_rgb[:min_length], vgg[:min_length]], dim=1).unsqueeze(0)
            else:
                i3d = None
            
            for t, qa in enumerate(out_dialog):
                if args.log:
                    logging.info('%d %s_%d' % (qa_id, vid, t))
                    logging.info('QS: ' + qa['question'])
                # prepare input data
                start_time = time.time()
                qa_id += 1

                ori_bsz = min(len(test_dataset[idx]['video_traj'][:args.top_n]),len(test_dataset[idx]['text_traj'][:args.top_n]))
                bsz = ori_bsz * vrd_option.test_rollouts
                # generate vrd triplet
                # 1. init Graph
                vrd_trip_data = video_loader.get_graph_data(vid)
                vrd_test_graph = Knowledge_graph(vrd_option, video_loader, vrd_trip_data)
                vrd_agent.set_graph(vrd_test_graph)
                # 2. init start entities
                vrd_start_trips = test_dataset[idx]["video_traj"]
                vrd_start_ent_str = [x[0] for x in vrd_start_trips[:ori_bsz]]
                vrd_start_rel_str = [x[1] for x in vrd_start_trips[:ori_bsz]]
                vrd_start_entity_id = [video_loader.entity2num[x] for x in vrd_start_ent_str]
                vrd_start_relation_id = [video_loader.relation2num[x] for x in vrd_start_rel_str]
                vrd_start_entity_id = np.repeat(vrd_start_entity_id, vrd_option.test_rollouts)
                vrd_start_relation_id = np.repeat(vrd_start_relation_id, vrd_option.test_rollouts)
                vrd_start_entity = torch.tensor(vrd_start_entity_id).to(args.device)
                vrd_relation = torch.tensor(vrd_start_relation_id).to(args.device)
                vrd_prev_relation = vrd_agent.get_dummy_start_relation(bsz).to(args.device)
                vrd_current_entity = vrd_start_entity
                vrd_log_current_prob = (torch.zeros(vrd_start_entity.size(0))).to(args.device)
                vrd_traj = []
                for x in vrd_start_ent_str:
                    for j in range(vrd_option.test_rollouts):
                        vrd_traj.append([x])
                vrd_state = torch.zeros(1, 2, bsz, vrd_agent.m * vrd_option.state_embed_size).to(args.device)
                vrd_log_probs = np.zeros((bsz, )) * 1.0

                # generate text triplet
                # 1. init graph
                text_trip_data = text_loader.get_graph_data(vid + '_' + str(subid))
                text_test_graph = Knowledge_graph(text_option, text_loader, text_trip_data)
                text_agent.set_graph(text_test_graph)
                # 2. init start entities
                text_start_trips = test_dataset[idx]["text_traj"]
                text_start_ent_str = [x[0] for x in text_start_trips[:ori_bsz]]
                text_start_rel_str = [x[1] for x in text_start_trips[:ori_bsz]]
                text_start_entity_id = [text_loader.entity2num[x] for x in text_start_ent_str]
                text_start_relation_id = [text_loader.relation2num[x] for x in text_start_rel_str]
                text_start_entity_id = np.repeat(text_start_entity_id, text_option.test_rollouts)
                text_start_relation_id = np.repeat(text_start_relation_id, text_option.test_rollouts)
                text_start_entity = torch.tensor(text_start_entity_id).to(args.device)
                text_relation = torch.tensor(text_start_relation_id).to(args.device)
                text_prev_relation = text_agent.get_dummy_start_relation(bsz).to(args.device)
                text_current_entity = text_start_entity
                text_log_current_prob = (torch.zeros(text_start_entity.size(0))).to(args.device)
                text_traj = []
                for x in text_start_ent_str:
                    for j in range(vrd_option.test_rollouts):
                        text_traj.append([x])
                text_state = torch.zeros(1, 2, bsz, text_agent.m * text_option.state_embed_size).to(args.device)
                text_log_probs = np.zeros((bsz, )) * 1.0

                range_arr = torch.arange(bsz).to(args.device)
                vrd_beam_probs = torch.zeros((bsz, 1)).to(args.device)
                text_beam_probs = torch.zeros((bsz, 1)).to(args.device)
                
                for i, step in enumerate(range(vrd_option.max_step_length)):

                    vrd_state, vrd_logits, vrd_action_idx, vrd_chosen_entity, vrd_chosen_relation, vrd_next_entities_id, vrd_next_relations_id = \
                        vrd_agent.test_step(vrd_state, vrd_prev_relation, vrd_current_entity, vrd_relation, range_arr, text_state)
                    # loss [B] logits [B, max_out] action_id [B]
                    text_state, text_logits, text_action_idx, text_chosen_entity, text_chosen_relation, text_next_entities_id, text_next_relations_id = \
                        text_agent.test_step(text_state, text_prev_relation, text_current_entity, text_relation, range_arr, vrd_state)
                    # loss [B] logits [B, max_out] action_id [B]
                    
                    
                    if args.beam:
                        k = vrd_option.test_rollouts
                        vrd_beam_probs = vrd_beam_probs.to(args.device)
                        new_vrd_log_probs = vrd_logits + vrd_beam_probs
                        new_vrd_log_probs = new_vrd_log_probs.cpu()
                        text_beam_probs = text_beam_probs.to(args.device)
                        new_text_log_probs = text_logits + text_beam_probs
                        new_text_log_probs = new_text_log_probs.cpu()
                        if i == 0:
                            vrd_idx = np.argsort(new_vrd_log_probs)
                            vrd_idx = vrd_idx[:, -k:]
                            vrd_range_idx = np.tile([b for b in range(ori_bsz)], ori_bsz)
                            vrd_idx = vrd_idx[np.arange(bsz), vrd_range_idx]
                            text_idx = np.argsort(new_text_log_probs)
                            text_idx = text_idx[:, -k:]
                            text_range_idx = np.tile([b for b in range(ori_bsz)], ori_bsz)
                            text_idx = text_idx[np.arange(bsz), text_range_idx]
                        else:
                            vrd_idx = top_k(new_vrd_log_probs, k, vrd_option.max_out) # [B*k]
                            text_idx = top_k(new_text_log_probs, k, text_option.max_out)
                        vrd_y = vrd_idx // vrd_option.max_out
                        vrd_x = vrd_idx % vrd_option.max_out
                        vrd_y += np.repeat([b*k for b in range(ori_bsz)], k)
                        vrd_current_entity = vrd_current_entity[vrd_y]
                        vrd_next_relations_id =  vrd_next_relations_id[vrd_y, :]
                        vrd_next_entities_id = vrd_next_entities_id[vrd_y, :]
                        vrd_state = vrd_state[:, :, vrd_y, :]
                        vrd_action_idx = vrd_x
                        vrd_chosen_relation = vrd_next_relations_id[np.arange(bsz), vrd_x]
                        vrd_chosen_entities = vrd_next_entities_id[np.arange(bsz), vrd_x]
                        vrd_beam_probs = new_vrd_log_probs[vrd_y, vrd_x]
                        vrd_beam_probs = vrd_beam_probs.reshape((-1, 1))
                        # vrd_traj = vrd_traj[vrd_y]

                        text_y = text_idx // text_option.max_out
                        text_x = text_idx % text_option.max_out
                        text_y += np.repeat([b*k for b in range(ori_bsz)], k)
                        text_current_entity = text_current_entity[text_y]
                        text_next_relations_id =  text_next_relations_id[text_y, :]
                        text_next_entities_id = text_next_entities_id[text_y, :]
                        text_state = text_state[:, :, text_y, :]
                        text_action_idx = text_x
                        text_chosen_relation = text_next_relations_id[np.arange(bsz), text_x]
                        text_chosen_entities = text_next_entities_id[np.arange(bsz), text_x]
                        text_beam_probs = new_text_log_probs[text_y, text_x]
                        text_beam_probs = text_beam_probs.reshape((-1, 1))
                        # text_traj = text_traj[text_y]
                    
                    vrd_prev_relation = vrd_chosen_relation
                    vrd_current_entity = vrd_chosen_entity
                    for j in range(bsz):
                        vrd_traj[j].append(vrd_id2rel[(int)(vrd_chosen_relation[j])])
                        vrd_traj[j].append(vrd_id2ent[(int)(vrd_chosen_entity[j])])   

                    text_prev_relation = text_chosen_relation
                    text_current_entity = text_chosen_entity
                    for j in range(bsz):
                        text_traj[j].append(text_id2rel[(int)(text_chosen_relation[j])])
                        text_traj[j].append(text_id2ent[(int)(text_chosen_entity[j])])  

                    if args.beam:
                        vrd_log_probs = vrd_beam_probs
                        text_log_probs = text_beam_probs
                    else:
                        vrd_log_probs += vrd_logits.clone().detach().cpu().numpy()[np.arange(vrd_log_probs.shape[0]), vrd_action_idx.cpu().numpy()]
                        text_log_probs += text_logits.clone().detach().cpu().numpy()[np.arange(text_log_probs.shape[0]), text_action_idx.cpu().numpy()] # [B*num_rollouts]
                
                # select traj by log prob
                vrd_log_probs = np.reshape(vrd_log_probs, (ori_bsz, vrd_option.test_rollouts))
                sorted_vrd_indx = np.argsort(-vrd_log_probs) # [ori_B, rollouts]
                vrd_traj_lst = []
                for j in range(ori_bsz):
                    vrd_traj_lst += vrd_traj[j*vrd_option.test_rollouts + sorted_vrd_indx[j][0]]
                if vrd_traj_lst == []:
                    vrd_traj_lst.append('<pad>')
                vrd_traj_id = [vrd_type_id] + list(chain(*[tokenizer.convert_tokens_to_ids(tokenizer.tokenize(d)) for d in vrd_traj_lst]))
                
                text_log_probs = np.reshape(text_log_probs, (ori_bsz, text_option.test_rollouts))
                sorted_text_indx = np.argsort(-text_log_probs) # [ori_B, rollouts]
                text_traj_lst = []
                for j in range(ori_bsz):
                    text_traj_lst += text_traj[j*text_option.test_rollouts + sorted_text_indx[j][0]]
                if text_traj_lst == []:
                    text_traj_lst.append('<pad>')
                text_traj_id = [text_type_id] + list(chain(*[tokenizer.convert_tokens_to_ids(tokenizer.tokenize(d)) for d in text_traj_lst]))

                

                # add selected traj to encoder input
                vrd_traj_id = [vrd_type_id] + list(chain(*[tokenizer.convert_tokens_to_ids(tokenizer.tokenize(d)) for d in vrd_traj_lst]))
                text_traj_id = [text_type_id] + list(chain(*[tokenizer.convert_tokens_to_ids(tokenizer.tokenize(d)) for d in text_traj_lst]))
                
                comm_traj_token_type = [vrd_type_id] * len(vrd_traj_id) + [text_type_id] * len(text_traj_id)
                comm_traj_id = vrd_traj_id + text_traj_id
                comm_traj_id = torch.tensor(comm_traj_id, device=args.device).long()
                comm_traj_token_type = torch.tensor(comm_traj_token_type, device=args.device).long()

                hypstr = beam_search(test_dataset[idx]["caption"], test_dataset[idx]["history"], tokenizer, model, args, comm_traj_id, comm_traj_token_type, video=i3d)
                hypstr = hypstr[0][0]
                hypstr=tokenizer.decode(hypstr, skip_special_tokens=True)

                # visualize triple
                if args.visualize:
                    ref_lsts = test_dataset[idx]['ref']
                    if ref_lsts == []:
                        ref_traj = ['<pad>', '<pad>', '<pad>']
                    else:
                        ref_traj = test_dataset[idx]['ref'][np.random.choice(len(test_dataset[idx]['ref']), 1)[0]]
                    ref_lst = ref_dct[vid]['reference']
                    sep = '   |   '
                    text_trip_str = ' '.join(text_traj_lst)
                    video_trip_str = ' '.join(vrd_traj_lst)
                    ref_trip_str = ' '.join(ref_traj)
                    ref_str = ''
                    for ref in ref_lst:
                        ref_str += ref + ' - '
                    with open('traj_logs/' + args.log_set + '_beam' + str(args.beam) + '_step' + str(args.step) + '.log', 'a') as fh:
                        fh.write(args.ckptid + '-' + vid + '\ntext_trip: ' + text_trip_str + '\nvideo_trip: ' + video_trip_str + '\nreference_trip: ' \
                             + ref_trip_str + '\ngenerated_answer: ' + hypstr + '\nreference_answer: ' + ref_str + '\n\n')
                
                if args.log:
                    logging.info('HYP: ' + hypstr)
                pred_dialog['dialog'][t]['answer'] = hypstr
                if args.log:
                    logging.info('ElapsedTime: %f' % (time.time() - start_time))
                    logging.info('-----------------------')
    result = {'dialogs': result_dialogs}

    if args.log:
        logging.info('----------------')
        logging.info('wall time = %f' % (time.time() - start_time))
    if args.output:
        if args.log:
            logging.info('writing results to ' + args.output)
        json.dump(result, open(args.output, 'w'), indent=4)
    if args.log:
        logging.info('done')
