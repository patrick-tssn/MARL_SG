# coding: utf-8
# author: noctli
import sys
sys.path.append('nltk_data')
import json
import copy
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset
from itertools import chain
from gensim.models import KeyedVectors
from du_rl.vrd_utils import compute_cosine_similarity

# from train import SPECIAL_TOKENS, MODEL_INPUTS, PADDED_INPUTS
SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>","<cap>", "<video>", "<pad>", "<vrd>", "<trd>", "<start>", "<equal>"]
SPECIAL_TOKENS_DICT = {'bos_token': "<bos>", 'eos_token': "<eos>", 'additional_special_tokens': ["<speaker1>", "<speaker2>", "<video>", "<cap>", "<vrd>", "<trd>", "<start>","<equal>"], 'pad_token': "<pad>"}
MODEL_INPUTS = ["input_ids", "token_type_ids","lm_labels"]
PADDED_INPUTS = ["input_ids", "token_type_ids","lm_labels"]


def tokenize(obj,tokenizer):
    if isinstance(obj, str): # 对 string 格式的文本 tokenize
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict): # 对字典格式的文本 tokenize -> key:tokenized value
        return dict((n, tokenize(o)) for n, o in obj.items())
    return list(tokenize(o) for o in obj) # 其他情况

def get_sorted_start_triple_lst(query, triples, model):
    if type(query) is str:
        res_sim_lst = []
        for triple in triples:
            try:
                head_ent_lst = triple[0].split()
            except Exception as e:
                print(e)
                print(triple)
            rel_lst = triple[1].split()
            tail_ent_lst = triple[2].split()
            sim = compute_cosine_similarity(head_ent_lst+rel_lst+tail_ent_lst, query.split(), model)
            res_sim_lst.append((triple, sim))
        res_sim_lst.sort(key=lambda x:x[1], reverse=True)
        res_lst = [x[0] for x in res_sim_lst]
        res_lst.append(['<pad>', '<pad>', '<pad>'])
    elif type(query) is list:
        trip_sim_dct = {}
        for que in query:
            for triple in triples:
                head_ent_lst = triple[0].split()
                rel_lst = triple[1].split()
                tail_ent_lst = triple[2].split()
                sim = compute_cosine_similarity(head_ent_lst+rel_lst+tail_ent_lst, que.split(), model)
                trip_key = triple[0] + '_' + triple[1] + '_' + triple[2]
                if trip_key not in trip_sim_dct:
                    trip_sim_dct[trip_key] = sim
                else:
                    trip_sim_dct[trip_key] = max(trip_sim_dct[trip_key], sim)
        trip_sim_lst = sorted(trip_sim_dct.items(), key=lambda x:x[1], reverse=True)
        res_lst = [x[0].split('_') for x in trip_sim_lst]
        res_lst.append(['<pad>', '<pad>', '<pad>'])
    else:
        raise ValueError('invalid query type')

    return res_lst

def get_dataset(tokenizer, data_file, feature_path=None, text_traj_path=None, video_traj_path=None, undisclosed_only=False, n_history=3):
    """
    input data format: read datafile: {"image_id": "", "summary": "", "dialogs""[{"answer":"","question":""}], "caption":""}
    output data format: dialog_list: num of dialogs * num of turns [{'vid':'','history':max 3 turns [[q],[a],...[cur q]],'answer':[a],'caption':[[caption list], [summary list]]}]
                        all_feature: {'vggish':{'vid':(filepath,filepath)},'i3d_flow':{},'i3d_rgb':{}}
    """
    train_ref_path = 'data/mm-graph/semantic_graph/avsd_ref/train_his_trip.json'
    train_ref_data = json.load(open(train_ref_path, 'r'))
    test_ref_path = 'data/mm-graph/semantic_graph/avsd_test_ref/test_his_trip.json'
    test_ref_data = json.load(open(test_ref_path, 'r'))
    dialog_data = json.load(open(data_file, 'r'))
    text_traj_data = json.load(open(text_traj_path, 'r'))
    video_traj_data = json.load(open(video_traj_path, 'r'))
    word2vec_model = KeyedVectors.load_word2vec_format('data/gensim-data/word2vec-google-news-300', binary=True)
    dialog_list = []
    vid_set = set()
    for dialog in dialog_data['dialogs']: # dict {}
        caption = [tokenize(dialog['caption'],tokenizer)] + [tokenize(dialog['summary'],tokenizer)] # capition 和 summary 合并 [[caption id list],[summary id list]]
        questions = [tokenize(d['question'],tokenizer) for d in dialog['dialog']] # [[question id list],[],...]
        answers = [tokenize(d['answer'],tokenizer) for d in dialog['dialog']] # [[answer id list], [], ...]
        queries = [d['question'] for d in dialog['dialog']]
        references = [d['answer'] for d in dialog['dialog']]
        vid = dialog["image_id"] # vid
        vid_set.add(vid) # vid set
        video_traj_lst = video_traj_data[vid]
        if undisclosed_only: # train data always false
            it = range(len(questions) - 1, len(questions))
        else: # train
            it = range(len(questions))
        qalist=[]
        history = [] # history: list
        if undisclosed_only: # test
            for n in range(len(questions)-1):
                qalist.append(questions[n])
                qalist.append(answers[n])
            history=qalist[max(-len(qalist),-n_history*2):]
        for n in it: # train range(len(questions))
            if undisclosed_only: # test
                assert dialog['dialog'][n]['answer'] == '__UNDISCLOSED__'
            question = questions[n]
            answer = answers[n]
            query = queries[n]
            reference = references[n]
            history.append(question)
            sorted_text_traj = get_sorted_start_triple_lst(query, text_traj_data[vid+'_'+str(n)], word2vec_model) # caption and context graph
            sorted_video_traj = get_sorted_start_triple_lst(query, video_traj_lst, word2vec_model)
            if not undisclosed_only:
                train_ref = train_ref_data[vid][n]
            else:
                train_ref = test_ref_data[vid]
            if n_history == 0:
                item = {'vid': vid, 'history': [question], 'answer': answer, 'caption': caption}
            else: # default 3
                item = {'vid': vid, 'history': history, 'answer': answer, 'caption': caption, 'text_traj': sorted_text_traj, 'video_traj': sorted_video_traj, 'ref': train_ref, 'subid': vid+'_'+str(n)}
            dialog_list.append(item)
            qalist.append(question)
            qalist.append(answer)
            history=qalist[max(-len(qalist),-n_history*2):]

    all_features = {}
    if feature_path is not None:
        fea_types = ['vggish', 'i3d_flow', 'i3d_rgb']
        dataname = '<FeaType>/<ImageID>.npy'
        for ftype in fea_types:
            if undisclosed_only:
                basename = dataname.replace('<FeaType>', ftype+'_testset')
            else:
                basename = dataname.replace('<FeaType>', ftype)
            features = {}
            for vid in vid_set:
                filename = basename.replace('<ImageID>', vid)
                filepath = feature_path + filename
                features[vid] = (filepath, filepath)
            all_features[ftype] = features
        return dialog_list, all_features
        """
        dialog_list: [{'vid':'','history':'','answer':'','caption':''}]
        all_feature: {'vggish':{'vid':(filepath,filepath)},'i3d_flow':{},'i3d_rgb':{}}
        """
    return dialog_list


class AVSDDataSet(Dataset):
    def __init__(self, dialogs, tokenizer, features=None, drop_rate=0, train=True, model='gpt'):
        self.dialogs = dialogs # dialog_list
        self.features = features # all_feature
        self.tokenizer = tokenizer
        self.drop_rate = drop_rate
        self.train = train
        self.model = model

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, index):
        dialog = self.dialogs[index]
        vid = dialog['vid']
        his = self.dialogs[index]['history'] # [[q],[a],...[q]] 
        cap = self.dialogs[index]['caption'] # [[caption ], [summary]]
        ans = self.dialogs[index]['answer'] # [[a]]
        text_traj_lst = self.dialogs[index]['text_traj'] # [[triple], ...]
        video_traj_lst = self.dialogs[index]['video_traj'] # [[triple], ...]
        ref_lst = self.dialogs[index]['ref'] # [[triple], ...]
        subid = self.dialogs[index]['subid'] # vid_n
        
        if self.drop_rate == 1: 
            instance, _ = build_input_from_segments(cap, his, ans, self.tokenizer, video=False, drop_caption=True, train=self.train, model=self.model)
        elif self.drop_rate == 0: # train/validate: drop_rate = 0
            instance, _ = build_input_from_segments(cap, his, ans, self.tokenizer, video=False, drop_caption=False, train=self.train, model=self.model)
        else:
            raise ValueError('NO IMPLEMENTED DROP_RATE')

        if self.drop_rate == 0:
            cap_ids = torch.Tensor(instance["cap_ids"]).long()
            cap_type_ids = torch.Tensor(instance["cap_type_ids"]).long()
        elif self.drop_rate == 1:
            cap_ids = None
            cap_type_ids = None
        his_ids = torch.Tensor(instance["his_ids"]).long()
        his_type_ids = torch.Tensor(instance["his_type_ids"]).long()
        query_ids = torch.Tensor(instance["query_ids"]).long()
        query_type_ids = torch.Tensor(instance["query_type_ids"]).long()
        answer_ids = torch.Tensor(instance["answer_ids"]).long()
        answer_type_ids = torch.Tensor(instance["answer_type_ids"]).long()
        
        if self.features is not None:
            try:
                vgg = np.load(self.features[0]["vggish"][vid][0]) # (-1, 128) 128-dimension
                i3d_flow = np.load(self.features[0]["i3d_flow"][vid][0]) # (-1, 2048) 2048-dimension
                i3d_rgb = np.load(self.features[0]["i3d_rgb"][vid][0]) # (-1, 2048)
            except KeyError: # validate_data
                vgg = np.load(self.features[1]["vggish"][vid][0])
                i3d_flow = np.load(self.features[1]["i3d_flow"][vid][0])
                i3d_rgb = np.load(self.features[1]["i3d_rgb"][vid][0])
            
            # sample_step = i3d_flow.shape[0] // vgg.shape[0]
            # if sample_step == 0:
            #     sample_step = 1
            sample_step = 1

            sample_i3d_flow = i3d_flow[range(1, i3d_flow.shape[0], sample_step)]
            sample_i3d_rgb = i3d_rgb[range(1, i3d_rgb.shape[0], sample_step)]

            vgg = torch.from_numpy(vgg).float()
            i3d_flow = torch.from_numpy(sample_i3d_flow).float()
            i3d_rgb = torch.from_numpy(sample_i3d_rgb).float()
            min_length = min([i3d_flow.size(0), i3d_rgb.size(0), vgg.size(0)])
            i3d = torch.cat([i3d_flow[:min_length], i3d_rgb[:min_length], vgg[:min_length]], dim=1) # (32, 4224)
        else:
            i3d = None

        return cap_ids, cap_type_ids, his_ids, his_type_ids, query_ids, query_type_ids, answer_ids, answer_type_ids, i3d, vid, text_traj_lst, video_traj_lst, ref_lst, subid


def collate_fn(batch, pad_token, features=None):
    def padding(seq, pad_token):
        max_len = max([i.size(0) for i in seq])
        if len(seq[0].size()) == 1:
            result = torch.ones((len(seq), max_len)).long() * pad_token
        else:
            result = torch.ones((len(seq), max_len, seq[0].size(-1))).float()
        for i in range(len(seq)):
            result[i, :seq[i].size(0)] = seq[i]
        return result
    
    cap_lst, cap_type_lst, his_lst, his_type_lst, query_lst, query_type_lst, ans_lst, ans_type_lst, i3d_lst, vid_lst, text_traj_lsts, video_traj_lsts, ref_lsts, subid_lsts = \
        [], [], [], [], [], [], [], [], [], [], [], [], [], []
    for i in batch:
        cap_lst.append(i[0])
        cap_type_lst.append(i[1])
        his_lst.append(i[2])
        his_type_lst.append(i[3])
        query_lst.append(i[4])
        query_type_lst.append(i[5])
        ans_lst.append(i[6])
        ans_type_lst.append(i[7])
        i3d_lst.append(i[8])
        vid_lst.append(i[9])
        text_traj_lsts.append(i[10])
        video_traj_lsts.append(i[11])
        ref_lsts.append(i[12])
        subid_lsts.append(i[13])
        
    if features is not None:
        i3d = padding(i3d_lst, pad_token)
    else:
        i3d = None

    return cap_lst, cap_type_lst, his_lst, his_type_lst, query_lst, query_type_lst, ans_lst, ans_type_lst, i3d, vid_lst, text_traj_lsts, video_traj_lsts, ref_lsts, subid_lsts

class AVSDDataSet_agent(Dataset):
    def __init__(self, dialogs, tokenizer, features=None, drop_rate=0, train=True, model='gpt'):
        self.dialogs = dialogs # dialog_list
        self.features = features # all_feature
        self.tokenizer = tokenizer
        self.drop_rate = drop_rate
        self.train = train
        self.model = model

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, index):
        dialog = self.dialogs[index]
        vid = dialog['vid']
        text_traj_lst = self.dialogs[index]['text_traj'] # [[triple], ...]
        video_traj_lst = self.dialogs[index]['video_traj'] # [[triple], ...]
        ref_lst = self.dialogs[index]['ref'] # [[triple], ...]
        subid = self.dialogs[index]['subid'] # vid_n

        return vid, text_traj_lst, video_traj_lst, ref_lst, subid


def collate_fn_agent(batch, pad_token, features=None):
    
    vid_lst, text_traj_lsts, video_traj_lsts, ref_lsts, subid_lsts = [], [], [], [], []
    for i in batch:
        vid_lst.append(i[0])
        text_traj_lsts.append(i[1])
        video_traj_lsts.append(i[2])
        ref_lsts.append(i[3])
        subid_lsts.append(i[4])

    return vid_lst, text_traj_lsts, video_traj_lsts, ref_lsts, subid_lsts

    



def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padd only batches but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding if name != "labels" else -1] * (max_l - len(x)) for x in dataset[name]]
    return dataset

def build_input_from_segments(caption, history, reply, tokenizer, with_eos=True, video=False, drop_caption=False, train=True, model='gpt'):
    """
    caption: [[caption], [summary]] history: [[q], [a], ..., [q]], reply: [a]  other: default if train dataset
    """
    """ Build a sequence of input from 3 segments: caption(caption+summary) history and last reply """
    bos, eos, speaker1, speaker2, cap = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-6])
    if not drop_caption: # train/validate/test
        instance = {}
        sequence = [[bos] + list(chain(*caption))] + history + [reply + ([eos] if with_eos else [])] 
        # [[bos, caption]] + [[q], [a], ...] + [[a, eos]] -> [[bos, caption], [q], [a], ... [a, eos]] # train with_eos
        sequence = [[cap] + sequence[0] + [eos]] + [[speaker2 if (len(sequence) - i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])] 
        # [[cap, bos, caption, eos]] + [[speaker1, q], [speaker2, a], ..., [speaker2, a, eos]]
        # -> [[cap, bos, caption, eos], [speaker1, q], [speaker2, a], ..., [speaker2, a, eos] ]

        # cap his query
        instance["cap_ids"] = list(chain(*[sequence[0]]))
        instance["cap_type_ids"] = [cap] * len(sequence[0])
        instance["his_ids"] = list(chain(*sequence[1:-2]))
        instance["his_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence[1:-2]) for _ in s]
        instance["query_ids"] = list(chain(*[sequence[-2]]))
        instance["query_type_ids"] = [speaker1] * len(sequence[-2])
        instance["answer_ids"] = list(chain(*[sequence[-1]]))
        instance["answer_type_ids"] = [speaker2] * len(sequence[-1])
        assert len(instance["query_ids"]) == len(instance["query_type_ids"])
        assert len(instance["answer_type_ids"]) == len(instance["answer_ids"])

        instance["input_ids"] = list(chain(*sequence))
        # [cap, bos, caption eos, speaker1, q, ..., speaker2, a, eos]
        instance["token_type_ids"] = [cap] * len(sequence[0]) + [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence[1:]) for _ in s]
        # [cap, ...] + [speaker1, ..., speaker2, ...] -> [cat, ..., speaker1, ..., speaker2, ...]
        if video and train:
            #instance["lm_labels"] = sequence[0] + ([-1]*sum(len(s) for s in sequence[1:-1])) + sequence[-1]
            instance["lm_labels"] = sequence[0] + ([-1]*sum(len(s) for s in sequence[1:-1])) + sequence[-1]
            # [cap, bos, caption, eos] + [-1, ... ] + [speaker2, a, eos] -> [cap, bos, caption, -1, ..., speaker2, a, eos]
            instance['type_labels'] = copy.deepcopy(instance["token_type_ids"])
            instance['type_labels'][:instance["lm_labels"].count(-1)] = instance["lm_labels"][:instance["lm_labels"].count(-1)]
        else:
            instance["lm_labels"] = ([-1]*sum(len(s) for s in sequence[:-1])) + sequence[-1]
            # [-1,..., speaker2, a, eos]
            instance['type_labels'] = copy.deepcopy(instance["token_type_ids"])
            instance['type_labels'][:instance["lm_labels"].count(-1)] = instance["lm_labels"][:instance["lm_labels"].count(-1)]
    else:
        instance = {}
        sequence = history + [reply + ([eos] if with_eos else [])]
        # [[q], [a], ..., [q]] + [[a, eos]] -> [[q], [a], ..., [q], [a, eos]]
        sequence = [[speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence)]
        # sequence = [[speaker1 if (len(sequence)-i) % 2 else speaker2] + s for i, s in enumerate(sequence)]
        # [[speaker1, q], ..., [speaker2, a, eos]]

        # his query
        instance["his_ids"] = list(chain(*sequence[:-2]))
        instance["his_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence[:-2]) for _ in s]
        instance["query_ids"] = list(chain(*[sequence[-2]]))
        instance["query_type_ids"] = [speaker1] * len(sequence[-2])
        instance["answer_ids"] = list(chain(*[sequence[-1]]))
        instance["answer_type_ids"] = [speaker2] * len(sequence[-1])
        assert len(instance["query_ids"]) == len(instance["query_type_ids"])
        assert len(instance["answer_ids"]) == len(instance["answer_type_ids"])


        instance["input_ids"] = list(chain(*sequence))
        # [speaker1, q, ..., speaker2, a, eos]
        instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
        # [speaker1, ..., speaker2, ...]
        if video:
            instance["lm_labels"] = ([-1]*sum(len(s) for s in sequence[:-1])) + sequence[-1]
            # [-1, ..., speaker2, a, eos]
            instance['type_labels'] = copy.deepcopy(instance["token_type_ids"])
            instance['type_labels'][:instance["lm_labels"].count(-1)] = instance["lm_labels"][:instance["lm_labels"].count(-1)]
        else:
            instance["lm_labels"] = ([-1]*sum(len(s) for s in sequence[:-1])) + sequence[-1]
            instance['type_labels'] = copy.deepcopy(instance["token_type_ids"])
            instance['type_labels'][:instance["lm_labels"].count(-1)] = instance["lm_labels"][:instance["lm_labels"].count(-1)]
            # [-1, ..., speaker2, a, eos]

    return instance, sequence


