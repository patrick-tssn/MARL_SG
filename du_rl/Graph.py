import torch
import numpy as np
from collections import defaultdict
import copy


class Knowledge_graph():
    def __init__(self, option, data_loader, data):
        self.option = option
        self.data = data
        self.data_loader = data_loader
        self.out_array = None
        self.all_correct = None
        self.rel_pad_id = data_loader.relation2num["<pad>"]
        self.ent_pad_id = data_loader.entity2num["<pad>"]
        self.construct_graph()

    # 根据原始数据构建知识图谱，out_array存储每个节点向外的出口路径数组
    def construct_graph(self):
        # default graph
        all_out_dict = defaultdict(list)
        for head, relation, tail in self.data:
            all_out_dict[head].append((relation, tail))

        all_correct = defaultdict(set)
        out_array = np.ones((self.option.num_entity, self.option.max_out, 2), dtype=np.int64)
        out_array[:, :, 0] *= self.data_loader.relation2num["<pad>"]
        out_array[:, :, 1] *= self.data_loader.entity2num["<pad>"]
        more_out_count = 0
        for head in all_out_dict:
            out_array[head, 0, 0] = self.data_loader.relation2num["<equal>"]
            out_array[head, 0, 1] = head
            num_out = 1
            for relation, tail in all_out_dict[head]:
                if num_out == self.option.max_out:
                    more_out_count += 1
                    break
                out_array[head, num_out, 0] = relation
                out_array[head, num_out, 1] = tail
                num_out += 1
                all_correct[(head, relation)].add(tail)
        self.out_array = torch.from_numpy(out_array)
        self.all_correct = all_correct
        # print("more_out_count", more_out_count)
        self.out_array = self.out_array.to(self.option.device)

    # 获取从图谱上current_entities的out_relations, out_entities
    def get_out(self, current_entities):
        # ret = copy.deepcopy(self.out_array[current_entities, :])
        ret = copy.deepcopy(self.out_array[current_entities, :, :]) # [B, max_out, 2]
        return ret

    def get_next(self, current_entities, start_entities, query_relations, num_rollout, last_step, answers, all_correct_answers):
        # default graph
        next_out = self.out_array[current_entities, :, :].copy() # [B, max_out, 2]
        for i in range(current_entities.shape[0]):
            if last_step:
                relations = next_out[i, :, 0]
                entities = next_out[i, :, 1]
                correct_e2 = answers[i]
                for j in range(entities.shape[0]):
                    if entities[j] in all_correct_answers[int(i/num_rollout)] and entities[j] != correct_e2:
                        relations[j] = self.rel_pad_id
                        entities[j] = self.ent_pad_id
        return next_out # [B, 2]
    
    def init_actions(self, current_entities, start_entities, query_relations, answers):
        next_out = self.out_array[current_entities, :, :].copy()
        for i in range(current_entities.shape[0]):
            if current_entities[i] == start_entities[i]:
                relations = next_out[i, :, 0]
                entities = next_out[i, :, 1]
                mask = np.logical_and(relations == query_relations[i], entities == answers[i])
                next_out[i, :, 0][mask] = self.rel_pad_id
                next_out[i, :, 1][mask] = self.ent_pad_id
        return next_out # [B, 2]