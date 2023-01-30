import json


class Video_data_loader():
    def __init__(self, option):
        self.option = option

        self.entity2num = None
        self.num2entity = None

        self.relation2num = None
        self.num2relation = None
        self.relation2inv = None

        self.num_relation = 0
        self.num_entity = 0
        self.num_operator = 0

        self.load_data_all()

    def load_data_all(self):

        text_trip_path = 'data/mm-graph/semantic_graph/avsd_video/cls_trip.json'
        
        with open('data/mm-graph/semantic_graph/avsd_video/cls_trip_rel2id.json') as jh:
            self.relation2num = json.load(jh)
            new_relation2num = {}
            for k in self.relation2num.keys():
                new_relation2num[k] = int(self.relation2num[k])
            self.relation2num = new_relation2num

        with open('data/mm-graph/semantic_graph/avsd_video/cls_trip_id2rel.json') as jh:
            self.num2relation = json.load(jh)
            new_num2relation = {}
            for k in self.num2relation.keys():
                new_num2relation[int(k)] = self.num2relation[k]
            self.num2relation = new_num2relation

        with open('data/mm-graph/semantic_graph/avsd_video/cls_trip_ent2id.json') as jh:
            self.entity2num = json.load(jh)
            new_entity2num = {}
            for k in self.entity2num.keys():
                new_entity2num[k] = int(self.entity2num[k])
            self.entity2num = new_entity2num
        with open('data/mm-graph/semantic_graph/avsd_video/cls_trip_id2ent.json') as jh:
            self.num2entity = json.load(jh)
            new_num2entity = {}
            for k in self.num2entity.keys():
                new_num2entity[int(k)] = self.num2entity[k]
            self.num2entity = new_num2entity
        
        self._add_item(self.relation2num, self.num2relation, "<equal>")
        self._add_item(self.relation2num, self.num2relation, "<pad>")
        self._add_item(self.relation2num, self.num2relation, "<start>")
        self._add_item(self.entity2num, self.num2entity, "<pad>")
        # print(self.relation2num)

        self.num_relation = len(self.relation2num)
        self.num_entity = len(self.entity2num)
        print("video_num_relation: ", self.num_relation)
        print("video_num_entity: ", self.num_entity)
        
        with open(text_trip_path) as jh:
            self.data = json.load(jh)

    def _add_item(self, obj2num, num2obj, item):
        count = len(obj2num)
        obj2num[item] = count
        num2obj[count] = item

    def get_graph_data(self, vid):
        # original graph
        triplets = []
        for trip in self.data[vid]:
            if trip != []:
                triplets.append([self.entity2num[trip[0]], self.relation2num[trip[1]], self.entity2num[trip[2]]])
            else:
                ent_pad_id = self.entity2num['<pad>']
                rel_pad_id = self.relation2num['<pad>']
                triplets.append([ent_pad_id, rel_pad_id, ent_pad_id])
        return triplets

    def get_all_ents(self, vid):
        ent_lst = []
        for trip in self.data[vid]:
            if trip != []:
                if trip[0] not in ent_lst:
                    ent_lst.append(trip[0])
                if trip[2] not in ent_lst:
                    ent_lst.append(trip[2])
        if '<pad>' not in ent_lst:
            ent_lst.append('<pad>')
        return ent_lst

