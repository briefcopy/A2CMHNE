#coding=utf-8

import random
import pickle
import numpy as np


class MetaNetwork(object):
    def __init__(self):
        self.type_id_index_dict = {}
        self.type_index_id_dict = {}
        self.last_index_dict = {}
        self.meta_adj_dict = {}
        self.meta_sample_list_dict = {}

        self.type_index_attrs_dict = {}
        self.type_data_dict = {}

    # ['content', 'image'] => 'content_image'
    @staticmethod
    def create_meta_path(node_types):
        return "_".join(node_types)

    # get value
    # if not exist, create value by default_value
    @staticmethod
    def get_or_create(dict_object, key, default_value):
        if key in dict_object:
            return dict_object[key]
        else:
            dict_object[key] = default_value
            return default_value

    # add edge with meta type: node_types[0] => node_types[1]
    def add_edge(self, node_types, index0, index1):
        meta_path = MetaNetwork.create_meta_path(node_types)
        adj_dict = MetaNetwork.get_or_create(self.meta_adj_dict, meta_path, {})
        adjs = MetaNetwork.get_or_create(adj_dict, index0, [])
        if index1 not in adjs:
            adjs.append(index1)

    # add edge with meta type: node_types[0] => node_types[1]
    def add_edges(self, node_types, index0, index1):
        self.add_edge(node_types, index0, index1)
        self.add_edge([node_types[1], node_types[0]], index1, index0)


    def build_meta_adj(self, meta_node_types, direct = False):
        meta_types = [self.create_meta_path(meta_node_types[i:i+2]) for i in range(len(meta_node_types) - 1)]
        meta_adjs = [self.meta_adj_dict[meta_type] for meta_type in meta_types]

        same_type = meta_node_types[0] == meta_node_types[-1]

        node_types  = [meta_node_types[0], meta_node_types[-1]]

        def add_meta(first_index_0, current_index_0, current_meta_index):

            if current_index_0 not in meta_adjs[current_meta_index]:
                return

            for index_1 in meta_adjs[current_meta_index][current_index_0]:
                if current_meta_index < len(meta_types) - 1:
                    add_meta(first_index_0, index_1, current_meta_index + 1)
                else:

                    if same_type and first_index_0 == index_1:
                        continue

                    if direct:
                        self.add_edge(node_types, first_index_0, index_1)
                    else:
                        self.add_edges(node_types, first_index_0, index_1)

        for first_index_0 in meta_adjs[0]:
            add_meta(first_index_0, first_index_0, 0)
        self.build_sample_list()




    # get index by id for node_type
    def get_index(self, node_type, id):
        return self.type_id_index_dict[node_type][id]

    def get_indices(self, node_type, node_ids):
        return [self.get_index(node_type, node_id) for node_id in node_ids]

    def has_id(self, node_type, id):
        return id in self.type_id_index_dict[node_type]

    # create index by id for node_type
    def get_or_create_index(self, node_type, id, attrs=None):
        id_index_dict = MetaNetwork.get_or_create(self.type_id_index_dict, node_type, {})
        if id in id_index_dict:
            return id_index_dict[id]
        else:
            last_index = MetaNetwork.get_or_create(self.last_index_dict, node_type, -1)
            last_index += 1
            self.last_index_dict[node_type] = last_index
            id_index_dict[id] = last_index

            index_id_dict = MetaNetwork.get_or_create(self.type_index_id_dict, node_type, {})
            index_id_dict[last_index] = id

            if attrs is not None:
                index_attrs_dict = MetaNetwork.get_or_create(self.type_index_attrs_dict, node_type, {})
                index_attrs_dict[last_index] = attrs

            return last_index

    def get_attr(self, node_type, node_index, attr_name):
        return self.type_index_attrs_dict[node_type][node_index][attr_name]

    def get_attr_list(self, node_type, node_indices, attr_name):
        return [self.get_attr(node_type, node_index, attr_name) for node_index in node_indices]

    def get_id(self, node_type, index):
        return self.type_index_id_dict[node_type][index]

    def get_ids(self, node_type, node_indices):
        return [self.get_id(node_type, node_index) for node_index in node_indices]

    def sample_node(self, node_type):
        last_index = self.last_index_dict[node_type]
        return random.randint(0, last_index)

    def meta_sample_node(self, meta_path):
        # meta_path = create_meta_path(node_types)
        sample_list = self.meta_sample_list_dict[meta_path]
        return sample_list[random.randint(0, len(sample_list) - 1)]

    # def sample_ndcg(self, node_types, num_negative_samples):
    #     meta_path = self.create_meta_path(node_types)
    #     node_a = self.meta_sample_node(meta_path)
    #     adjs = self.meta_adj_dict[meta_path][node_a]
    #     node_b = adjs[random.randint(0, len(adjs)) - 1]
    #
    #     node_neg_list = []
    #
    #     for _ in range(num_negative_samples):
    #         while True:
    #             node_neg = self.sample_node(node_types[1])
    #             if node_neg in adjs or node_neg == node_a or node_neg in node_neg_list:
    #                 continue
    #             else:
    #                 break
    #         node_neg_list.append(node_neg)
    #     return [[node_a, node_b, node_neg] for node_neg in node_neg_list]

    def sample_ndcg(self, node_types, num_positive_samples, num_negative_samples):
        meta_path = self.create_meta_path(node_types)
        while True:
            node_a = self.meta_sample_node(meta_path)
            adjs = self.meta_adj_dict[meta_path][node_a]


            if len(adjs) < num_positive_samples:
                continue

            node_b_list = random.sample(adjs, num_positive_samples)
            node_neg_list = []

            for _ in range(num_negative_samples):
                while True:
                    node_neg = self.sample_node(node_types[1])
                    if node_neg in adjs or node_neg == node_a or node_neg in node_neg_list:
                        continue
                    else:
                        break
                node_neg_list.append(node_neg)
            answer_list = node_b_list + node_neg_list
            # print([[node_a, answer] for answer in answer_list])
            return [[node_a, answer] for answer in answer_list]

    def sample_ndcgs(self, node_types, num, num_positive_samples, num_negative_samples):
        samples = [
            self.sample_ndcg(node_types, num_positive_samples, num_negative_samples) for _ in range(num)
        ]
        return np.concatenate(samples, 0)

    # def sample_ndcgs(self, node_types, num, num_negative_samples):
    #     samples = [
    #         self.sample_ndcg(node_types, num_negative_samples) for _ in range(num)
    #     ]
    #     return np.concatenate(samples, 0)
        # return [list(t) for t in list(zip(*samples))]

    # def sample_full_ndcg(self, node_types, num_negative_samples):
    #     meta_path = self.create_meta_path(node_types)
    #     node_a = 3577#self.meta_sample_node(meta_path)
    #     adjs = self.meta_adj_dict[meta_path][node_a]
    #
    #     samples = []
    #
    #     for node_b in adjs[0:4]:
    #         node_neg_list = []
    #         for _ in range(num_negative_samples):
    #             while True:
    #                 node_neg = self.sample_node(node_types[1])
    #                 if node_neg in adjs or node_neg == node_a or node_neg in node_neg_list:
    #                     continue
    #                 else:
    #                     break
    #             node_neg_list.append(node_neg)
    #             samples.append([node_a, node_b, node_neg])
    #     samples = np.array(samples)
    #     return samples

    def sample_triple(self, node_types, node_a=None):
        meta_path = self.create_meta_path(node_types)
        if node_a is None:
            node_a = self.meta_sample_node(meta_path)
        adjs = self.meta_adj_dict[meta_path][node_a]
        node_b = adjs[random.randint(0, len(adjs)) - 1]
        while True:
            node_neg = self.sample_node(node_types[1])
            if node_neg in adjs or node_neg == node_a:
                continue
            else:
                break
        return node_a, node_b, node_neg

    def sample_triples(self, node_types, num):
        samples = []
        for i in range(num):
            samples.append(self.sample_triple(node_types))
        return [list(t) for t in list(zip(*samples))]

    def create_triple_batch_generator(self, node_types, batch_size):
        question_datas = self.type_data_dict[node_types[0]]
        answer_datas = self.type_data_dict[node_types[1]]

        while True:
            batch_question_indice, batch_answer_indice, batch_negative_answer_indice = \
                self.sample_triples(node_types, batch_size)

            batch_question_datas = question_datas[batch_question_indice]
            batch_answer_datas = answer_datas[batch_answer_indice]
            batch_negative_answer_datas = answer_datas[batch_negative_answer_indice]

            yield batch_question_datas, batch_answer_datas, batch_negative_answer_datas

    def num_nodes(self, node_type):
        return len(self.type_id_index_dict[node_type])

    def compute_adj(self, node_types, dims=None):

        if dims is None:
            dims = [self.num_nodes(node_type) for node_type in node_types]

        adj = np.zeros(dims, dtype=np.float32)
        meta_path = MetaNetwork.create_meta_path(node_types)
        adj_dict = self.meta_adj_dict[meta_path]
        for node_index0, value in adj_dict.items():
            for node_index1 in value:
                adj[node_index0, node_index1] = 1
        return adj

    def build_sample_list(self):
        for meta_path in self.meta_adj_dict:
            self.meta_sample_list_dict[meta_path] = list(self.meta_adj_dict[meta_path].keys())

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def compute_transition_matrix(adj):
        result = adj / np.sum(adj, axis=1).reshape(-1, 1)
        result[np.isnan(result)] = 0.0
        return result

    @staticmethod
    def compute_PPMI(Ak):
        Ak = Ak / Ak.sum(axis=1).reshape(-1, 1)
        Ak[np.isnan(Ak)] = 0.0

        PPMI = np.log(Ak / np.sum(Ak, axis=0)) - np.log(1.0 / Ak.shape[0])
        PPMI[PPMI < 0] = 0
        PPMI[np.isnan(PPMI)] = 0

        PPMI[np.isinf(PPMI)] = 0.0

        PPMI[np.isneginf(PPMI)] = 0.0
        return PPMI




