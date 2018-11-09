# coding=utf-8

import numpy as np
import pickle
from gca.config import *
from gca.data.network import MetaNetwork
from gca.util import is_english_word


def read_node_id_label_dict(labels_path):
    node_id_label_dict = {}
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            items = line.split()
            node_id_label_dict[items[0]] = int(items[1])
    return node_id_label_dict

def read_labeled_node_ids(labels_path):
    return list(read_node_id_label_dict().keys())
    # labeled_node_ids = []
    # with open(labels_path, "r", encoding="utf-8") as f:
    #     for line in f:
    #         items = line.split()
    #         node_id = items[0]
    #         labeled_node_ids.append(node_id)
    # return labeled_node_ids


# use node ids or indices
def split_train_and_test_nodes(nodes, training_rate, convert_to_set=False):
    nodes = np.array(nodes)
    shuffled_node_ids = list(np.random.permutation(nodes))
    split = int(len(nodes) * training_rate)
    train_nodes = shuffled_node_ids[:split]
    test_nodes = shuffled_node_ids[split:]
    if convert_to_set:
        train_nodes = set(train_nodes)
        test_nodes = set(test_nodes)
    return train_nodes, test_nodes

def load_data(pickle_data_path):
    with open(pickle_data_path, "rb") as f:
        network, train_node_indices, test_node_indices = pickle.load(f)
    return network, train_node_indices, test_node_indices

# def build_network(adjedges_path, labels_path, docs_path):
def build_data(data_dir, training_rate, pickle_data_path):
    labels_path = os.path.join(data_dir, "labels.txt")
    adjedges_path = os.path.join(data_dir, "adjedges.txt")
    docs_path = os.path.join(data_dir, "docs.txt")

    network = MetaNetwork()

    node_id_label_dict = read_node_id_label_dict(labels_path)
    labeled_node_ids = set(node_id_label_dict.keys())

    with open(docs_path, "r", encoding="utf-8") as f:
        for line in f:
            items = line.replace(":", " ").replace(".", " ").replace(",", " ")\
                .replace("'", " ").replace("/", " ").replace("(", " ").replace(")", " ").replace("-", " ").replace("?", " ").lower().split()
            # items = line.lower().split()
            node_id = items[0]
            tokens = [token for token in items[1:] if is_english_word(token) and len(token) > 1]

            # if node_0_id in node_id_label_dict:
            #     attrs = {"label": node_id_label_dict[node_0_id]}
            # else:
            #     attrs = None
            attrs = {"label": node_id_label_dict[node_id]}
            node_index = network.get_or_create_index(TYPE_NODE, node_id, attrs=attrs)
            content_node_index = network.get_or_create_index(TYPE_CONTENT, node_id)

            word_node_indices = []
            for token in tokens:
                word_node_id = "W_"+token
                word_node_index = network.get_or_create_index(TYPE_WORD, word_node_id)
                network.add_edges([TYPE_NODE, TYPE_WORD], node_index, word_node_index)
                network.add_edges([TYPE_CONTENT, TYPE_WORD], content_node_index, word_node_index)
                for context_word_index in word_node_indices:
                    network.add_edges([TYPE_WORD, TYPE_WORD], context_word_index, word_node_index)
                word_node_indices.append(word_node_index)

    labeled_node_indices = network.get_indices(TYPE_NODE, labeled_node_ids)
    train_node_indices, test_node_indices = split_train_and_test_nodes(labeled_node_indices, training_rate)
    train_node_labels = network.get_attr_list(TYPE_NODE, train_node_indices, "label")
    for train_node_index, label in zip(train_node_indices, train_node_labels):
        label_id = "l_{}".format(label)
        label_node_index = network.get_or_create_index(TYPE_LABEL, label_id)
        # print(train_node_index, label_node_index)
        network.add_edges([TYPE_NODE, TYPE_LABEL], train_node_index, label_node_index)
        network.add_edges([TYPE_CONTENT, TYPE_LABEL], train_node_index, label_node_index)

    with open(adjedges_path, "r", encoding="utf-8") as f:
        for line in f:
            items = line.split()
            node_0_id = items[0]
            # if len(items) == 1:
            #     labeled_node_ids.remove(node_0_id)
            node_0_index = network.get_or_create_index(TYPE_NODE, node_0_id)
            for node_1_id in items[1:]:
                # if node_1_id not in node_id_label_dict:
                #     continue
                node_1_index = network.get_or_create_index(TYPE_NODE, node_1_id)
                network.add_edges([TYPE_NODE, TYPE_NODE], node_0_index, node_1_index)

                # print(network.get_attr(TYPE_NODE, node_1_index, "label"))
    network.build_sample_list()

    with open(pickle_data_path, "wb") as f:
        pickle.dump([network, train_node_indices, test_node_indices], f)

    return network, train_node_indices, test_node_indices


# def load_data(data_dir, training_rate):
#     labels_path = os.path.join(data_dir, "labels.txt")
#     adjedges_path = os.path.join(data_dir, "adjedges.txt")
#     docs_path = os.path.join(data_dir, "docs.txt")
#     network, labeled_node_indices = build_network(adjedges_path, labels_path, docs_path)
#     train_node_indices, test_node_indices = split_train_and_test_nodes(labeled_node_indices, training_rate)
#     return network, train_node_indices, test_node_indices












    # samples = network.sample_triples([TYPE_NODE, TYPE_WORD], num=10)
    # print(network.get_ids(TYPE_NODE, samples[0]))
    # print(network.get_ids(TYPE_WORD, samples[1]))
    # print(network.get_ids(TYPE_WORD, samples[2]))
    # for a,b,c in zip(*samples):
    #     print(network.get_id(TYPE_NODE, a))
    #     print(network.get_id(TYPE_WORD, b))
    #     print(network.get_id(TYPE_WORD, c))
    #     print()


# build_network(adjedges_path, labels_path, docs_path)




