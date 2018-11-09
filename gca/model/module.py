# coding=utf-8

import numpy as np
import tensorflow as tf


from gca.data.network import MetaNetwork


def create_embedding_tables(network:MetaNetwork, node_types, embedding_size, second_order=True):
    # node_type => in/out => embedding_matrix
    type_embedding_dict = {}
    for node_type in node_types:
        num_nodes = network.num_nodes(node_type)
        in_embeddings = tf.get_variable("in_embedding_{}".format(node_type), [num_nodes, embedding_size], initializer=tf.truncated_normal_initializer(stddev=1.0 / embedding_size))
        if second_order:
            out_embeddings = tf.get_variable("out_embedding_{}".format(node_type), [num_nodes, embedding_size], initializer=tf.truncated_normal_initializer(stddev=1.0 / embedding_size))
        else:
            out_embeddings = in_embeddings
        type_embedding_dict[node_type] = {
            "in": in_embeddings,
            "out": out_embeddings
        }
    return type_embedding_dict


def eval_embeddings(embeddings, indices, sess):
    embedded = tf.nn.embedding_lookup(embeddings, indices)
    return sess.run(embedded)


def build_line_losses_by_indices(in_embeddings, out_embeddings, a_indices, b_indices, neg_b_indices, drop_rate=None):
    embedded_a = tf.nn.embedding_lookup(in_embeddings, a_indices)
    embedded_b = tf.nn.embedding_lookup(out_embeddings, b_indices)
    embedded_neg_b = tf.nn.embedding_lookup(out_embeddings, neg_b_indices)
    return build_line_losses(embedded_a, embedded_b, embedded_neg_b, drop_rate)

    # pos_logits = tf.reduce_sum(embedded_a * embedded_b, axis=-1)
    # neg_logits = tf.reduce_sum(embedded_a * embedded_neg_b, axis=-1)
    #
    # # pos_logits = tf.reduce_sum(tf.multiply(embedded_a, embedded_b), axis=-1)
    # # neg_logits = tf.reduce_sum(tf.multiply(embedded_a, embedded_neg_b), axis=-1)
    #
    # pos_losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=pos_logits, labels=tf.ones_like(pos_logits))
    # neg_losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=neg_logits, labels=tf.zeros_like(neg_logits))
    #
    # losses = pos_losses + neg_losses
    # # loss = tf.reduce_mean(losses)
    # return losses


def build_line_losses(embedded_a, embedded_b, embedded_neg_b, drop_rate=None):
    if drop_rate is not None:
        embedded_a = tf.layers.dropout(embedded_a, rate=drop_rate)
        embedded_b = tf.layers.dropout(embedded_b, rate=drop_rate)
        embedded_neg_b = tf.layers.dropout(embedded_neg_b, rate=drop_rate)

    pos_logits = tf.reduce_sum(embedded_a * embedded_b, axis=-1)
    neg_logits = tf.reduce_sum(embedded_a * embedded_neg_b, axis=-1)
    pos_losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=pos_logits, labels=tf.ones_like(pos_logits))
    neg_losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=neg_logits, labels=tf.zeros_like(neg_logits))
    losses = pos_losses + neg_losses
    return losses


def build_bilinear_line_losses_by_indices(in_embeddings, out_embeddings, a_indices, b_indices, neg_b_indices, W):
    embedded_a = tf.nn.embedding_lookup(in_embeddings, a_indices)
    embedded_b = tf.nn.embedding_lookup(out_embeddings, b_indices)
    embedded_neg_b = tf.nn.embedding_lookup(out_embeddings, neg_b_indices)
    return build_bilinear_line_losses(embedded_a, embedded_b, embedded_neg_b, W)


def build_bilinear_line_losses(embedded_a, embedded_b, embedded_neg_b, W):
    embedded_a = tf.matmul(embedded_a, W)
    pos_logits = tf.reduce_sum(embedded_a * embedded_b, axis=-1)
    neg_logits = tf.reduce_sum(embedded_a * embedded_neg_b, axis=-1)
    pos_losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=pos_logits, labels=tf.ones_like(pos_logits))
    neg_losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=neg_logits, labels=tf.zeros_like(neg_logits))
    losses = pos_losses + neg_losses
    return losses


class Attention(tf.keras.Model):
    def __init__(self, num_units, num_heads, drop_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # num_heads = 1
        # num_units = 1
        self.Q_func = tf.layers.Dense(num_units, activation=tf.nn.relu, name="Q")
        self.K_func = tf.layers.Dense(num_units, activation=tf.nn.relu, name="K")
        self.V_func = tf.layers.Dense(num_units, activation=tf.nn.relu, name="V")
        # self.V_func = tf.layers.Dense(num_units, activation=None, name="V")
        self.num_heads = num_heads
        self.drop_rate = drop_rate

    def call(self, inputs, training=None, mask=None):
        # return inputs[0] * 0.5 + inputs[1] * 0.5
        # queries = tf.stack(inputs, axis=1)
        # keys = queries
        queries, keys = inputs

        # queries = tf.layers.dropout(queries, rate=self.drop_rate)
        # keys = tf.layers.dropout(keys, rate=self.drop_rate)

        Q = self.Q_func(queries)
        K = self.K_func(keys)
        V = self.V_func(keys)

        Q_ = tf.concat(tf.split(Q, self.num_heads, axis=-1), axis=0)
        K_ = tf.concat(tf.split(K, self.num_heads, axis=-1), axis=0)
        V_ = tf.concat(tf.split(V, self.num_heads, axis=-1), axis=0)

        attention_matrix = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
        attention_matrix = tf.nn.softmax(attention_matrix)
        attention_matrix = tf.layers.dropout(attention_matrix, rate=self.drop_rate)


        outputs = tf.matmul(attention_matrix, V_)
        outputs = tf.concat(tf.split(outputs, self.num_heads, axis=0), axis=-1)
        outputs= queries * 0.5 + outputs * 0.5

        # outputs = tf.layers.dropout(outputs, rate=self.drop_rate)
        return outputs

    def l2_loss(self):
        kernel_vars = [var for var in self.variables if "kernel" in var.name]
        l2_losses = sum([tf.nn.l2_loss(kernel_var) for kernel_var in kernel_vars])
        return l2_losses
