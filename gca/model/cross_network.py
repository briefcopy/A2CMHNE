# coding=utf-8

import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression

from gca.config import *
from gca.data.data_processing import load_data, read_node_id_label_dict, build_data
from gca.data.network import MetaNetwork
from gca.evaluation import evaluation
from gca.model.module import create_embedding_tables, eval_embeddings, build_line_losses, build_line_losses_by_indices, \
    Attention, build_bilinear_line_losses, build_bilinear_line_losses_by_indices

network, train_node_indices, test_node_indices  = build_data(data_dir, training_rate=training_rate, pickle_data_path=pickle_data_path)

print("building embedding tables")
embedding_tables = create_embedding_tables(network, [TYPE_NODE, TYPE_CONTENT, TYPE_WORD, TYPE_LABEL], embedding_size)
# embedding_tables = create_embedding_tables(network, [TYPE_NODE], embedding_size)
print(embedding_tables)

node_embeddings = embedding_tables[TYPE_NODE]["in"]
content_embeddings = embedding_tables[TYPE_CONTENT]["in"]
label_out_embeddings = embedding_tables[TYPE_LABEL]["out"]


train_node_ids = network.get_ids(TYPE_NODE, train_node_indices)
test_node_ids = network.get_ids(TYPE_NODE, test_node_indices)

attention_node_indices_placeholder = tf.placeholder(tf.int32, shape=[None])

# attentive_embedded = (tf.nn.embedding_lookup(node_embeddings, attention_node_indices_placeholder) \
#                      + tf.nn.embedding_lookup(content_embeddings, attention_node_indices_placeholder))/2


a_placeholder = tf.placeholder(tf.int32, shape=[None])
b_placeholder = tf.placeholder(tf.int32, shape=[None])
neg_b_placeholder = tf.placeholder(tf.int32, shape=[None])
drop_rate_placeholder = tf.placeholder(tf.float32)


node_types_list = [
    # [TYPE_NODE, TYPE_NODE],
    # [TYPE_NODE, TYPE_NODE],
    [TYPE_NODE, TYPE_NODE],
    [TYPE_WORD, TYPE_WORD],
    [TYPE_NODE, TYPE_WORD],
    [TYPE_WORD, TYPE_NODE]
    # [TYPE_CONTENT, TYPE_WORD],
    # [TYPE_ATTENTION, TYPE_NODE, TYPE_LABEL],
    # [TYPE_NODE, TYPE_LABEL],
    # [TYPE_CONTENT, TYPE_LABEL],
    # [TYPE_ATTENTION, TYPE_NODE, TYPE_LABEL],
    # [TYPE_ATTENTION, TYPE_NODE, TYPE_NODE],
    # [TYPE_ATTENTION, TYPE_CONTENT, TYPE_WORD]
]


def is_attention(node_types):
    return node_types[0] == TYPE_ATTENTION


attention_func = Attention(num_units=300, num_heads=6, drop_rate=drop_rate_placeholder)


def attentive_embed(node_indices):
    return tf.nn.embedding_lookup(node_embeddings, node_indices)
    multimodal_embedded = tf.stack([
        tf.nn.embedding_lookup(node_embeddings, node_indices),
        tf.nn.embedding_lookup(content_embeddings, node_indices)
    ], axis=1)
    for i in range(1):
        multimodal_embedded = attention_func([multimodal_embedded, multimodal_embedded])
    return multimodal_embedded[:,1,:]
    return tf.reduce_mean(multimodal_embedded, axis=1)
    # return attention_func([
    #     tf.nn.embedding_lookup(node_embeddings, node_indices),
    #     tf.nn.embedding_lookup(content_embeddings, node_indices)
    # ])


def average_embed(node_indices):
    return 0.5 * tf.nn.embedding_lookup(node_embeddings, node_indices) + 0.5 * tf.nn.embedding_lookup(content_embeddings, node_indices)

# embedded_attentive_a = attention_func([
#     tf.nn.embedding_lookup(node_embeddings, a_placeholder),
#     tf.nn.embedding_lookup(content_embeddings, a_placeholder)
# ])

embedded_average_a = average_embed(a_placeholder)
embedded_attentive_a = attentive_embed(a_placeholder)
#attentive_out_embeddings = tf.get_variable("attentive_out_embeddings", [network.num_nodes(TYPE_NODE), embedding_size], initializer=tf.truncated_normal_initializer(stddev=1.0 / embedding_size))
# embedded_attentive_b = attentive_embed(b_placeholder)
# embedded_attentive_neg_b = attentive_embed(neg_b_placeholder)

embedding_vars = [var for var in tf.global_variables() if "embedding" in var.name]
mean_meta_losses = []
optimizers = []


global_step = tf.get_variable("global_step", initializer=0)
embedding_decayed_learning_rate = tf.train.exponential_decay(learning_rate, global_step, 100000, 0.9999)
# attention_decayed_learning_rate = tf.train.exponential_decay(5e-3, global_step, 100000, 0.9999)
attention_decayed_learning_rate = tf.train.exponential_decay(2e-3, global_step, 100000, 0.9999)

# embedding_decayed_learning_rate = learning_rate
# attention_decayed_learning_rate = 1e-2
# l2_coe = 1e-3
l2_coe = 1e-3

# W = tf.get_variable("W", shape=[embedding_size, embedding_size], initializer=tf.truncated_normal_initializer(stddev=1/embedding_size))

for node_types in node_types_list:
    if is_attention(node_types):
        out_node_type = node_types[2]
        embedded_a = embedded_attentive_a
        out_embeddings = embedding_tables[out_node_type]["out"]
        #out_embeddings = attentive_out_embeddings#embedding_tables[out_node_type]["out"]

        embedded_b = tf.nn.embedding_lookup(out_embeddings, b_placeholder)
        embedded_neg_b = tf.nn.embedding_lookup(out_embeddings, neg_b_placeholder)
        # embedded_b = embedded_attentive_b
        # embedded_neg_b = embedded_attentive_neg_b

        meta_losses = build_line_losses(embedded_a, embedded_b, embedded_neg_b) + attention_func.l2_loss() * l2_coe
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(meta_losses, var_list=attention_func.variables)
        var_list = None if out_node_type == TYPE_LABEL else attention_func.variables
        optimizer = tf.train.AdamOptimizer(learning_rate=attention_decayed_learning_rate).minimize(meta_losses, var_list=var_list, global_step=global_step)
        # optimizer = tf.train.AdamOptimizer(learning_rate=5e-3).minimize(meta_losses, var_list=var_list)#, var_list=attention_func.variables + [label_out_embeddings]),
            # tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(meta_losses, var_list=embedding_vars)

        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(meta_losses)#, var_list=attention_func.variables)
        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(meta_losses)
    else:
        in_node_type, out_node_type = node_types
        in_embeddings = embedding_tables[in_node_type]["in"]
        out_embeddings = embedding_tables[out_node_type]["out"]
        meta_losses = build_line_losses_by_indices(in_embeddings, out_embeddings, a_placeholder, b_placeholder, neg_b_placeholder)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=embedding_decayed_learning_rate).minimize(meta_losses, global_step=global_step)

    mean_meta_losses.append(tf.reduce_mean(meta_losses))
    optimizers.append(optimizer)

# embedded_nodes = tf.nn.embedding_lookup(node_embeddings, a_placeholder)


def evaluate(embedded, sess):
    train_vec = sess.run(embedded, feed_dict={a_placeholder: train_node_indices, drop_rate_placeholder: 0})
    train_y = network.get_attr_list(TYPE_NODE, train_node_indices, "label")
    test_vec = sess.run(embedded, feed_dict={a_placeholder: test_node_indices, drop_rate_placeholder: 0})
    test_y = network.get_attr_list(TYPE_NODE, test_node_indices, "label")

    # clf = LogisticRegression()
    # clf.fit(train_vec, train_y)
    # print(clf.score(test_vec, test_y))
    evaluation(train_vec, test_vec, train_y, test_y)


batch_size = 2000

saver = tf.train.Saver(var_list=embedding_vars)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    if not pretrain:
        saver.restore(sess, model_save_path)
    for step in range(1000000):
        evaluated = False
        for meta_index, (node_types, mean_meta_loss, optimizer) in enumerate(zip(node_types_list, mean_meta_losses, optimizers)):
            if step < 1000:
                if is_attention(node_types) and node_types[-1] != TYPE_LABEL:
                    continue
            # if is_attention(node_types):
            #     if pretrain:
            #         continue
            # else:
            #     if not pretrain:
            #         continue

            if is_attention(node_types):
                a, b, neg_b = network.sample_triples([node_types[1], node_types[2]], batch_size)
                # if step < 1000:
                #     continue
            else:
                a, b, neg_b = network.sample_triples(node_types, batch_size)
            feed_dict = {
                a_placeholder: a,
                b_placeholder: b,
                neg_b_placeholder: neg_b,
                drop_rate_placeholder: 1e-1

            }
            _, loss = sess.run([optimizer, mean_meta_loss], feed_dict=feed_dict)
            # loss = sess.run([meta_loss], feed_dict=feed_dict)
            if step % 100 == 0:
                print("step = {}\t{}\tloss = {}".format(step, node_types, loss))

            if step % evaluation_interval == 0 and evaluated is False:
                evaluated = True
                if pretrain:
                    saver.save(sess, model_save_path)
                    print("save model")

                print("average:")
                evaluate(embedded_average_a, sess)
                print("attention:")
                evaluate(embedded_attentive_a, sess)
                # node_id_label_dict = read_node_id_label_dict(os.path.join(data_dir, "labels.txt"))

                # train_node_ids = network.get_ids(TYPE_NODE, train_node_indices)
                # test_node_ids = network.get_ids(TYPE_NODE, test_node_indices)

                # train_vec = sess.run(embedded_attentive_a, feed_dict={a_placeholder: train_node_indices, drop_rate_placeholder: 0})
                # train_y = network.get_attr_list(TYPE_NODE, train_node_indices, "label")
                # test_vec = sess.run(embedded_attentive_a, feed_dict={a_placeholder: test_node_indices, drop_rate_placeholder: 0})
                # test_y = network.get_attr_list(TYPE_NODE, test_node_indices, "label")
                #
                # # clf = LogisticRegression()
                # # clf.fit(train_vec, train_y)
                # # print(clf.score(test_vec, test_y))
                # evaluation(train_vec, test_vec, train_y, test_y)




