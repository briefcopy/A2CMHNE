# coding=utf-8

import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression

from gca.config import *
from gca.data.data_processing import load_data, read_node_id_label_dict
from gca.data.network import MetaNetwork
from gca.evaluation import evaluation
from gca.model.module import create_embedding_tables, eval_embeddings, build_line_losses, build_line_losses_by_indices

network, train_node_indices, test_node_indices  = load_data(data_dir, training_rate=training_rate)
print("building embedding tables")
embedding_tables = create_embedding_tables(network, [TYPE_NODE, TYPE_CONTENT, TYPE_WORD], embedding_size)
# embedding_tables = create_embedding_tables(network, [TYPE_NODE], embedding_size)
print(embedding_tables)

node_embeddings = embedding_tables[TYPE_NODE]["in"]
content_embeddings = embedding_tables[TYPE_CONTENT]["in"]


train_node_ids = network.get_ids(TYPE_NODE, train_node_indices)
test_node_ids = network.get_ids(TYPE_NODE, test_node_indices)

attention_node_indices_placeholder = tf.placeholder(tf.int32, shape=[None])

# attentive_embedded = (tf.nn.embedding_lookup(node_embeddings, attention_node_indices_placeholder) \
#                      + tf.nn.embedding_lookup(content_embeddings, attention_node_indices_placeholder))/2


a_placeholder = tf.placeholder(tf.int32, shape=[None])
b_placeholder = tf.placeholder(tf.int32, shape=[None])
neg_b_placeholder = tf.placeholder(tf.int32, shape=[None])


node_types_list = [
    [TYPE_NODE, TYPE_NODE],
    # [TYPE_CONTENT, TYPE_WORD]
]




mean_meta_losses = []
optimizers = []
for node_types in node_types_list:
    in_node_type, out_node_type = node_types
    in_embeddings = embedding_tables[in_node_type]["in"]
    out_embeddings = embedding_tables[out_node_type]["out"]
    meta_losses = build_line_losses_by_indices(in_embeddings, out_embeddings, a_placeholder, b_placeholder, neg_b_placeholder)


    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(meta_losses)
    mean_meta_losses.append(tf.reduce_mean(meta_losses))
    optimizers.append(optimizer)

# embedded_nodes = tf.nn.embedding_lookup(node_embeddings, a_placeholder)

batch_size = 2000

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(1000000):
        for meta_index, (node_types, mean_meta_loss, optimizer) in enumerate(zip(node_types_list, mean_meta_losses, optimizers)):
            a, b, neg_b = network.sample_triples(node_types, batch_size)
            feed_dict = {
                a_placeholder: a,
                b_placeholder: b,
                neg_b_placeholder: neg_b

            }
            _, loss = sess.run([optimizer, mean_meta_loss], feed_dict=feed_dict)
            # loss = sess.run([meta_loss], feed_dict=feed_dict)

            if step % 100 == 0:
                print("step = {}\t{}\tloss = {}".format(step, node_types, loss))

            if step % 2000 == 0 and meta_index == 0:
                node_id_label_dict = read_node_id_label_dict(os.path.join(data_dir, "labels.txt"))

                train_node_ids = network.get_ids(TYPE_NODE, train_node_indices)
                test_node_ids = network.get_ids(TYPE_NODE, test_node_indices)

                train_vec = eval_embeddings(node_embeddings, train_node_indices, sess)# + eval_embeddings(content_embeddings, train_node_indices, sess)
                train_y = network.get_attr_list(TYPE_NODE, train_node_indices, "label")
                test_vec = eval_embeddings(node_embeddings, test_node_indices, sess)# + eval_embeddings(content_embeddings, test_node_indices, sess)
                test_y = network.get_attr_list(TYPE_NODE, test_node_indices, "label")

                # clf = LogisticRegression()
                # clf.fit(train_vec, train_y)
                # print(clf.score(test_vec, test_y))
                evaluation(train_vec, test_vec, train_y, test_y)



    # output_feed_dict = {
    #     attention_node_indices_placeholder: train_node_indices
    # }
    # print(sess.run(attentive_embedded, feed_dict=output_feed_dict))


# print(train_node_ids)
# print(test_node_ids)


