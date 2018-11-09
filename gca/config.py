# coding=utf-8
import os
data_dir = "/home/hujun/mygit/2018_tomm_a2cmhne/data/M10"
tmp_data_dir = "/home/hujun/data/a2cmhne/"
pretrain = True
if pretrain:
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

model_save_path = os.path.join(tmp_data_dir, "model/model.ckpt")
pickle_data_path = os.path.join(tmp_data_dir, "data.p")

TYPE_NODE = "NODE"
TYPE_WORD = "WORD"
TYPE_CONTENT = "CONTENT"
TYPE_ATTENTION = "ATTENTION"
TYPE_LABEL = "LABEL"

embedding_size = 300
# learning_rate = 2.5e-2#2.5e-2

learning_rate = 1e-1#1e-1
attention_learning_rate = 1e-3#1e-3#1e-3
l2_coe = 0#1e-3#1e-3#1e-3#2e-3
dropout_rate = 0#1e-1#1e-1

training_rate = 0.7

evaluation_interval = 500