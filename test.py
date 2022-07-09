# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

# #Prepare to feed input, i.e. feed_dict and placeholders
# w1 = tf.placeholder("float", name="w1")
# w2 = tf.placeholder("float", name="w2")
# b1= tf.Variable(2.0,name="bias")
# feed_dict ={w1:4,w2:8}

# #Define a test operation that we will restore
# w3 = tf.add(w1,w2)
# w4 = tf.multiply(w3,b1,name="op_to_restore")
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())

# #Create a saver object which will save all the variables
# saver = tf.train.Saver(save_relative_paths = True)

# #Run the operation by feeding input
# print (sess.run(w4,feed_dict))
# #Prints 24 which is sum of (w1+w2)*b1

# #Now, save the graph
# saver.save(sess, './my_test_model/model',global_step=1)

# import numpy as np
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('logs')
# r = 5
# for i in range(100):
#     writer.add_scalars('run_14h', {'xsinx':i*np.sin(i/r),
#                                    'xcosx':i*np.cos(i/r),
#                                    'tanx': np.tan(i/r)}, i)
#     writer.close()
#     # This call adds three values to the same scalar plot with the tag
#     # 'run_14h' in TensorBoard's scalar section.

# import numpy as np


# scores = np.arange(10)
# n_docs = scores.shape[0]
# exp_scores = np.exp(scores)
# exp_scores[np.array([], dtype=np.int32)] = 0
# probs = exp_scores / np.sum(exp_scores)
# safe_n = np.sum(probs > 10 ** (-4) / n_docs)
# safe_k = np.minimum(safe_n, 10)

# print(probs > 10 ** (-4) / n_docs)
# print(exp_scores)
# print(probs)
# print(safe_n)

import keyword
import torch
from torch.utils.tensorboard import SummaryWriter
meta = []
while len(meta)<100:
    meta = meta+keyword.kwlist # get some strings
meta = meta[:100]

for i, v in enumerate(meta):
    meta[i] = v+str(i)

label_img = torch.rand(100, 3, 10, 32)
for i in range(100):
    label_img[i]*=i/100.0


writer = SummaryWriter('logs')
writer.add_embedding(torch.randn(100, 5), metadata=meta, label_img=label_img)
writer.add_embedding(torch.randn(100, 5), label_img=label_img)
writer.add_embedding(torch.randn(100, 5), metadata=meta)
