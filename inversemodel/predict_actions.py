import sys
sys.path.append('/home/mudigonda/anaconda3/envs/gelsight/lib/python3.5/site-packages')
import numpy as np
import os
import tensorflow as tf
slim = tf.contrib.slim
import argparse as AP
import time
import alexnet_geurzhoy

CONFIG = tf.ConfigProto()
CONFIG.gpu_options.allow_growth = True


BATCH_SIZE = 50
GRAD_CLIP_NORM = 40
IM_SIZE = 100
ENCODING_SIZE = 100
FEAT_SIZE = 100
CHANNELS = 3
ACTION_DIMS = 2

def init_weights(name, shape):
    return tf.get_variable(name, shape=shape, initializer=tf.random_normal_initializer(0, 0.01))


def create_network(x,layers):
    input_size = x.shape
    output_size = ACTION_DIMS
    
    cur = x
    for a,b in enumerate(layers):
        W = init_weights("W" + str(a), [b[0],b[1]])
        B = init_weights("B" + str(a), b[1])
        cur = tf.nn.elu(tf.matmul(cur, W) +B)
    
    W = init_weights("W_act", [b[1],ACTION_DIMS])
    B = init_weights("B_act", [b[1],ACTION_DIMS])
    prediction = tf.matmul(cur,W) + B 
    return prediction


class GelSight():
    def __init__(self,name,action_lr=1e-4):
        print("GelSight Class")
        self.name = name
        print(self.name)
        self.get_batch = self.generate_toy_data
        self.image_PH = tf.placeholder(tf.float32, [None, IM_SIZE,IM_SIZE,CHANNELS], name = 'image_PH')
        self.goal_image_PH = tf.placeholder(tf.float32, [None,IM_SIZE,IM_SIZE,CHANNELS], name = 'goal_image_PH')
        self.gtAction_PH = tf.placeholder(tf.float32, [None,ACTION_DIMS])
        #get latent embeddings
        latent_image, latent_conv5_image = alexnet_geurzhoy.network(self.image_PH, trainable=True, num_outputs=ENCODING_SIZE)
        latent_goal_image, latent_conv5_goal_image = alexnet_geurzhoy.network(self.goal_image_PH, trainable=True, num_outputs=ENCODING_SIZE, reuse=True)

        # concatenate the latent representations and share information
        #features = tf.concat(1, [latent_image, latent_goal_image])
        features = tf.concat([latent_image, latent_goal_image],axis=1)

        with tf.variable_scope("concat_fc"):
            x = tf.nn.relu(features)
            x = slim.fully_connected(x, FEAT_SIZE, scope="concat_fc")

        #Create pred network
        pred_actions = create_network(x,[[100,200],[200,100]])
        #Loss
        pred_loss = tf.nn.l2_loss(pred_actions -self.gtAction_PH)
        tf.add_to_collection('pred_loss',pred_loss)
        inv_vars_no_alex = [v for v in tf.trainable_variables() if 'alexnet' not in v.name]
        print('Action prediction tensors consist {0} out of {1}'.format(len(inv_vars_no_alex), len(tf.trainable_variables())))
        action_optimizer = tf.train.AdamOptimizer(action_lr)
        action_grads, _ = zip(*action_optimizer.compute_gradients(pred_loss, inv_vars_no_alex))
        action_grads, _ = tf.clip_by_global_norm(action_grads, GRAD_CLIP_NORM)
        action_grads = zip(action_grads, inv_vars_no_alex)

        action_grads_full, _ = zip(*action_optimizer.compute_gradients(pred_loss, tf.trainable_variables()))
        action_grads_full, _ = tf.clip_by_global_norm(action_grads_full, GRAD_CLIP_NORM)
        action_grads_full = zip(action_grads_full, tf.trainable_variables())

        self.optimize_action_no_alex = action_optimizer.apply_gradients(action_grads)
        self.optimize_action_alex = action_optimizer.apply_gradients(action_grads_full)

        #Logging
        tf.summary.scalar('model/action_loss',pred_loss,collections=['train'])
        tf.summary.image('before',self.image_PH,max_outputs=5,collections=['train'])
        tf.summary.image('after',self.goal_image_PH,max_outputs=5,collections=['train'])
        
        self.train_summaries = tf.summary.merge_all('train')
        self.writer = tf.summary.FileWriter('./results/{0}/logs/{1}'.format(self.name, time.time()))

        self.saver = tf.train.Saver(max_to_keep=None)

        self.sess = tf.Session(config=CONFIG)
        self.sess.run(tf.global_variables_initializer())

        self.model_directory = './results/{0}/models/'.format(self.name)
        if not os.path.exists(self.model_directory):
            os.makedirs(self.model_directory)

    def generate_toy_data(self):
        images = np.random.randn(BATCH_SIZE,IM_SIZE,IM_SIZE,CHANNELS)
        goal_images = np.random.randn(BATCH_SIZE,IM_SIZE,IM_SIZE,CHANNELS)
        actions = np.random.randn(BATCH_SIZE,ACTION_DIMS)
        feed_dict = {
        self.goal_image_PH:goal_images,
        self.image_PH:images,
        self.gtAction_PH:actions}
        return feed_dict 


    def train(self,niters=1):
        print("Wil train for 1 steps")
        feed_data = self.get_batch()
        print("Feed data procured")
        outputs = self.sess.run(self.optimize_action_no_alex,feed_dict=feed_data)
        return
    

if __name__ == "__main__":
    parser = AP.ArgumentParser()
    parser.add_argument("--input",default=None,type=str,help="File name with data")
    parsed = parser.parse_args()

    if parsed.input is not None:
       data = np.load(parsed.input)

    GS = GelSight("Predictions")
    GS.train()
