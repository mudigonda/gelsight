import sys
sys.path.append('/home/mudigonda/anaconda3/envs/gelsight/lib/python3.5/site-packages')
import deepdish as dd
import numpy as np
import os
import tensorflow as tf
slim = tf.contrib.slim
import argparse as AP
import time
import alexnet_randinit
from skimage.transform import resize
from scipy.ndimage import shift

CONFIG = tf.ConfigProto()
CONFIG.gpu_options.allow_growth = True

BATCH_SIZE = 10
GRAD_CLIP_NORM = 40
IM_SIZE0 = 100 
#IM_SIZE = IM_SIZE0 
IM_SIZE = [48,64]
ENCODING_SIZE = 100
FEAT_SIZE = 200
CHANNELS = 3
ACTION_DIMS = 3
TrainSplit = 5500 
ValSplit = 500
Episode_Len = 18
BINS = 10

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
    
    '''
    W = init_weights("W_act", [b[1],ACTION_DIMS])
    B = init_weights("B_act", [b[1],ACTION_DIMS])
    prediction = tf.matmul(cur,W) + B  
    return prediction
    '''
    return cur

def leaky_relu(x, alpha):
    return tf.maximum(x, alpha * x)

class GelSight():
    def __init__(self,name,DEBUG=False,
        action_lr=1e-4, deconv_lr=1e-3,
        gtAction=False,discreteAction=False,diffIm=False,optimizer='Adam',saved_model= None):
        print("GelSight Class")
        self.gtAction = gtAction
        self.discreteAction = discreteAction
        self.diffIm = diffIm
        self.optimizer = optimizer
        self.saved_model = saved_model
        self.name = '{0}_{1}_{2}_{3}_{4}'.format(name, 
            'discrAct' + str(discreteAction), 
            'diffIm' + str(diffIm),
            'action_lr' + str(action_lr),
            'optimizer' + str(optimizer) )
        self.start = 0
        print(self.name)
        if DEBUG:
          print("Debug mode")
          self.get_batch = self.generate_toy_data
        else:
          print("Real Data")
          self.path = '/home/ubuntu/Data/hd5/'
          self.normalize = True 
          print("Data loading")
          self.load_data()
          self.get_batch = self.generate_gelsight_data
        self.image_PH = tf.placeholder(tf.float32, [None, IM_SIZE[0],IM_SIZE[1],CHANNELS], name = 'image_PH')
        if self.diffIm == False:
            self.goal_image_PH = tf.placeholder(tf.float32, [None,IM_SIZE,IM_SIZE,CHANNELS], name = 'goal_image_PH')
        if self.discreteAction:
            self.gtTheta_PH = tf.placeholder(tf.int32,[None,BINS])
            self.gtRho_PH = tf.placeholder(tf.int32,[None,BINS])
        else:
            self.gtAction_PH = tf.placeholder(tf.float32, [None,ACTION_DIMS])


        #get latent embeddings
        latent_image, latent_conv5_image = alexnet_randinit.network(self.image_PH, trainable=True, num_outputs=ENCODING_SIZE)
        if self.diffIm == False:
          latent_goal_image, latent_conv5_goal_image = alexnet_randinit.network(self.goal_image_PH, trainable=True, num_outputs=ENCODING_SIZE, reuse=True)
          # concatenate the latent representations and share information
          features = tf.concat([latent_image, latent_goal_image],axis=1)
        else:
          features = latent_image
        tf.summary.histogram('features',features,collections=['train'])
        with tf.variable_scope("concat_fc"):
            x = tf.nn.relu(features)
            x = slim.fully_connected(x, FEAT_SIZE, scope="concat_fc")

        #Create pred network
        if self.discreteAction:
          pred_actions = create_network(x,[[FEAT_SIZE,200],[200,100],[100,BINS*2]]) #For theta and rho we create Bin sized vector
        else:
          pred_actions = create_network(x,[[FEAT_SIZE,200],[200,100],[100,ACTION_DIMS]])
        #Loss
        if self.discreteAction:
          rho_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred_actions[:,:BINS],labels=self.gtRho_PH))
          theta_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred_actions[:,BINS:],labels=self.gtTheta_PH))
          pred_loss = rho_loss + theta_loss	
        else:
          #pred_loss = tf.nn.l2_loss(pred_actions -self.gtAction_PH)/(2*BATCH_SIZE)
          pred_loss = tf.reduce_mean(tf.reduce_sum((pred_actions -self.gtAction_PH)**2,axis=1))
        tf.add_to_collection('pred_loss',pred_loss)
        if self.discreteAction:	
          tf.add_to_collection('theta_loss',theta_loss)
          tf.add_to_collection('rho_loss',rho_loss)
        inv_vars_no_alex = [v for v in tf.trainable_variables() if 'alexnet' not in v.name]
        print('Action prediction tensors consist {0} out of {1}'.format(len(inv_vars_no_alex), len(tf.trainable_variables())))
        if self.optimizer == 'Adam':
          action_optimizer = tf.train.AdamOptimizer(action_lr)
        else:
          action_optimizer = tf.train.MomentumOptimizer(learning_rate=action_lr,momentum=0.9)
        action_grads, _ = zip(*action_optimizer.compute_gradients(pred_loss, inv_vars_no_alex))
        action_grads, _ = tf.clip_by_global_norm(action_grads, GRAD_CLIP_NORM)
        action_grads = zip(action_grads, inv_vars_no_alex)

        action_grads_full, _ = zip(*action_optimizer.compute_gradients(pred_loss, tf.trainable_variables()))
        action_grads_full, _ = tf.clip_by_global_norm(action_grads_full, GRAD_CLIP_NORM)
        action_grads_full = zip(action_grads_full, tf.trainable_variables())
        self.optimize_action_no_alex = action_optimizer.apply_gradients(action_grads)
        self.optimize_action_alex = action_optimizer.apply_gradients(action_grads_full)
        #Eval
        self.pred_actions = pred_actions
        self.pred_loss = pred_loss
        tf.add_to_collection("pred_actions",pred_actions)
        tf.add_to_collection("pred_loss",pred_loss)
        tf.add_to_collection("image",self.image_PH)


        #Logging
        tf.summary.scalar('model/action_loss',pred_loss,collections=['train'])
        if self.discreteAction:
          tf.summary.scalar('model/theta_loss',theta_loss,collections=['train'])
          tf.summary.scalar('model/rho_loss',rho_loss,collections=['train'])
        tf.summary.image('before',self.image_PH/255.,max_outputs=5,collections=['train'])
        if self.diffIm == False:
          tf.summary.image('after',self.goal_image_PH/255.,max_outputs=5,collections=['train'])

        for var in tf.trainable_variables():
            tf.summary.histogram(var.name,var,collections=['train'])
        for grad,var in action_grads_full:
            tf.summary.histogram(var.name + '/gradient_action', grad,collections=['train'])
        self.train_summaries = tf.summary.merge_all('train')
        self.writer = tf.summary.FileWriter('./results/{0}/logs/{1}'.format(self.name, time.time()))

        self.saver = tf.train.Saver(max_to_keep=5)

        self.sess = tf.Session(config=CONFIG)
        self.sess.run(tf.global_variables_initializer())
        if self.saved_model:
           print("Sess state is {}".format(self.sess._closed))
           #with self.sess as sess:
           self.saver.restore(self.sess,self.saved_model)
           print("Sess state is {}".format(self.sess._closed))

        self.model_directory = './results/{0}/models/'.format(self.name)
        if not os.path.exists(self.model_directory):
            os.makedirs(self.model_directory)
        print("Sess state is {}".format(self.sess._closed))

    def generate_toy_data(self,isTraining=True):
        images = np.random.randn(BATCH_SIZE,IM_SIZE0,IM_SIZE0,CHANNELS)
        goal_images = np.random.randn(BATCH_SIZE,IM_SIZE0,IM_SIZE0,CHANNELS)
        if self.discreteAction:
          from sklearn.preprocessing import MultiLabelBinarizer
          mlb = MultiLabelBinarizer(classes = np.arange(BINS))
          theta = np.random.randint(0,BINS,(BATCH_SIZE,1))
          rho = np.random.randint(0,BINS,(BATCH_SIZE,1))
          theta_one_hot = mlb.fit_transform(theta)
          rho_one_hot = mlb.fit_transform(rho)
          feed_dict = {
          self.goal_image_PH:goal_images,
          self.image_PH:images,
          self.gtTheta_PH:theta_one_hot,
          self.gtRho_PH:rho_one_hot,
          self.autoencode_PH:False}
        else:
          actions = np.ones((BATCH_SIZE,ACTION_DIMS))
          feed_dict = {
          self.goal_image_PH:goal_images,
          self.image_PH:images,
          self.gtAction_PH:actions,
          self.autoencode_PH:False}
        return feed_dict 

    def load_data(self):
        fnames = os.listdir(self.path)
        self.images = np.zeros((TrainSplit+ValSplit,Episode_Len,IM_SIZE[0],IM_SIZE[1],CHANNELS))
        self.actions = np.zeros((TrainSplit+ValSplit,Episode_Len,CHANNELS)) 
        for ii  in range(TrainSplit+ValSplit):
          data  = dd.io.load(self.path + fnames[ii],'/')
          if np.mod(ii,10) == 0:
            print("The idx is {}".format(ii))
          for jj in range(18):
            self.images[ii,jj,...] = data['img_'+str(jj)]
            self.actions[ii,jj,...] = data['action_'+str(jj)]
        print("We are taking the mean based on the first 50 samples")
        self.mean = self.images[:50,...].mean() #assumes the means are same across the channels which is true for sim data
        self.std = self.images[:50].std() #assumes the stds are same across the channels which is true for sim data
        if self.normalize:
          print("Mean subtraction, Std Dev for image")
          self.images = (self.images -  self.mean)/self.std
        if self.normalize and self.discreteAction==False:
          print("Normalize actions")
          self.actions = (self.actions - self.actions.mean(axis=0)) / self.actions.std(axis=0)
        return


    def generate_gelsight_data(self,isTraining=True):
        if isTraining:
          idx1 = np.random.randint(0,TrainSplit,BATCH_SIZE)
          idx2 = np.random.randint(0,Episode_Len-1,BATCH_SIZE)

        else:
          idx1 = np.random.randint(TrainSplit,self.images.shape[0],ValSplit)
          idx2 = np.random.randint(0,Episode_Len-1,ValSplit)
        #resizing to address the 200 200 issue
        tmp_im = np.zeros((len(idx1),IM_SIZE[0],IM_SIZE[1],CHANNELS))
        tmp_action = np.zeros((len(idx2),ACTION_DIMS))
        for ii in np.arange(len(idx1)):
            tmp_im[ii,...] = self.images[idx1[ii],idx2[ii]+1,...] -  self.images[idx1[ii],idx2[ii],...]
            tmp_action[ii,...] = self.actions[idx1[ii],idx2[ii],...]

        #tmp_goal_im = np.zeros((len(idx),IM_SIZE[0],IM_SIZE[1],CHANNELS))
        if self.diffIm: #No Goal Im
          if self.discreteAction:
            feed_dict = {
            self.image_PH:tmp_im,
            self.gtTheta_PH:self.actions[1][idx],#theta is the second array
            self.gtRho_PH:self.actions[0][idx],#rho is the first array
            }
          else:
            feed_dict = {
            self.image_PH:tmp_im ,
            self.gtAction_PH:tmp_action,
            }
        else:
            raise NotImplementedError("Aaaah we did not implent this")
        return feed_dict 


    def train(self,niters=1):
        print("Will train for {} steps".format(niters))
        for ii in range(self.start, niters):
            print(ii)
            feed_dict = self.get_batch(isTraining=True)

            ops_to_run = []
            ops_to_run.append(self.pred_loss)
            ops_to_run.append(self.optimize_action_alex)

            ops_to_run.append(self.train_summaries)
            op_results = self.sess.run(ops_to_run, feed_dict=feed_dict)
            train_summaries = op_results[-1]
            if ii%100 ==0:
                print("L2 Norm of Train loss is {} \n".format(op_results[0]))

            if ii % 100 == 0:
                self.writer.add_summary(train_summaries, ii)

            # validate on 1000 images
            # split into batches of 100 because of memory issues
            if ii % 100 == 0:
                self.saver.save(self.sess, self.model_directory + 'inverse', global_step=ii)
                print('Saved at timestep {0}'.format(ii))

                feed_dict = self.get_batch(isTraining=False)
                pred_actions, pred_loss = self.sess.run([self.pred_actions,self.pred_loss],feed_dict=feed_dict)
                print("L2 Norm of Validation loss is {} \n".format(pred_loss))

                summaries = tf.Summary(value=[tf.Summary.Value(tag='val/pred_loss', simple_value=pred_loss)])
                self.writer.add_summary(summaries, ii)

            self.writer.flush()
        import IPython; IPython.embed()
        return

def str2bool(varName):
    if varName == "False":
        return False
    else:    
        return True



if __name__ == "__main__":
    parser = AP.ArgumentParser()
    parser.add_argument("--input",default=None,type=str,help="File name with data")
    parser.add_argument("--niters",default=1,type=int,help="Number of training iterations.Default is 1.")
    parser.add_argument("--action_lr",default=1e-4,type=float,help="Learning rate for the action network. Default is 1e-4")
    parser.add_argument("--fwd",default="False",type=str,help="Boolean flag that trains for forward consistency. Default is False")
    parser.add_argument("--name",default="Predictions",type=str,help="Expt Name")
    parser.add_argument("--debug",default="False",type=str,help="Boolean flag that sets Debug. Default is False")
    parser.add_argument("--diffIm",default="False",type=str,help="Boolean flag that sets if we should use inputs as diff images. Default is False")
    parser.add_argument("--discrete",default="False",type=str,help="Boolean flag that sets Discrete Actions. Default is False")
    parser.add_argument("--optimizer",default="Adam",type=str,help="Optimizer when specified by Adam runs Adam the else condition runs GD")
    parsed = parser.parse_args()
    parsed.fwd = str2bool(parsed.fwd)
    parsed.debug = str2bool(parsed.debug)
    parsed.diffIm = str2bool(parsed.diffIm)
    parsed.discrete = str2bool(parsed.discrete)
    print("Job Parameters are")
    print(parsed)
    GS = GelSight(name=parsed.name,DEBUG=parsed.debug,discreteAction=parsed.discrete,action_lr=parsed.action_lr,diffIm=parsed.diffIm,optimizer=parsed.optimizer)
    GS.train(parsed.niters)
