import sys
sys.path.append('/home/mudigonda/anaconda3/envs/gelsight/lib/python3.5/site-packages')
import numpy as np
import os
import tensorflow as tf
slim = tf.contrib.slim
import argparse as AP
import time
#import alexnet_randinit
import cifar10
from skimage.transform import resize

CONFIG = tf.ConfigProto()
CONFIG.gpu_options.allow_growth = True


BATCH_SIZE = 100
GRAD_CLIP_NORM = 40
IM_SIZE0 = 100 
#IM_SIZE = IM_SIZE0 
IM_SIZE = 32 
ENCODING_SIZE = 100
FEAT_SIZE = 200
CHANNELS = 3
ACTION_DIMS = 2
TrainSplit = 48000 
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
    def __init__(self,name,DEBUG=False,unfreeze_time=30000, autoencode=False,
        action_lr=1e-4, deconv_lr=1e-3, fwd_consist=False, baseline_reg=False, softmaxBackprop=True,
        gtAction=False,discreteAction=False,diffIm=False,optimizer='Adam'):
        print("GelSight Class")
        self.unfreeze_time = unfreeze_time
        self.autoencode = autoencode
        self.gtAction = gtAction
        self.discreteAction = discreteAction
        self.diffIm = diffIm
        self.optimizer = optimizer
        self.name = '{0}_{1}_{2}_{3}_{4}_{5}_{6}'.format(name, 'fwdconsist' + str(fwd_consist), 
            'autoencode' + str(autoencode),
            'discrAct' + str(discreteAction), 
            'diffIm' + str(diffIm),
            'action_lr' + str(action_lr),
            'optimizer' + str(optimizer) )
        self.fwd_consist = fwd_consist
        self.start = 0
        print(self.name)
        if DEBUG:
          print("Debug mode")
          self.get_batch = self.generate_toy_data
        else:
          print("Real Data")
          self.path = '/home/ubuntu/Data/gelsight/'
          self.normalize = True 
          print("Data loading")
          self.load_data()
          print("Ordering Data")
          self.order_data()
          self.get_batch = self.generate_gelsight_data
        self.image_PH = tf.placeholder(tf.float32, [None, IM_SIZE,IM_SIZE,CHANNELS], name = 'image_PH')
        if self.diffIm == False:
            self.goal_image_PH = tf.placeholder(tf.float32, [None,IM_SIZE,IM_SIZE,CHANNELS], name = 'goal_image_PH')
        if self.discreteAction:
            self.gtTheta_PH = tf.placeholder(tf.int32,[None,BINS])
            self.gtRho_PH = tf.placeholder(tf.int32,[None,BINS])
        else:
            self.gtAction_PH = tf.placeholder(tf.float32, [None,ACTION_DIMS])
        self.autoencode_PH = tf.placeholder(tf.bool)

        #get latent embeddings
        cifar10_logits = cifar10.network(self.image_PH)
        if self.diffIm == False:
          print("Does not work for a pair of images")
          return


        #Loss
        if self.discreteAction:
          rho_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=cifar10_logits[:,:BINS],labels=self.gtRho_PH))
          theta_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=cifar10_logits[:,BINS:],labels=self.gtTheta_PH))
          pred_loss = rho_loss + theta_loss	
        else:
          print("Does not work for continuous actions")
          return 
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

        #Eval
        self.pred_actions = cifar10_logits 
        self.pred_loss = pred_loss
        #import IPython; IPython.embed()

        #################################
        # FORWARD CONSISTENCY
        #################################
        if self.fwd_consist:
            print("Forward Consistency Enabled")
            with tf.variable_scope('fwd_consist'):
                '''
                if softmaxBackprop:
                    location_pred = tf.nn.softmax(location_pred)
                    theta_pred = tf.nn.softmax(theta_pred)
                    length_pred = tf.nn.softmax(length_pred)
                '''

                # baseline regularization => gradients flow only to alexnet, not action pred
                if baseline_reg:
                    print('baseline')
                    action_embed = tf.concat(1, [self.location_ph, self.theta_ph, self.length_ph])
                else:
                    print("Gradient flowing through action prediction")
                    # fwd_consist => gradients flow through action prediction
                    latent_conv5_image = tf.stop_gradient(latent_conv5_image)
                    '''
                    action_embed = tf.cond(self.gtAction_ph,
                        lambda: tf.concat(1, [self.location_ph, self.theta_ph, self.length_ph]),
                        lambda: tf.concat(1, [location_pred, theta_pred, length_pred]))
                    '''
                    action_embed = pred_actions

                action_embed = slim.fully_connected(action_embed, 363)
                action_embed = tf.reshape(action_embed, [-1, 11, 11, 3])
                #action_embed = slim.fully_connected(action_embed, 125)
                #action_embed = tf.reshape(action_embed, [-1, 5, 5, 3])
                # concat along depth
                fwd_features = tf.concat([latent_conv5_image, action_embed],axis=3)
                # deconvolution
                batch_size = tf.shape(fwd_features)[0]

                wt1 = tf.Variable(tf.truncated_normal([5, 5, 64, 259], stddev=0.1))
                deconv1 = tf.nn.conv2d_transpose(fwd_features, wt1, [batch_size, 22, 22, 64], [1, 2, 2, 1])
                deconv1 = leaky_relu(deconv1, 0.2)
                wt2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
                deconv2 = tf.nn.conv2d_transpose(deconv1, wt2, [batch_size, 44, 44, 32], [1, 2, 2, 1])
                deconv2 = leaky_relu(deconv2, 0.2)
                wt3 = tf.Variable(tf.truncated_normal([5, 5, 3, 32], stddev=0.1))
                deconv3 = tf.nn.conv2d_transpose(deconv2, wt3, [batch_size, 88, 88, 3], [1, 2, 2, 1])
                deconv3 = tf.nn.tanh(deconv3)
                # loss from upsampled deconvolution and goal image
                upsampled_deconv_img = tf.image.resize_images(deconv3, [200, 200])
                #upsampled_deconv_img = tf.image.resize_images(deconv3, [100, 100])
                tf.add_to_collection('upsampled_deconv_img', upsampled_deconv_img)

                # image inputs are -255 to 255 ??? for some reason
                # whether to autoencode or not

                normalized_goal_img = tf.cond(self.autoencode_PH, lambda: self.image_PH / 255.0, lambda: self.goal_image_PH / 255.0)

                # just to visualize
                deconv_log_img = (upsampled_deconv_img + 1.0) * 127.5

                # variables of only forward model
                fwd_vars = [v for v in tf.trainable_variables() if 'fwd_consist' in v.name]
                print('Forward consistency tensors consist {0} out of {1}'.format(len(fwd_vars), len(tf.trainable_variables())))

                fwd_consist_loss = tf.reduce_mean(tf.abs(upsampled_deconv_img - normalized_goal_img))
                if self.optimizer == 'Adam':
                  deconv_optimizer = tf.train.AdamOptimizer(deconv_lr)
                else:
                  deconv_optimizer = tf.train.MomentumOptimizer(learning_rate=deconv_lr,momentum=0.9)

                fwd_consist_grads, _ = zip(*deconv_optimizer.compute_gradients(fwd_consist_loss, fwd_vars))
                fwd_consist_grads, _ = tf.clip_by_global_norm(fwd_consist_grads, GRAD_CLIP_NORM)
                fwd_consist_grads = zip(fwd_consist_grads, fwd_vars)

                fwd_consist_grads_full, _ = zip(*deconv_optimizer.compute_gradients(fwd_consist_loss, tf.trainable_variables()))
                fwd_consist_grads_full, _ = tf.clip_by_global_norm(fwd_consist_grads_full, GRAD_CLIP_NORM)
                fwd_consist_grads_full = zip(fwd_consist_grads_full, tf.trainable_variables())

                self.optimize_fwd_freeze = deconv_optimizer.apply_gradients(fwd_consist_grads)
                list_fwd_consist_grads_full = list(fwd_consist_grads_full)
                list_action_grads_full = list(action_grads_full)
                with tf.control_dependencies([list_fwd_consist_grads_full[0][0][0], list_action_grads_full[0][0][0]]):
                    self.optimize_fwd_full = deconv_optimizer.apply_gradients(list_fwd_consist_grads_full)
                    self.optimize_action_full = action_optimizer.apply_gradients(list_action_grads_full)

        self.optimize_action_no_alex = action_optimizer.apply_gradients(action_grads)
        if self.fwd_consist:
          self.optimize_action_alex = action_optimizer.apply_gradients(list_action_grads_full)
        else:
          self.optimize_action_alex = action_optimizer.apply_gradients(action_grads_full)

        #Logging
        tf.summary.scalar('model/action_loss',pred_loss,collections=['train'])
        if self.discreteAction:
          tf.summary.scalar('model/theta_loss',theta_loss,collections=['train'])
          tf.summary.scalar('model/rho_loss',rho_loss,collections=['train'])
        tf.summary.image('before',self.image_PH/255.,max_outputs=5,collections=['train'])
        if self.diffIm == False:
          tf.summary.image('after',self.goal_image_PH/255.,max_outputs=5,collections=['train'])
        if self.fwd_consist:
            tf.summary.scalar('model/fwd_consist_loss', fwd_consist_loss, collections=['train'])
            tf.summary.image('upsampled_deconv_image', deconv_log_img, max_outputs=5, collections=['train'])

        for var in tf.trainable_variables():
            tf.summary.histogram(var.name,var,collections=['train'])
        if self.fwd_consist:
          for grad,var in list_action_grads_full:
              tf.summary.histogram(var.name + '/gradient_action', grad,collections=['train'])
        else:
          for grad,var in action_grads_full:
              tf.summary.histogram(var.name + '/gradient_action', grad,collections=['train'])
        if self.fwd_consist:
          for grad,var in list_fwd_consist_grads_full:
              tf.summary.histogram(var.name + '/gradient_fwd_consist', grad,collections=['train'])
        self.train_summaries = tf.summary.merge_all('train')
        self.writer = tf.summary.FileWriter('./results/{0}/logs/{1}'.format(self.name, time.time()))

        self.saver = tf.train.Saver(max_to_keep=5)

        self.sess = tf.Session(config=CONFIG)
        self.sess.run(tf.global_variables_initializer())

        self.model_directory = './results/{0}/models/'.format(self.name)
        if not os.path.exists(self.model_directory):
            os.makedirs(self.model_directory)

    def generate_toy_data(self,isTraining=True):
        images = np.random.randn(BATCH_SIZE,IM_SIZE,IM_SIZE,CHANNELS)
        if self.discreteAction:
          from sklearn.preprocessing import MultiLabelBinarizer
          mlb = MultiLabelBinarizer(classes = np.arange(BINS))
          theta = np.random.randint(0,BINS,(BATCH_SIZE,1))
          rho = np.random.randint(0,BINS,(BATCH_SIZE,1))
          theta_one_hot = mlb.fit_transform(theta)
          rho_one_hot = mlb.fit_transform(rho)
          feed_dict = {
          self.image_PH:images,
          self.gtTheta_PH:theta_one_hot,
          self.gtRho_PH:rho_one_hot,
          self.autoencode_PH:False}
        else:
          print("Need discrete Action")
          return
        return feed_dict 

    def load_data(self):
        self.images = np.load(self.path + 'inputs.npy').transpose(0,2,3,1)
        self.mean = self.images.mean() #assumes the means are same across the channels which is true for sim data
        self.std = self.images.std() #assumes the stds are same across the channels which is true for sim data
        if self.discreteAction:
          self.actions = np.load(self.path + 'rho_theta_one_hot.npy')
        else:
          self.actions = np.load(self.path + 'input_actions.npy')
        if self.normalize:
          print("Mean subtraction, Std Dev for image")
          self.images = (self.images -  self.mean)/self.std
        self.goal_images = np.load(self.path + 'outputs.npy')
        if self.normalize:
          print("Mean subtraction, Std Dev for target image")
          self.goal_images = (self.goal_images - self.mean)/self.std
        if self.normalize and self.discreteAction==False:
          print("Normalize actions")
          self.actions = (self.actions - self.actions.mean(axis=0)) / self.actions.std(axis=0)
        return

    def order_data(self,lag=1):
        #init
        episode_len = 50-lag
        inputs = np.zeros((980*episode_len,IM_SIZE0,IM_SIZE0,CHANNELS))
        outputs = np.zeros((980*episode_len,IM_SIZE0,IM_SIZE0,CHANNELS))
        #Loop from 0 to IM_SIZE0 (number of episodes). 980 because the job died?
        for ii in range(980):
        #For each input up to end-lag, take the output by shifting by lag
            inputs[(ii*episode_len):((ii+1) * episode_len)] = self.images[(ii*episode_len):((ii+1)*episode_len)]
            outputs[((ii*episode_len) + lag): ((ii+1) *episode_len)] = self.goal_images[((ii*episode_len)+lag) : ((ii+1)* episode_len)]
        self.images = inputs
        self.goal_images = outputs
        return


    def generate_gelsight_data(self,isTraining=True):
        if isTraining:
          idx = np.random.randint(0,TrainSplit,BATCH_SIZE)
        else:
          #idx = np.random.randint(TrainSplit,self.images.shape[0],self.images.shape[0]-TrainSplit)
          idx = np.arange(self.images.shape[0]-100,self.images.shape[0])
        #resizing to address the 200 200 issue
        tmp_im = np.zeros((len(idx),IM_SIZE,IM_SIZE,CHANNELS))
        tmp_goal_im = np.zeros((len(idx),IM_SIZE,IM_SIZE,CHANNELS))
        for ii in range(len(idx)):
            tmp_im[ii,...] = resize(self.images[idx[ii],...],[IM_SIZE,IM_SIZE])  
            tmp_goal_im[ii,...] = resize(self.goal_images[idx[ii],...], [IM_SIZE, IM_SIZE])
        '''
        tmp_im = self.images[idx,...]
        tmp_goal_im = self.goal_images[idx,...]
        '''
        if self.diffIm: #No Goal Im
          if self.discreteAction:
            feed_dict = {
            self.image_PH:tmp_im - tmp_goal_im,
            self.gtTheta_PH:self.actions[1][idx],#theta is the second array
            self.gtRho_PH:self.actions[0][idx],#rho is the first array
            self.autoencode_PH:False}
          else:
            print("Need discrete ACtion")
            return
        return feed_dict 

    def train(self,niters=1):
        print("Will train for {} steps".format(niters))
        for ii in range(self.start, niters):
            print(ii)
            feed_dict = self.get_batch(isTraining=True)

            ops_to_run = []
            ops_to_run.append(self.pred_loss)
            '''		
            if ii < self.unfreeze_time:
                ops_to_run.append(self.optimize_action_no_alex)
                if self.fwd_consist:
                    ops_to_run.append(self.optimize_fwd_freeze)
                    if self.autoencode and i < self.unfreeze_time * (2/3):
                        feed_dict[self.autoencode_ph] = True
                if self.gtAction:
                    feed_dict[self.gtAction_ph] = True
            else:
                if self.fwd_consist:
                    ops_to_run.append(self.optimize_fwd_full)
                    ops_to_run.append(self.optimize_action_full)
                else:
                    ops_to_run.append(self.optimize_action_alex)
            '''
            #We are not dealing with fwd freeze loss. ###BEWARE#### NO FREEZE or AUTOENCODE
            if self.fwd_consist:
                ops_to_run.append(self.optimize_fwd_full)
                ops_to_run.append(self.optimize_action_full)
            else:
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
    GS = GelSight(name=parsed.name,fwd_consist = parsed.fwd,DEBUG=parsed.debug,discreteAction=parsed.discrete,action_lr=parsed.action_lr,diffIm=parsed.diffIm,optimizer=parsed.optimizer)
    GS.train(parsed.niters)
