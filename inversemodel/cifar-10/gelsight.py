import numpy as np
from skimage.transform import resize
import tensorflow as tf

PATH = '/home/ubuntu/Data/gelsight/'
NUM_CLASSES = 10
#NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 45000
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 49000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 4000

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  tf.summary.image('images', images)
  import IPython; IPython.embed()
  return images, tf.reshape(label_batch, [batch_size])

def load(batch_size=50):
    actions = np.load(PATH + 'rho_theta_one_hot.npy')
    images = np.load(PATH + 'inputs.npy').transpose(0,2,3,1)
    output_images = np.load(PATH + 'outputs.npy')
    diff_im_orig  =  output_images - images
    #diff_im_orig = tf.cast(diff_im_orig,tf.float32)
    #diff_im = tf.image.per_image_standardization(diff_im)
    diff_im = np.zeros((49000,32,32,3))
    for ii in range(49000):
       diff_im[ii,...] =  resize(diff_im_orig[ii,...],[32,32,3]) 
    diff_im = (diff_im - diff_im.mean(axis=0))/ diff_im.std(axis=0)
    diff_im = tf.cast(diff_im,tf.float32)
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print ('Filling queue with %d CIFAR images before starting to train. '
           'This will take a few minutes.' % min_queue_examples)

    #return diff_im, actions[0].argmax(axis=1)
    
    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(diff_im, actions[0].argmax(axis=1),
                                           min_queue_examples, batch_size,
                                           shuffle=True)
