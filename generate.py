import tensorlayer as tl
import tensorflow as tf
from pretrain.model import SRGAN_g, SRGAN_d, Vgg19_simple_api
import time
from PIL import Image
import numpy as np



def generate(img_name):
    # imgs = tl.files.load_file_list(path="bot_image/from_usr/", regx='.*.png', printable=False)
    # print(img_name)
    image = tl.vis.read_images([img_name], path="bot_image/from_usr/", n_threads=32)[0]
    image = (image / 127.5) - 1  # rescale to ［－1, 1]
    # size = image[0].shape
    t_image = tf.placeholder('float32', [1, None, None, 3], name='input_image')
    srgan_g = "generator/g_srgan.npz"

    net_g = SRGAN_g(t_image, is_train=False, reuse=False)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name = srgan_g, network=net_g)
    start_time = time.time()
    out = sess.run(net_g.outputs, {t_image: [image]})
    total_time = time.time() - start_time
    tl.vis.save_image(out[0], f'bot_image/to_usr/{img_name}'.replace(".png","")+'_new.png')
    return total_time

def main():
    generate("Yeamao_BQADBQADhgADHgr5VDva3BemqyHBFgQ_image.png")

if __name__ == "__main__":
    main()