from network import Generator, Discriminator
import Utils_model, Utils
from Utils_model import VGG_LOSS

from keras.models import Model
from keras.layers import Input
from tqdm import tqdm
import numpy as np
import argparse

np.random.seed(10)
downscale_factor = 4
# 圖片的大小在這個改變
image_shape = (384,384,3)


def get_gan_network(discriminator, shape, generator, optimizer, vgg_loss):
    discriminator.trainable = False
    gan_input = Input(shape=shape)
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=[x,gan_output])
    gan.compile(loss=[vgg_loss, "binary_crossentropy"],
                loss_weights=[1., 1e-3],
                optimizer=optimizer)
    return gan


def train(epochs, batch_size, input_dir, output_dir, model_save_dir, number_of_images, train_test_ratio):
    # train_test_ratio 基本上是控制train data 的數量
    x_train_lr, x_train_hr, x_test_lr, x_test_hr = Utils.load_training_data(input_dir, '.png', number_of_images=number_of_images, train_test_ratio = train_test_ratio) 
    loss = VGG_LOSS(image_shape)  
    
    batch_count = int(x_train_hr.shape[0] / batch_size)
    shape = (image_shape[0]//downscale_factor, image_shape[1]//downscale_factor, image_shape[2])
    
    generator = Generator(shape).generator()
    discriminator = Discriminator(image_shape).discriminator()

    optimizer = Utils_model.get_optimizer()
    generator.compile(loss=loss.vgg_loss, optimizer=optimizer)
    discriminator.compile(loss="binary_crossentropy", optimizer=optimizer)
    
    gan = get_gan_network(discriminator, shape, generator, optimizer, loss.vgg_loss)
    
    loss_file = open(model_save_dir + 'losses.txt' , 'w+')
    loss_file.close()

    for e in range(1, epochs+1):
        print ('-'*15, 'Epoch %d' % e, '-'*15)
        for _ in tqdm(range(batch_count)):
            
            rand_nums = np.random.randint(0, x_train_hr.shape[0], size=batch_size)
            
            image_batch_hr = x_train_hr[rand_nums]
            image_batch_lr = x_train_lr[rand_nums]
            
            generated_images_sr = generator.predict(image_batch_lr)

            real_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
            fake_data_Y = np.random.random_sample(batch_size)*0.2
            
            discriminator.trainable = True
            
            d_loss_real = discriminator.train_on_batch(image_batch_hr, real_data_Y)
            d_loss_fake = discriminator.train_on_batch(generated_images_sr, fake_data_Y)
            discriminator_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
            
            rand_nums = np.random.randint(0, x_train_hr.shape[0], size=batch_size)
            image_batch_hr = x_train_hr[rand_nums]
            image_batch_lr = x_train_lr[rand_nums]

            gan_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
            discriminator.trainable = False
            gan_loss = gan.train_on_batch(image_batch_lr, [image_batch_hr,gan_Y])
            
            
        print("discriminator loss : %f" % discriminator_loss)
        print("gan_loss :", gan_loss)
        gan_loss = str(gan_loss)
        
        loss_file = open(model_save_dir + 'losses.txt' , 'a')
        loss_file.write('epoch%d : gan_loss = %s ; discriminator loss = %f\n' %(e, gan_loss, discriminator_loss) )
        loss_file.close()

        if e == 1 or e % 5 == 0:
            Utils.plot_generated_images(output_dir, e, generator, x_test_hr, x_test_lr)
        if e % 500 == 0:
            generator.save(model_save_dir + 'gen_model%d.h5' % e)
            discriminator.save(model_save_dir + 'dis_model%d.h5' % e)

train(500, 32, "dataset", "output", "model", 100, 0.8)