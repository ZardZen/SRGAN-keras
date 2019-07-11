import numpy as np
from keras.optimizers import Adam, SGD
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import argparse
from srgan import srgan, generator, discriminator
import os
from keras.utils import multi_gpu_model
import tensorflow as tf
from keras import backend as K
from keras.losses import mean_absolute_error, mean_squared_error
from keras.applications.vgg19 import VGG19

from keras.models import Model

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class VGG_LOSS(object):
    def __init__(self, image_shape):  
        self.image_shape = image_shape
    # computes VGG loss or content loss
    def vgg_loss(self, y_true, y_pred): 
        vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.image_shape)
        vgg19.trainable = False
        # Make trainable as False
        for l in vgg19.layers:
            l.trainable = False
        model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
        model.trainable = False  
        return K.mean(K.square(model(y_true) - model(y_pred)))  

def get_optimizer():
    adam = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    return adam

# def plot_generated_images(output_dir, epoch, generator, x_test_hr, x_test_lr , dim=(1, 3), figsize=(15, 5)):

#     examples = x_test_hr.shape[0]
#     print(examples)
#     value = randint(0, examples)
#     image_batch_hr = denormalize(x_test_hr)
#     image_batch_lr = x_test_lr
#     gen_img = generator.predict(image_batch_lr)
#     generated_image = denormalize(gen_img)
#     image_batch_lr = denormalize(image_batch_lr)

#     plt.figure(figsize=figsize) 
#     plt.subplot(dim[0], dim[1], 1)
#     plt.imshow(image_batch_lr[value], interpolation='nearest')
#     plt.axis('off')      
#     plt.subplot(dim[0], dim[1], 2)
#     plt.imshow(generated_image[value], interpolation='nearest')
#     plt.axis('off')  
#     plt.subplot(dim[0], dim[1], 3)
#     plt.imshow(image_batch_hr[value], interpolation='nearest')
#     plt.axis('off')   
#     plt.tight_layout()
#     plt.savefig(output_dir + 'generated_image_%d.png' % epoch)
#     plt.close()

    #plt.show()
def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser(description='Keras Training')
    # ========= paths for training
    ap.add_argument("-npath", "--npy_path", default="data/", required=False,
                    help="path to npy. files to train")
#     ap.add_argument("-mpath", "--model_path", default="model_save/", required=False,
#                     help="path to save the output model")
#     ap.add_argument("-lpath", "--log_path", default="log/", required=False,
#                     help="path to save the 'log' files")
#     ap.add_argument("-name","--model_name", default="edsr.h5", required=False,
#                     help="output of model name")
    # ========= parameters for training
#     ap.add_argument("-p", "--pretrain", default=0, required=False, type=int,
#                     help="load pre-train model or not")

#     ap.add_argument('-bs', '--batch_size', default=2, type=int,
#                     help='batch size')
#     ap.add_argument('-ep', '--epoch', default=30, type=int,
#                     help='epoch')
    ap.add_argument('-m', '--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
    
    ap.add_argument('-i', '--input_dir', action='store', dest='input_dir', default='./data/' ,
                    help='Path for input images')                 
    ap.add_argument('-o', '--output_dir', action='store', dest='output_dir', default='./output/' ,
                    help='Path for Output images')
    ap.add_argument('-mpath', '--model_save_dir', action='store', dest='model_save_dir', default='./model/' ,
                    help='Path for model')
    ap.add_argument('-b', '--batch_size', action='store', dest='batch_size', default=8,
                    help='Batch Size', type=int)                   
    ap.add_argument('-e', '--epochs', action='store', dest='epochs', default=10 ,
                    help='number of iteratios for trainig', type=int)                   
#     ap.add_argument('-n', '--number_of_images', action='store', dest='number_of_images', default=1000 ,
#                     help='Number of Images', type= int)                    
#     ap.add_argument('-r', '--train_test_ratio', action='store', dest='train_test_ratio', default=0.8 ,
#                     help='Ratio of train and test Images', type=float)
    
    args = vars(ap.parse_args())
    return args


def train(args):
    scale = 4
    image_shape = (384,384,3)
    X_train = np.load(args["npy_path"] + 'lr.npy')
    #X_val = np.load(args["npy_path"] + 'X_val.npy')
    y_train = np.load(args["npy_path"] + 'hr.npy')
    #y_val = np.load(args["npy_path"] + 'y_val.npy')

    loss = VGG_LOSS(image_shape) 
    batch_count = int(y_train.shape[0] / batch_size)
    
#     lr_decay = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, verbose=1, min_lr=1e-5)
#     checkpointer = ModelCheckpoint(args["model_path"] + args["model_name"], verbose=1, save_best_only=True)
#     tensorboard = TensorBoard(log_dir=args["log_path"])
#     callback_list = [lr_decay, checkpointer, tensorboard]

    #optimizer = SGD(lr=1e-5, momentum=args["momentum"], nesterov=False)
    optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999)
    
    G = generator(num_filters=64, num_res_blocks=16)
    D = discriminator(num_filters=64)
    G.compile(loss=loss.vgg_loss, optimizer=optimizer)
    D.compile(loss="binary_crossentropy", optimizer=optimizer)
    
    gan = srgan(G, D)
    gan.compile(loss=[loss.vgg_loss, "binary_crossentropy"],
                loss_weights=[1., 1e-3],
                optimizer=optimizer)
    #gan.summary()
    loss_file = open(model_save_dir + 'losses.txt' , 'w+')

    loss_file.close()
    for e in range(1, epochs+1):
        print ('-'*15, 'Epoch %d' % e, '-'*15)
        for _ in tqdm(range(batch_count)):
       
            rand_nums = np.random.randint(0, y_train.shape[0], size=batch_size)
            image_batch_hr = y_train[rand_nums]
            image_batch_lr = X_train[rand_nums]
            generated_images_sr = generator.predict(image_batch_lr)

            real_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
            fake_data_Y = np.random.random_sample(batch_size)*0.2
            
            D.trainable = True
            
            d_loss_real = D.train_on_batch(image_batch_hr, real_data_Y)
            d_loss_fake = D.train_on_batch(generated_images_sr, fake_data_Y)
            discriminator_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

            rand_nums = np.random.randint(0, y_train.shape[0], size=batch_size)
            image_batch_hr = y_train[rand_nums]
            image_batch_lr = X_train[rand_nums]

            gan_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
            D.trainable = False
            gan_loss = gan.train_on_batch(image_batch_lr, [image_batch_hr,gan_Y])       
           

        print("discriminator_loss : %f" % discriminator_loss)
        print("gan_loss :", gan_loss)
        gan_loss = str(gan_loss)

        loss_file = open(model_save_dir + 'losses.txt' , 'a')
        loss_file.write('epoch%d : gan_loss = %s ; discriminator_loss = %f\n' %(e, gan_loss, discriminator_loss) )
        loss_file.close()

#         if e == 1 or e % 5 == 0:
#             Utils.plot_generated_images(output_dir, e, G, x_test_hr, x_test_lr)
        if e % 5 == 0:
            G.save(model_save_dir + 'gen_model%d.h5' % e)
            D.save(model_save_dir + 'dis_model%d.h5' % e)          
   
if __name__ == "__main__":
    args = args_parse()
    train(args)
