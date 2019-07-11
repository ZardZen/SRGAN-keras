import os
import argparse
import logging
import numpy as np
from srgan import srgan
from keras.losses import mean_squared_error
from keras.optimizers import Adam
from keras.applications.vgg19 import preprocess_input
from keras.utils.data_utils import Sequence
from contextlib import contextmanager
from PIL import Image
from keras import backend as K
from keras.utils.data_utils import OrderedEnqueuer
import tensorflow as tf

@contextmanager
def concurrent_generator(sequence, num_workers=8, max_queue_size=32, use_multiprocessing=False):
    enqueuer = OrderedEnqueuer(sequence, use_multiprocessing=use_multiprocessing)
    try:
        enqueuer.start(workers=num_workers, max_queue_size=max_queue_size)
        yield enqueuer.get()
    finally:
        enqueuer.stop()

def init_session(gpu_memory_fraction):
    K.tensorflow_backend.set_session(tensorflow_session(gpu_memory_fraction=gpu_memory_fraction))
    
def tensorflow_session(gpu_memory_fraction):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
    return tf.Session(config=config)

logger = logging.getLogger(__name__)

DOWNGRADES = ['bicubic', 'bicubic_jpeg_75', 'bicubic_jpeg_90', 'unknown']

class DIV2KSequence(Sequence):
    def __init__(self,
                 path,
                 scale=2,
                 subset='train',
                 downgrade='bicubic',
                 image_ids=None,
                 random_rotate=True,
                 random_flip=True,
                 random_crop=True,
                 crop_size=96,
                 batch_size=16):
        """
        Sequence over a DIV2K subset.
        Reads DIV2K images that have been converted to numpy arrays with convert.py.
        :param path: path to DIV2K dataset with images stored as numpy arrays.
        :param scale: super resolution scale, either 2, 3 or 4.
        :param subset:  either 'train' or 'valid', referring to training and validation subset, respectively.
        :param downgrade: downgrade operator, see DOWNGRADES.
        :param image_ids: list of image ids to use from the specified subset. Default is None which means
                          all image ids from the specified subset.
        :param random_rotate: if True images are randomly rotated by 0, 90, 180 or 270 degrees.
        :param random_flip: if True images are randomly flipped horizontally.
        :param random_crop: if True images are randomly cropped.
        :param crop_size: size of crop window in HR image. Only used if random_crop=True.
        :param batch_size: size of generated batches.
        """


        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} doesn't exist")
        if scale not in [2, 3, 4]:
            raise ValueError('scale must be 2, 3 or 4')
        if subset not in ['train', 'valid']:
            raise ValueError("subset must be 'train' or 'valid'")
        if downgrade not in DOWNGRADES:
            raise ValueError(f"downgrade must be in {DOWNGRADES}")
        if not random_crop and batch_size != 1:
            raise ValueError('batch_size must be 1 if random_crop=False')

        self.path = path
        self.scale = scale
        self.subset = subset
        self.downgrade = downgrade

        if image_ids is None:
            if subset == 'train':
                self.image_ids = range(1, 801)
            else:
                self.image_ids = range(801, 901)
        else:
            self.image_ids = image_ids
        self.random_rotate = random_rotate
        self.random_flip = random_flip
        self.random_crop = random_crop
        self.crop_size = crop_size
        self.batch_size = batch_size

    def __getitem__(self, index):
        if self.batch_size == 1:
            return self._batch_1(self.image_ids[index])
        else:
            beg = index * self.batch_size
            end = (index + 1) * self.batch_size
            return self._batch_n(self.image_ids[beg:end])

    def __len__(self):
        return int(np.ceil(len(self.image_ids) / self.batch_size))

    def _batch_1(self, id):
        lr, hr = self._pair(id)

        return np.expand_dims(np.array(lr, dtype='uint8'), axis=0), \
               np.expand_dims(np.array(hr, dtype='uint8'), axis=0)

    def _batch_n(self, ids):
        lr_crop_size = self.crop_size // self.scale
        hr_crop_size = self.crop_size

        lr_batch = np.zeros((len(ids), lr_crop_size, lr_crop_size, 3), dtype='uint8')
        hr_batch = np.zeros((len(ids), hr_crop_size, hr_crop_size, 3), dtype='uint8')

        for i, id in enumerate(ids):
            lr, hr = self._pair(id)
            lr_batch[i] = lr
            hr_batch[i] = hr

        return lr_batch, hr_batch

    def _pair(self, id):
        lr_path = self._lr_image_path(id)
        hr_path = self._hr_image_path(id)

        lr = np.load(lr_path)
        hr = np.load(hr_path)

        if self.random_crop:
            lr, hr = _random_crop(lr, hr, self.crop_size, self.scale)
        if self.random_flip:
            lr, hr = _random_flip(lr, hr)
        if self.random_rotate:
            lr, hr = _random_rotate(lr, hr)

        return lr, hr

    def _hr_image_path(self, id):
        return os.path.join(self.path, f'DIV2K_{self.subset}_HR', f'{id:04}.npy')

    def _lr_image_path(self, id):
        return os.path.join(self.path, f'DIV2K_{self.subset}_LR_{self.downgrade}', f'X{self.scale}', f'{id:04}x{self.scale}.npy')

def cropped_sequence(path, scale, subset, downgrade, image_ids=None, batch_size=16):
    return DIV2KSequence(path=path, scale=scale, subset=subset, downgrade=downgrade, image_ids=image_ids,
                         batch_size=batch_size, crop_size=48 * scale)

def fullsize_sequence(path, scale, subset, downgrade, image_ids=None):
    return DIV2KSequence(path=path, scale=scale, subset=subset, downgrade=downgrade, image_ids=image_ids,
                         batch_size=1, random_rotate=False, random_flip=False, random_crop=False)

def _random_crop(lr_img, hr_img, hr_crop_size, scale):
    lr_crop_size = hr_crop_size // scale

    lr_w = np.random.randint(lr_img.shape[1] - lr_crop_size + 1)
    lr_h = np.random.randint(lr_img.shape[0] - lr_crop_size + 1)

    hr_w = lr_w * scale
    hr_h = lr_h * scale

    lr_img_cropped = lr_img[lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size]
    hr_img_cropped = hr_img[hr_h:hr_h + hr_crop_size, hr_w:hr_w + hr_crop_size]

    return lr_img_cropped, hr_img_cropped

def _random_flip(lr_img, hr_img):
    if np.random.rand() > 0.5:
        return np.fliplr(lr_img), np.fliplr(hr_img)
    else:
        return lr_img, hr_img

def _random_rotate(lr_img, hr_img):
    k = np.random.choice(range(4))
    return np.rot90(lr_img, k), np.rot90(hr_img, k)

def learning_rate(step_size, decay, verbose=1):
    def schedule(epoch, lr):
        if epoch > 0 and epoch % step_size == 0:
            return lr * decay
        else:
            return lr
    return LearningRateScheduler(schedule, verbose=verbose)

def create_train_workspace(path):
    train_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_dir = os.path.join(path, train_dir)
    models_dir = os.path.join(train_dir, 'models')
    os.makedirs(train_dir, exist_ok=True)
    os.mkdir(models_dir)
    return train_dir, models_dir

def write_args(path, args):
    with open(os.path.join(path, 'args.txt'), 'w') as f:
        for k, v in sorted(args.__dict__.items()):
            f.write(f'{k}={v}\n')
            
def content_loss(hr, sr):
    vgg = srgan.vgg_54()
    sr = preprocess_input(sr)
    hr = preprocess_input(hr)
    sr_features = vgg(sr)
    hr_features = vgg(hr)
    return mean_squared_error(hr_features, sr_features)

def main(args):
    train_dir, models_dir = create_train_workspace(args.outdir)
    losses_file = os.path.join(train_dir, 'losses.csv')
    write_args(train_dir, args)
    logger.info('Training workspace is %s', train_dir)

    sequence = DIV2KSequence(args.dataset,
                             scale=args.scale,
                             subset='train',
                             downgrade=args.downgrade,
                             image_ids=range(1,801),
                             batch_size=args.batch_size,
                             crop_size=96)


#     if args.generator == 'edsr-gen':
#         generator = edsr.edsr_generator(args.scale, args.num_filters, args.num_res_blocks)
#     else:
#         generator = srgan.generator(args.num_filters, args.num_res_blocks)
    generator = srgan.generator(args.num_filters, args.num_res_blocks)

    if args.pretrained_model:
        generator.load_weights(args.pretrained_model)


    generator_optimizer = Adam(lr=args.generator_learning_rate)

    discriminator = srgan.discriminator()
    discriminator_optimizer = Adam(lr=args.discriminator_learning_rate)
    discriminator.compile(loss='binary_crossentropy',
                          optimizer=discriminator_optimizer,
                          metrics=[])


    gan = srgan.srgan(generator, discriminator)
    gan.compile(loss=[content_loss, 'binary_crossentropy'],
                loss_weights=[0.006, 0.001],
                optimizer=generator_optimizer,
                metrics=[])



    generator_lr_scheduler = learning_rate(step_size=args.learning_rate_step_size, decay=args.learning_rate_decay, verbose=0)
    generator_lr_scheduler.set_model(gan)

    discriminator_lr_scheduler = learning_rate(step_size=args.learning_rate_step_size, decay=args.learning_rate_decay, verbose=0)
    discriminator_lr_scheduler.set_model(discriminator)

    with open(losses_file, 'w') as f:
        f.write('Epoch,Discriminator loss,Generator loss\n')

    with concurrent_generator(sequence, num_workers=1) as gen:
        for epoch in range(args.epochs):
            generator_lr_scheduler.on_epoch_begin(epoch)
            discriminator_lr_scheduler.on_epoch_begin(epoch)
            d_losses = []
            g_losses_0 = []
            g_losses_1 = []
            g_losses_2 = []

            for iteration in range(args.iterations_per_epoch):
                # ----------------------
                #  Train Discriminator
                # ----------------------

                lr, hr = next(gen)
                sr = generator.predict(lr)

                hr_labels = np.ones(args.batch_size) + args.label_noise * np.random.random(args.batch_size)
                sr_labels = np.zeros(args.batch_size) + args.label_noise * np.random.random(args.batch_size)

                hr_loss = discriminator.train_on_batch(hr, hr_labels)
                sr_loss = discriminator.train_on_batch(sr, sr_labels)

                d_losses.append((hr_loss + sr_loss) / 2)

                # ------------------
                #  Train Generator
                # ------------------

                lr, hr = next(gen)

                labels = np.ones(args.batch_size)

                perceptual_loss = gan.train_on_batch(lr, [hr, labels])

                g_losses_0.append(perceptual_loss[0])
                g_losses_1.append(perceptual_loss[1])
                g_losses_2.append(perceptual_loss[2])



                print(f'[{epoch:03d}-{iteration:03d}] '

                      f'discriminator loss = {np.mean(d_losses[-50:]):.3f} '

                      f'generator loss = {np.mean(g_losses_0[-50:]):.3f} ('

                      f'mse = {np.mean(g_losses_1[-50:]):.3f} '

                      f'bxe = {np.mean(g_losses_2[-50:]):.3f})')



            generator_lr_scheduler.on_epoch_end(epoch)
            discriminator_lr_scheduler.on_epoch_end(epoch)

            with open(losses_file, 'a') as f:
                f.write(f'{epoch},{np.mean(d_losses)},{np.mean(g_losses_0)}\n')

            model_path = os.path.join(models_dir, f'generator-epoch-{epoch:03d}.h5')
            print('Saving model', model_path)
            generator.save(model_path)





def parser():
    parser = argparse.ArgumentParser(description='GAN training with custom generator')
    parser.add_argument('-o', '--outdir', type=str, default='./output',
                        help='output directory')

    # --------------
    #  Dataset
    # --------------

    parser.add_argument('-d', '--dataset', type=str, default='./DIV2K_BIN',
                        help='path to DIV2K dataset with images stored as numpy arrays')
    parser.add_argument('-s', '--scale', type=int, default=4, choices=[4],
                        help='super-resolution scale')
    parser.add_argument('--downgrade', type=str, default='bicubic', choices=DOWNGRADES,
                        help='downgrade operation')

    # --------------
    #  Model
    # --------------

#     parser.add_argument('-g', '--generator', type=str, default='edsr-gen', choices=['edsr-gen', 'sr-resnet'],
#                         help='generator model name')
    parser.add_argument('--num-filters', type=int, default=64,
                        help='number of filters in generator')
    parser.add_argument('--num-res-blocks', type=int, default=16,
                        help='number of residual blocks in generator')
    parser.add_argument('--pretrained-model', type=str,
                        help='path to pre-trained generator model')

    # --------------
    #  Training
    # --------------

    parser.add_argument('--epochs', type=int, default=150,
                        help='number of epochs to train')
    parser.add_argument('--iterations-per-epoch', type=int, default=1000,
                        help='number of update iterations per epoch')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='mini-batch size for training')
    parser.add_argument('--generator-learning-rate', type=float, default=1e-4,
                        help='generator learning rate')
    parser.add_argument('--discriminator-learning-rate', type=float, default=1e-4,
                        help='discriminator learning rate')
    parser.add_argument('--learning-rate-step-size', type=int, default=100,
                        help='learning rate step size in epochs')
    parser.add_argument('--learning-rate-decay', type=float, default=0.1,
                        help='learning rate decay at each step')
    parser.add_argument('--label-noise', type=float, default=0.05,
                        help='amount of noise added to labels for discriminator training')
    # --------------
    #  Hardware
    # --------------

    parser.add_argument('--gpu-memory-fraction', type=float, default=0.8,
                        help='fraction of GPU memory to allocate')

    return parser





if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    args = parser().parse_args()
    init_session(args.gpu_memory_fraction)
    main(args)
