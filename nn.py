import nn_utils as utils
import random
from scipy.misc import imread
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve
from tensorflow.python.keras.layers import Conv2D, Activation, Input, Add
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam

K = 256
GREYSCALE = 1


def read_image(filename, representation):
    """
    Reads image into either rgb or greyscale representation
    :param filename: file to open
    :param representation: 1 for greyscale, 2 for rgb
    :return: the representation (np.float64)
    """
    im = imread(filename)
    im_float = im.astype(np.float64) / K
    if representation == 1:
        return rgb2gray(im_float)
    if representation == 2:
        return im_float
    print('invalid representation, use 1 or 2')
    return


def load_dataset(filenames, batch_size, path_mapping, crop_size):
    s_cache = dict()
    t_cache = dict()
    while True:
        # randomly choose batch_size filenames
        names = list()
        for i in range(batch_size):
            names.append(random.choice(filenames))
        # create set of source images (from cache or directly)
        s_images = list()
        t_images = list()
        for name in names:
            if name not in s_cache:
                s_cache[name] = read_image(name, GREYSCALE)
                t_cache[name] = read_image(utils.v_to_t(name), GREYSCALE)
            s_images.append(s_cache[name])
            t_images.append(t_cache[name])
        # populate arrays
        source = np.ndarray((batch_size, crop_size[0], crop_size[1], 1))
        target = np.ndarray((batch_size, crop_size[0], crop_size[1], 1))
        i = 0
        for index in range(len(s_images)):
            # sample a large random crop
            h = random.randint(0, s_images[index].shape[0] - (crop_size[0] * 3))
            w = random.randint(0, s_images[index].shape[1] - (crop_size[1] * 3))
            large_s = s_images[index][h:h + (crop_size[0] * 3), w:w + (crop_size[1] * 3)]
            # apply corruption function
            large_t = t_images[index][h:h + (crop_size[0] * 3), w:w + (crop_size[1] * 3)]
            # take a random crop of requested size from original crop and corrupted crop
            h = random.randint(0, large_s.shape[0] - (crop_size[0]))
            w = random.randint(0, large_s.shape[1] - (crop_size[1]))
            target[i, :, :, :] = large_s[h:h + (crop_size[0]), w:w + (crop_size[1]), np.newaxis]
            source[i, :, :, :] = large_t[h:h + (crop_size[0]), w:w + (crop_size[1]), np.newaxis]
            i += 1
        yield (source - 0.5, target - 0.5)


def resblock(input_tensor, num_channels):
    out = Conv2D(num_channels, (3, 3), padding='same')(input_tensor)
    out = Activation('relu')(out)
    out = Conv2D(num_channels, (3, 3), padding='same')(out)
    out = Add()([out, input_tensor])
    out = Activation('relu')(out)
    return out


def build_nn_model(height, width, num_channels, num_res_blocks):
    input_tensor = Input(shape=(height, width, 1))
    out = Conv2D(num_channels, (3, 3), padding='same')(input_tensor)
    out = Activation('relu')(out)
    for i in range(num_res_blocks):
        out = resblock(out, num_channels)
    out = Conv2D(1, (3, 3), padding='same')(out)
    out = Add()([out, input_tensor])
    return Model(inputs=input_tensor, outputs=out)


def _split(filenames):
    """
    Performs 80-20 split
    :param filenames: List of input file names
    :return: A training subset of filenames and a validation subset of filenames
    """
    aux = list(filenames)
    random.shuffle(aux)
    cut = int(len(filenames) * 0.8)
    return aux[:cut], aux[cut:]


def train_model(model, images, corruption_func, batch_size, steps_per_epoch, num_epochs, num_valid_samples):
    training_set, validation_set = _split(images)
    crop_size = model.input_shape[1:3]
    training_gen = load_dataset(training_set, batch_size, corruption_func, crop_size)
    validation_gen = load_dataset(validation_set, batch_size, corruption_func, crop_size)
    model.compile(loss='mean_squared_error', optimizer=Adam(beta_2=0.9))
    model.fit_generator(training_gen, steps_per_epoch=steps_per_epoch, epochs=num_epochs,
                        validation_data=validation_gen, validation_steps=num_valid_samples//batch_size)


def generate_image(corrupted_image, base_model):
    shape = corrupted_image.shape
    input_tensor = Input((shape[0], shape[1], 1))
    b = base_model(input_tensor)
    new_model = Model(inputs=input_tensor, outputs=b)
    input_image = corrupted_image - 0.5
    output_image = new_model.predict(input_image[np.newaxis, :, :, np.newaxis])[0] + 0.5
    output_image = np.clip(output_image, 0, 1).astype(np.float64)
    return output_image[:, :, 0]


def learn_thermal_to_visual(num_res_blocks=5, quick_mode=False):
    paths = utils.visible_images()
    batch_size, steps_per_epoch, num_epochs, num_samples = 100, 100, 5, 1000
    if quick_mode:
        batch_size, steps_per_epoch, num_epochs, num_samples = 10, 3, 2, 30
    model = build_nn_model(24, 24, 48, num_res_blocks)
    train_model(model, paths, lambda x: utils.v_to_t(x), batch_size, steps_per_epoch, num_epochs,
                num_samples)
    return model


def test_thermal_to_visual():
    visual_path = utils.visible_images()[0]
    print(visual_path)
    thermal_path = utils.v_to_t(visual_path)
    print(thermal_path)
    visual_im = read_image(visual_path, GREYSCALE)
    thermal_im = read_image(thermal_path, GREYSCALE)
    plt.imshow(visual_im, cmap='gray')
    plt.title('Original')
    plt.show()
    plt.imshow(thermal_im, cmap='gray')
    plt.title('Thermal')
    plt.show()
    errors = list()
    for i in range(1, 6):
        print('Thermal to Visual, residual blocks: ' + str(i))
        model = learn_thermal_to_visual(i)
        errors.append(model.history.history['val_loss'][-1])
        fixed = generate_image(thermal_im, model)
        plt.imshow(fixed, cmap='gray')
        plt.title('Thermal to Visual, residual blocks: ' + str(i))
        plt.show()
    return errors


if __name__ == '__main__':
    x = range(1, 6)

    denoise_error = test_thermal_to_visual()
    print('Error array for denoise:')
    print(denoise_error)
    fig, ax = plt.subplots()
    ax.plot(x, denoise_error)
    ax.set(xlabel='Residual blocks', ylabel='MSE loss',
           title='Denoise performance')
    ax.grid()
    plt.show()
