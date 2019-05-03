from keras.preprocessing.image import load_img, save_img, img_to_array
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
import argparse

from keras.applications import vgg19
from keras import backend as K

"""*********************Input ***************************"""
base_image_path = 'content.png'
style_reference_image_path = 'style.png'
result_prefix = './'

iterations = 10
img_nrows = 400
"""*********************Input ***************************"""


# dimensions of the generated picture.
width, height = load_img(base_image_path).size
img_ncols = int(width * img_nrows / height)

class ImageProcessing:
    @staticmethod
    def preprocess_image(image_path):
        img = load_img(image_path, target_size=(img_nrows, img_ncols))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = vgg19.preprocess_input(img)
        return img

    @staticmethod
    def deprocess_image(x):
        x = x.reshape((img_nrows, img_ncols, 3))
        # Remove zero-center by mean pixel
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
        # 'BGR'->'RGB'
        x = x[:, :, ::-1]
        x = np.clip(x, 0, 255).astype('uint8')
        return x


class LossCalculation:
    total_variation_weight = 1.0
    style_weight = 1.0
    content_weight = 0.25

    def gram_matrix(self, x):
        assert K.ndim(x) == 3
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
        gram = K.dot(features, K.transpose(features))
        return gram

    def style_loss(self, style, generated):
        assert K.ndim(style) == 3
        assert K.ndim(generated) == 3
        S = self.gram_matrix(style)
        C = self.gram_matrix(generated)
        channels = 3
        size = img_nrows * img_ncols
        return K.sum(K.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

    def content_loss(self, base, generated):
        return K.sum(K.square(generated - base))

    def total_variation_loss(self, x):
        # designed to keep the generated image locally coherent
        assert K.ndim(x) == 4

        a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
        b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
        return K.sum(K.pow(a + b, 1.25))


class NST:
    def __init__(self):
        self.loss_value = None
        self.grads_values = None
        pass

    def input(self, base_image_path, style_reference_image_path):
        # get tensor representations of our images
        self.base_image = K.variable(ImageProcessing.preprocess_image(base_image_path))
        self.style_reference_image = K.variable(ImageProcessing.preprocess_image(style_reference_image_path))

        # this will contain our generated image
        self.generated_output_image = K.placeholder((1, img_nrows, img_ncols, 3))

        # combine the 3 images into a single Keras tensor
        self.input_tensor = K.concatenate([self.base_image,
                                      self.style_reference_image,
                                      self.generated_output_image], axis=0)

    def load_network(self):
        model = vgg19.VGG19(input_tensor=self.input_tensor,
                            weights='imagenet', include_top=False)
        print('Model loaded.')

        # get the symbolic outputs of each "key" layer (we gave them unique names).
        self.outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
        pass

    def getting_loss_function(self, loss_calculator = LossCalculation()):

        # combine these loss functions into a single scalar
        loss = K.variable(0.0)
        layer_features = self.outputs_dict['block5_conv2']
        base_image_features = layer_features[0, :, :, :]
        generated_features = layer_features[2, :, :, :]
        loss += loss_calculator.content_weight \
                * loss_calculator.content_loss(base_image_features,
                                               generated_features)

        feature_layers = ['block1_conv1', 'block2_conv1',
                          'block3_conv1', 'block4_conv1',
                          'block5_conv1']
        for layer_name in feature_layers:
            layer_features = self.outputs_dict[layer_name]
            style_reference_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            sl = loss_calculator.style_loss(style_reference_features, combination_features)
            loss += (loss_calculator.style_weight / len(feature_layers)) * sl
        loss += loss_calculator.total_variation_weight\
                * loss_calculator.total_variation_loss(self.generated_output_image)

        # get the gradients of the generated image wrt the loss
        grads = K.gradients(loss, self.generated_output_image)

        outputs = [loss]
        if isinstance(grads, (list, tuple)):
            outputs += grads
        else:
            outputs.append(grads)

        self.loss_function = K.function([self.generated_output_image], outputs)

        return self.loss_function

    def train(self, iterations):

        # run scipy-based optimization (L-BFGS) over the pixels of the generated image
        # so as to minimize the neural style loss
        x = ImageProcessing.preprocess_image(base_image_path)

        for i in range(iterations):
            print('Start of iteration', i)
            start_time = time.time()
            x, min_val, info = fmin_l_bfgs_b(self.loss, x.flatten(),
                                             fprime=self.grads, maxfun=20)
            print('Current loss value:', min_val)
            # save current generated image
            img = ImageProcessing.deprocess_image(x.copy())
            fname = result_prefix + '_at_iteration_%d.png' % i
            save_img(fname, img)
            end_time = time.time()
            print('Image saved as', fname)
            print('Iteration %d completed in %ds' % (i, end_time - start_time))

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = self.eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

    def eval_loss_and_grads(self, x):

        x = x.reshape((1, img_nrows, img_ncols, 3))
        outs = self.loss_function([x])
        loss_value = outs[0]
        if len(outs[1:]) == 1:
            grad_values = outs[1].flatten().astype('float64')
        else:
            grad_values = np.array(outs[1:]).flatten().astype('float64')
        return loss_value, grad_values
if __name__ == '__main__':
    nst = NST()

    nst.input(base_image_path, style_reference_image_path)
    nst.load_network()

    nst.getting_loss_function()

    nst.train(iterations)





