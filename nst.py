import keras as k
import tensorflow as tf
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import time

class NST:
    def __init__(self):
        self.max_dim = 512

        # Content layer where will pull our feature maps
        self.content_layers = ['block5_conv2']

        # Style layer we are interested in
        self.style_layers = ['block1_conv1',
                             'block2_conv1',
                             'block3_conv1',
                             'block4_conv1',
                             'block5_conv1'
                             ]

        self.num_content_layers = len(self.content_layers)
        self.num_style_layers = len(self.style_layers)

        # Weights for each class
        self.content_weight = 1e3
        self.style_weight = 1e-2

        pass

    def load_image(self, filepath):

        img = cv.imread(filepath)
        long = max(img.shape)
        scale = self.max_dim / long
        img = cv.resize(img, (int(round(img.shape[1] * scale)), int(round(img.shape[0] * scale))), cv.INTER_CUBIC)

        img_target = np.zeros([self.max_dim, self.max_dim, 3], np.float)
        img_target[0:img.shape[0],0:img.shape[1],:] = img/2.55
        img = np.expand_dims(img_target, axis=0)
        return img

    def preprocess_image(self, img):
        processed_img = k.applications.vgg19.preprocess_input(img)
        return processed_img

    def deprocess_img(self, processed_img):
        x = processed_img.copy()
        if len(x.shape) == 4:
            x = np.squeeze(x, 0)
        assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                                   "dimension [1, height, width, channel] or [height, width, channel]")
        if len(x.shape) != 3:
            raise ValueError("Invalid input to deprocessing image")

        # perform the inverse of the preprocessiing step
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
        x = x[:, :, ::-1]

        x = np.clip(x, 0, 255).astype('uint8')
        return x

    def show_img(img, title=None):
        # Remove the batch dimension
        out = np.squeeze(img, axis=0)
        # Normalize for display
        out = out.astype('uint8')
        plt.imshow(out)
        if title is not None:
            plt.title(title)
        plt.imshow(out)

    def create_model(self):
            """ Creates our model with access to intermediate layers.

            This function will load the VGG19 model and access the intermediate layers.
            These layers will then be used to create a new model that will take input image
            and return the outputs from these intermediate layers from the VGG model.

            Returns:
              returns a keras model that takes image inputs and outputs the style and
                content intermediate layers.
            """
            # Load our model. We load pretrained VGG, trained on imagenet data
            vgg = k.applications.VGG19(include_top=False, weights='imagenet', input_shape=[self.max_dim, self.max_dim, 3])
            vgg.trainable = False
            # Get output layers corresponding to style and content layers
            style_outputs = [vgg.get_layer(name).output for name in self.style_layers]
            content_outputs = [vgg.get_layer(name).output for name in self.content_layers]
            model_outputs = style_outputs + content_outputs

            # Build model
            self.model = k.models.Model(vgg.input, model_outputs)

    def content_loss(self, content, target):
        return tf.reduce_mean(tf.square(content - target))

    @staticmethod
    def gram_matrix(input_tensor):
        # We make the image channels first
        channels = int(input_tensor.shape[-1])
        a = tf.reshape(input_tensor, [-1, channels])
        n = tf.shape(a)[0]
        gram = tf.matmul(a, a, transpose_a=True)
        return gram / tf.cast(n, tf.float32)

    def style_loss(self, style, gram_target):
        """Expects two images of dimension h, w, c"""
        # height, width, num filters of each layer
        # We scale the loss at a given layer by the size of the feature map and the number of filters
        height, width, channels = style.get_shape().as_list()
        gram_style = self.gram_matrix(style)

        return tf.reduce_mean(tf.square(gram_style - gram_target))  # / (4. * (channels ** 2) * (width * height) ** 2)

    def get_features(self, content_path, style_path):
        """Helper function to compute our content and style feature representations.

         This function will simply load and preprocess both the content and style
         images from their path. Then it will feed them through the network to obtain
         the outputs of the intermediate layers.

         Arguments:
           model: The model that we are using.
           content_path: The path to the content image.
           style_path: The path to the style image

         Returns:
           returns the style features and the content features.
         """
        # Load our images in
        content_image = self.load_image(content_path)
        style_image = self.load_image(style_path)

        content_image = self.preprocess_image(content_image)
        style_image = self.preprocess_image(style_image)

        # batch compute content and style features
        style_outputs = self.model.predict(style_image)
        content_outputs = self.model.predict(content_image)

        # Get the style and content feature representations from our model
        style_features = [style_layer[0] for style_layer in style_outputs[:self.num_style_layers]]
        content_features = [content_layer[0] for content_layer in content_outputs[self.num_style_layers:]]

        return style_features, content_features

    def compute_loss(self, init_image, gram_style_features, content_features):
        """This function will compute the loss total loss.

        Arguments:
          model: The model that will give us access to the intermediate layers
          loss_weights: The weights of each contribution of each loss function.
            (style weight, content weight, and total variation weight)
          init_image: Our initial base image. This image is what we are updating with
            our optimization process. We apply the gradients wrt the loss we are
            calculating to this image.
          gram_style_features: Precomputed gram matrices corresponding to the
            defined style layers of interest.
          content_features: Precomputed outputs from defined content layers of
            interest.

        Returns:
          returns the total loss, style loss, content loss, and total variational loss
        """
        # style_weight, content_weight = loss_weights

        # Feed our init image through our model. This will give us the content and
        # style representations at our desired layers. Since we're using eager
        # our model is callable just like any other function!
        model_outputs = self.model.predict(init_image)

        style_output_features = model_outputs[:self.num_style_layers]
        content_output_features = model_outputs[self.num_style_layers:]

        style_score = 0
        content_score = 0

        # Accumulate style losses from all layers
        # Here, we equally weight each contribution of each loss layer
        weight_per_style_layer = 1.0 / float(self.num_style_layers)
        for target_style, comb_style in zip(gram_style_features, style_output_features):
            style_score += weight_per_style_layer * self.style_loss(comb_style[0], target_style)

        # Accumulate content losses from all layers
        weight_per_content_layer = 1.0 / float(self.num_content_layers)
        for target_content, comb_content in zip(content_features, content_output_features):
            content_score += weight_per_content_layer * self.content_loss(comb_content[0], target_content)

        style_score *= self.style_weight
        content_score *= self.content_weight

        # Get total loss
        loss = style_score + content_score
        return loss, style_score, content_score

    def compute_grads(self, cfg):
        with tf.GradientTape() as tape:
            all_loss = self.compute_loss(**cfg)
        # Compute gradients wrt input image
        total_loss = all_loss[0]
        return tape.gradient(total_loss, cfg['init_image']), all_loss

    def run_style_transfer(self, content_path, style_path, num_iterations=1000,):
        # We don't need to (or want to) train any layers of our model, so we set their
        # trainable to false.
        self.create_model()
        for layer in self.model.layers:
            layer.trainable = False

        # Get the style and content feature representations (from our specified intermediate layers)
        style_features, content_features = self.get_features(content_path, style_path)
        gram_style_features = [self.gram_matrix(style_feature) for style_feature in style_features]

        # Set initial image
        init_image = self.load_image(content_path)
        init_image = self.preprocess_image(init_image)
        init_image = tf.Variable(init_image, dtype=tf.float32)

        # Create our optimizer
        opt = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)

        # For displaying intermediate images
        iter_count = 1

        # Store our best result
        best_loss, best_img = float('inf'), None

        # Create a nice config
        loss_weights = (self.style_weight, self.content_weight)
        cfg = {
            'model': self.model,
            'loss_weights': loss_weights,
            'init_image': init_image,
            'gram_style_features': gram_style_features,
            'content_features': content_features
        }

        # For displaying
        num_rows = 2
        num_cols = 5
        display_interval = num_iterations / (num_rows * num_cols)
        start_time = time.time()
        global_start = time.time()

        norm_means = np.array([103.939, 116.779, 123.68])
        min_vals = -norm_means
        max_vals = 255 - norm_means

        imgs = []
        for i in range(num_iterations):
            grads, all_loss = self.compute_grads(cfg)
            loss, style_score, content_score = all_loss
            opt.apply_gradients([(grads, init_image)])
            clipped = tf.clip_by_value(init_image, min_vals, max_vals)
            init_image.assign(clipped)
            end_time = time.time()

            if loss < best_loss:
                # Update best loss and best image from total loss.
                best_loss = loss
                best_img = self.deprocess_img(init_image.numpy())

            if i % display_interval == 0:
                start_time = time.time()

                # Use the .numpy() method to get the concrete numpy array
                plot_img = init_image.numpy()
                plot_img = self.deprocess_img(plot_img)
                imgs.append(plot_img)
                # IPython.display.clear_output(wait=True)
                # IPython.display.display_png(Image.fromarray(plot_img))
                print('Iteration: {}'.format(i))
                print('Total loss: {:.4e}, '
                      'style loss: {:.4e}, '
                      'content loss: {:.4e}, '
                      'time: {:.4f}s'.format(loss, style_score, content_score, time.time() - start_time))
        print('Total time: {:.4f}s'.format(time.time() - global_start))
        # IPython.display.clear_output(wait=True)
        plt.figure(figsize=(14, 4))
        for i, img in enumerate(imgs):
            plt.subplot(num_rows, num_cols, i + 1)
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])

        return best_img, best_loss


nst = NST()

content_path = './image-076.png'
style_path = './AUDI-A4YH_20131010_082605_frame600.png'
nst.run_style_transfer(content_path, style_path, 100)
