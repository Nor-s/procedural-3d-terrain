from os.path import exists
#DataTools.py
""" Module for the data loading pipeline for the model to train """
IN_CHANNEL = 4


def get_data_loader(dataset, batch_size, num_workers):
    """
    generate the data_loader from the given dataset
    :param dataset: dataset for training (Should be a PyTorch dataset)
                    Make sure every item is an Image
    :param batch_size: batch size of the data
    :param num_workers: num of parallel readers
    :return: dl => dataloader for the dataset
    """
    from torch.utils.data import DataLoader

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    return dl
#Losses.py
""" Module implementing various loss functions """

import torch as th


# =============================================================
# Interface for the losses
# =============================================================

class GANLoss:
    """ Base class for all losses

        @args:
            dis: Discriminator used for calculating the loss
                 Note this must be a part of the GAN framework
    """

    def __init__(self, dis):
        self.dis = dis

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        """
        calculate the discriminator loss using the following data
        :param real_samps: batch of real samples
        :param fake_samps: batch of generated (fake) samples
        :param height: current height at which training is going on
        :param alpha: current value of the fader alpha
        :return: loss => calculated loss Tensor
        """
        raise NotImplementedError("dis_loss method has not been implemented")

    def gen_loss(self, real_samps, fake_samps, height, alpha):
        """
        calculate the generator loss
        :param real_samps: batch of real samples
        :param fake_samps: batch of generated (fake) samples
        :param height: current height at which training is going on
        :param alpha: current value of the fader alpha
        :return: loss => calculated loss Tensor
        """
        raise NotImplementedError("gen_loss method has not been implemented")

class ConditionalGANLoss:
    """ Base class for all conditional losses """

    def __init__(self, dis):
        self.dis = dis

    def dis_loss(self, real_samps, fake_samps, labels, height, alpha):
        raise NotImplementedError("dis_loss method has not been implemented")

    def gen_loss(self, real_samps, fake_samps, labels, height, alpha):
        raise NotImplementedError("gen_loss method has not been implemented")

# =============================================================
# Normal versions of the Losses:
# =============================================================

class StandardGAN(GANLoss):

    def __init__(self, dis):
        from torch.nn import BCEWithLogitsLoss

        super().__init__(dis)

        # define the criterion and activation used for object
        self.criterion = BCEWithLogitsLoss()

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        # small assertion:
        assert real_samps.device == fake_samps.device, \
            "Real and Fake samples are not on the same device"

        # device for computations:
        device = fake_samps.device

        # predictions for real images and fake images separately :
        r_preds = self.dis(real_samps, height, alpha)
        f_preds = self.dis(fake_samps, height, alpha)

        # calculate the real loss:
        real_loss = self.criterion(
            th.squeeze(r_preds),
            th.ones(real_samps.shape[0]).to(device))

        # calculate the fake loss:
        fake_loss = self.criterion(
            th.squeeze(f_preds),
            th.zeros(fake_samps.shape[0]).to(device))

        # return final losses
        return (real_loss + fake_loss) / 2

    def gen_loss(self, _, fake_samps, height, alpha):
        preds, _, _ = self.dis(fake_samps, height, alpha)
        return self.criterion(th.squeeze(preds),
                              th.ones(fake_samps.shape[0]).to(fake_samps.device))


class WGAN_GP(GANLoss):

    def __init__(self, dis, drift=0.001, use_gp=False):
        super().__init__(dis)
        self.drift = drift
        self.use_gp = use_gp

    def __gradient_penalty(self, real_samps, fake_samps,
                           height, alpha, reg_lambda=10):
        """
        private helper for calculating the gradient penalty
        :param real_samps: real samples
        :param fake_samps: fake samples
        :param height: current depth in the optimization
        :param alpha: current alpha for fade-in
        :param reg_lambda: regularisation lambda
        :return: tensor (gradient penalty)
        """
        batch_size = real_samps.shape[0]

        # generate random epsilon
        epsilon = th.rand((batch_size, 1, 1, 1)).to(fake_samps.device)

        # create the merge of both real and fake samples
        merged = epsilon * real_samps + ((1 - epsilon) * fake_samps)
        merged.requires_grad_(True)

        # forward pass
        op = self.dis(merged, height, alpha)

        # perform backward pass from op to merged for obtaining the gradients
        gradient = th.autograd.grad(outputs=op, inputs=merged,
                                    grad_outputs=th.ones_like(op), create_graph=True,
                                    retain_graph=True, only_inputs=True)[0]

        # calculate the penalty using these gradients
        gradient = gradient.view(gradient.shape[0], -1)
        penalty = reg_lambda * ((gradient.norm(p=2, dim=1) - 1) ** 2).mean()

        # return the calculated penalty:
        return penalty

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        # define the (Wasserstein) loss
        fake_out = self.dis(fake_samps, height, alpha)
        real_out = self.dis(real_samps, height, alpha)

        loss = (th.mean(fake_out) - th.mean(real_out)
                + (self.drift * th.mean(real_out ** 2)))

        if self.use_gp:
            # calculate the WGAN-GP (gradient penalty)
            gp = self.__gradient_penalty(real_samps, fake_samps, height, alpha)
            loss += gp

        return loss

    def gen_loss(self, _, fake_samps, height, alpha):
        # calculate the WGAN loss for generator
        loss = -th.mean(self.dis(fake_samps, height, alpha))

        return loss


class LSGAN(GANLoss):

    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        return 0.5 * (th.mean((self.dis(real_samps, height, alpha) - 1) ** 2)
                      + (th.mean(self.dis(fake_samps, height, alpha) ** 2)))

    def gen_loss(self, _, fake_samps, height, alpha):
        return 0.5 * (th.mean((self.dis(fake_samps, height, alpha) - 1) ** 2))


class LSGAN_SIGMOID(GANLoss):

    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        from torch.nn.functional import sigmoid
        real_scores = sigmoid(self.dis(real_samps, height, alpha))
        fake_scores = sigmoid(self.dis(fake_samps, height, alpha))
        return 0.5 * ((th.mean((real_scores - 1) ** 2)) + th.mean(fake_scores ** 2))

    def gen_loss(self, _, fake_samps, height, alpha):
        from torch.nn.functional import sigmoid
        scores = sigmoid(self.dis(fake_samps, height, alpha))
        return 0.5 * (th.mean((scores - 1) ** 2))


class HingeGAN(GANLoss):

    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        r_preds = self.dis(real_samps, height, alpha)
        f_preds = self.dis(fake_samps, height, alpha)

        loss = (th.mean(th.nn.ReLU()(1 - r_preds)) +
                th.mean(th.nn.ReLU()(1 + f_preds)))

        return loss

    def gen_loss(self, _, fake_samps, height, alpha):
        return -th.mean(self.dis(fake_samps, height, alpha))


class RelativisticAverageHingeGAN(GANLoss):

    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        # Obtain predictions
        r_preds = self.dis(real_samps, height, alpha)
        f_preds = self.dis(fake_samps, height, alpha)

        # difference between real and fake:
        r_f_diff = r_preds - th.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - th.mean(r_preds)

        # return the loss
        loss = (th.mean(th.nn.ReLU()(1 - r_f_diff))
                + th.mean(th.nn.ReLU()(1 + f_r_diff)))

        return loss

    def gen_loss(self, real_samps, fake_samps, height, alpha):
        # Obtain predictions
        r_preds = self.dis(real_samps, height, alpha)
        f_preds = self.dis(fake_samps, height, alpha)

        # difference between real and fake:
        r_f_diff = r_preds - th.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - th.mean(r_preds)

        # return the loss
        return (th.mean(th.nn.ReLU()(1 + r_f_diff))
                + th.mean(th.nn.ReLU()(1 - f_r_diff)))

# =============================================================
# Conditional versions of the Losses:
# =============================================================

class CondStandardGAN(ConditionalGANLoss):

    def __init__(self, dis):
        from torch.nn import BCEWithLogitsLoss

        super().__init__(dis)

        # define the criterion and activation used for object
        self.criterion = BCEWithLogitsLoss()

    def dis_loss(self, real_samps, fake_samps, labels, height, alpha):
        # small assertion:
        assert real_samps.device == fake_samps.device, \
            "Real and Fake samples are not on the same device"

        # device for computations:
        device = fake_samps.device

        # predictions for real images and fake images separately:
        r_preds = self.dis(real_samps, labels, height, alpha)
        f_preds = self.dis(fake_samps, labels, height, alpha)

        # calculate the real loss:
        real_loss = self.criterion(
            th.squeeze(r_preds),
            th.ones(real_samps.shape[0]).to(device))

        # calculate the fake loss:
        fake_loss = self.criterion(
            th.squeeze(f_preds),
            th.zeros(fake_samps.shape[0]).to(device))

        # return final loss
        return (real_loss + fake_loss) / 2

    def gen_loss(self, _, fake_samps, labels, height, alpha):
        preds, _, _ = self.dis(fake_samps, labels, height, alpha)
        return self.criterion(th.squeeze(preds),
                              th.ones(fake_samps.shape[0]).to(fake_samps.device))


class CondWGAN_GP(ConditionalGANLoss):

    def __init__(self, dis, drift=0.001, use_gp=False):
        super().__init__(dis)
        self.drift = drift
        self.use_gp = use_gp

    def __gradient_penalty(self, real_samps, fake_samps, labels,
                           height, alpha, reg_lambda=10):
        """
        private helper for calculating the gradient penalty
        :param real_samps: real samples
        :param fake_samps: fake samples
        :param labels: used for conditional loss calculation
                       Note that this is just [Batch x 1] plain integer labels
        :param height: current depth in the optimization
        :param alpha: current alpha for fade-in
        :param reg_lambda: regularisation lambda
        :return: tensor (gradient penalty)
        """
        from torch.autograd import grad

        batch_size = real_samps.shape[0]

        # generate random epsilon
        epsilon = th.rand((batch_size, 1, 1, 1)).to(fake_samps.device)

        # create the merge of both real and fake samples
        merged = (epsilon * real_samps) + ((1 - epsilon) * fake_samps)
        merged.requires_grad_(True)

        # forward pass
        op = self.dis(merged, labels, height, alpha)

        # obtain gradient of op wrt. merged
        gradient = grad(outputs=op, inputs=merged,
                        grad_outputs=th.ones_like(op), create_graph=True,
                        retain_graph=True, only_inputs=True)[0]

        # calculate the penalty using these gradients
        gradient = gradient.view(batch_size, -1)
        penalty = reg_lambda * ((gradient.norm(p=2, dim=1) - 1) ** 2).mean()

        # return the calculated penalty:
        return penalty

    def dis_loss(self, real_samps, fake_samps, labels, height, alpha):
        # define the (Wasserstein) loss
        fake_out = self.dis(fake_samps, labels, height, alpha)
        real_out = self.dis(real_samps, labels, height, alpha)

        loss = (th.mean(fake_out) - th.mean(real_out)
                + (self.drift * th.mean(real_out ** 2)))

        if self.use_gp:
            # calculate the WGAN-GP (gradient penalty)
            gp = self.__gradient_penalty(real_samps, fake_samps,
                                         labels, height, alpha)
            loss += gp

        return loss

    def gen_loss(self, _, fake_samps, labels, height, alpha):
        # calculate the WGAN loss for generator
        loss = -th.mean(self.dis(fake_samps, labels, height, alpha))

        return loss


class CondLSGAN(ConditionalGANLoss):

    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps, labels, height, alpha):
        return 0.5 * ((th.mean((self.dis(real_samps, labels, height, alpha) - 1) ** 2))
                      + (th.mean(self.dis(fake_samps, labels, height, alpha) ** 2)))

    def gen_loss(self, _, fake_samps, labels, height, alpha):
        return 0.5 * (th.mean((self.dis(fake_samps, labels, height, alpha) - 1) ** 2))


class CondLSGAN_SIGMOID(ConditionalGANLoss):

    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps, labels, height, alpha):
        from torch.nn.functional import sigmoid
        real_scores = sigmoid(self.dis(real_samps, labels, height, alpha))
        fake_scores = sigmoid(self.dis(fake_samps, labels, height, alpha))
        return 0.5 * (th.mean((real_scores - 1) ** 2) + th.mean(fake_scores ** 2))

    def gen_loss(self, _, fake_samps, labels, height, alpha):
        from torch.nn.functional import sigmoid
        scores = sigmoid(self.dis(fake_samps, labels, height, alpha))
        return 0.5 * (th.mean((scores - 1) ** 2))


class CondHingeGAN(ConditionalGANLoss):

    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps, labels, height, alpha):
        r_preds = self.dis(real_samps, labels, height, alpha)
        f_preds = self.dis(fake_samps, labels, height, alpha)

        loss = (th.mean(th.nn.ReLU()(1 - r_preds)) +
                th.mean(th.nn.ReLU()(1 + f_preds)))

        return loss

    def gen_loss(self, _, fake_samps, labels, height, alpha):
        return -th.mean(self.dis(fake_samps, labels, height, alpha))


class CondRelativisticAverageHingeGAN(ConditionalGANLoss):

    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps, labels, height, alpha):
        # Obtain predictions
        r_preds = self.dis(real_samps, labels, height, alpha)
        f_preds = self.dis(fake_samps, labels, height, alpha)

        # difference between real and fake:
        r_f_diff = r_preds - th.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - th.mean(r_preds)

        # return the loss
        loss = (th.mean(th.nn.ReLU()(1 - r_f_diff))
                + th.mean(th.nn.ReLU()(1 + f_r_diff)))

        return loss

    def gen_loss(self, real_samps, fake_samps, labels, height, alpha):
        # Obtain predictions
        r_preds = self.dis(real_samps, labels, height, alpha)
        f_preds = self.dis(fake_samps, labels, height, alpha)

        # difference between real and fake:
        r_f_diff = r_preds - th.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - th.mean(r_preds)

        # return the loss
        return (th.mean(th.nn.ReLU()(1 + r_f_diff))
                + th.mean(th.nn.ReLU()(1 - f_r_diff)))


#CustomLayers.py
""" Module containing custom layers """

import torch as th


# extending Conv2D and Deconv2D layers for equalized learning rate logic
class _equalized_conv2d(th.nn.Module):
    """ conv2d with the concept of equalized learning rate
        Args:
            :param c_in: input channels
            :param c_out:  output channels
            :param k_size: kernel size (h, w) should be a tuple or a single integer
            :param stride: stride for conv
            :param pad: padding
            :param bias: whether to use bias or not
    """

    def __init__(self, c_in, c_out, k_size, stride=1, pad=0, bias=True):
        """ constructor for the class """
        from torch.nn.modules.utils import _pair
        from numpy import sqrt, prod

        super(_equalized_conv2d, self).__init__()

        # define the weight and bias if to be used
        self.weight = th.nn.Parameter(th.nn.init.normal_(
            th.empty(c_out, c_in, *_pair(k_size))
        ))

        self.use_bias = bias
        self.stride = stride
        self.pad = pad

        if self.use_bias:
            self.bias = th.nn.Parameter(th.FloatTensor(c_out).fill_(0))

        fan_in = prod(_pair(k_size)) * c_in  # value of fan_in
        self.scale = sqrt(2) / sqrt(fan_in)

    def forward(self, x):
        """
        forward pass of the network
        :param x: input
        :return: y => output
        """
        from torch.nn.functional import conv2d

        return conv2d(input=x,
                      weight=self.weight * self.scale,  # scale the weight on runtime
                      bias=self.bias if self.use_bias else None,
                      stride=self.stride,
                      padding=self.pad)

    def extra_repr(self):
        return ", ".join(map(str, self.weight.shape))


class _equalized_deconv2d(th.nn.Module):
    """ Transpose convolution using the equalized learning rate
        Args:
            :param c_in: input channels
            :param c_out: output channels
            :param k_size: kernel size
            :param stride: stride for convolution transpose
            :param pad: padding
            :param bias: whether to use bias or not
    """

    def __init__(self, c_in, c_out, k_size, stride=1, pad=0, bias=True):
        """ constructor for the class """
        from torch.nn.modules.utils import _pair
        from numpy import sqrt

        super(_equalized_deconv2d, self).__init__()

        # define the weight and bias if to be used
        self.weight = th.nn.Parameter(th.nn.init.normal_(
            th.empty(c_in, c_out, *_pair(k_size))
        ))

        self.use_bias = bias
        self.stride = stride
        self.pad = pad

        if self.use_bias:
            self.bias = th.nn.Parameter(th.FloatTensor(c_out).fill_(0))

        fan_in = c_in  # value of fan_in for deconv
        self.scale = sqrt(2) / sqrt(fan_in)

    def forward(self, x):
        """
        forward pass of the layer
        :param x: input
        :return: y => output
        """
        from torch.nn.functional import conv_transpose2d

        return conv_transpose2d(input=x,
                                weight=self.weight * self.scale,  # scale the weight on runtime
                                bias=self.bias if self.use_bias else None,
                                stride=self.stride,
                                padding=self.pad)

    def extra_repr(self):
        return ", ".join(map(str, self.weight.shape))


class _equalized_linear(th.nn.Module):
    """ Linear layer using equalized learning rate
        Args:
            :param c_in: number of input channels
            :param c_out: number of output channels
            :param bias: whether to use bias with the linear layer
    """

    def __init__(self, c_in, c_out, bias=True):
        """
        Linear layer modified for equalized learning rate
        """
        from numpy import sqrt

        super(_equalized_linear, self).__init__()

        self.weight = th.nn.Parameter(th.nn.init.normal_(
            th.empty(c_out, c_in)
        ))

        self.use_bias = bias

        if self.use_bias:
            self.bias = th.nn.Parameter(th.FloatTensor(c_out).fill_(0))

        fan_in = c_in
        self.scale = sqrt(2) / sqrt(fan_in)

    def forward(self, x):
        """
        forward pass of the layer
        :param x: input
        :return: y => output
        """
        from torch.nn.functional import linear
        return linear(x, self.weight * self.scale,
                      self.bias if self.use_bias else None)


# ----------------------------------------------------------------------------
# Pixelwise feature vector normalization.
# reference: https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L120
# ----------------------------------------------------------------------------
class PixelwiseNorm(th.nn.Module):
    def __init__(self):
        super(PixelwiseNorm, self).__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the module
        :param x: input activations volume
        :param alpha: small number for numerical stability
        :return: y => pixel normalized activations
        """
        y = x.pow(2.).mean(dim=1, keepdim=True).add(alpha).sqrt()  # [N1HW]
        y = x / y  # normalize the input x volume
        return y


# ==========================================================
# Layers required for Building The generator and
# discriminator
# ==========================================================
class GenInitialBlock(th.nn.Module):
    """ Module implementing the initial block of the input """

    def __init__(self, in_channels, use_eql):
        """
        constructor for the inner class
        :param in_channels: number of input channels to the block
        :param use_eql: whether to use equalized learning rate
        """
        from torch.nn import LeakyReLU

        super(GenInitialBlock, self).__init__()

        if use_eql:
            self.conv_1 = _equalized_deconv2d(in_channels, in_channels, (4, 4), bias=True)
            self.conv_2 = _equalized_conv2d(in_channels, in_channels, (3, 3),
                                            pad=1, bias=True)

        else:
            from torch.nn import Conv2d, ConvTranspose2d
            self.conv_1 = ConvTranspose2d(in_channels, in_channels, (4, 4), bias=True)
            self.conv_2 = Conv2d(in_channels, in_channels, (3, 3), padding=1, bias=True)

        # Pixelwise feature vector normalization operation
        self.pixNorm = PixelwiseNorm()

        # leaky_relu:
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the block
        :param x: input to the module
        :return: y => output
        """
        # convert the tensor shape:
        y = th.unsqueeze(th.unsqueeze(x, -1), -1)

        # perform the forward computations:
        y = self.lrelu(self.conv_1(y))
        y = self.lrelu(self.conv_2(y))

        # apply pixel norm
        y = self.pixNorm(y)

        return y

# ==========================================================
# Layers required for Building The generator and
# discriminator
# ==========================================================
class ConditionalGenInitialBlock(th.nn.Module):
    """ Module implementing the initial block of the input """

    def __init__(self, in_channels, num_classes, embed_size, use_eql):
        """
        constructor for the inner class
        :param in_channels: number of input channels to the block
        :param use_eql: whether to use equalized learning rate
        """
        from torch.nn import LeakyReLU, Embedding

        super(ConditionalGenInitialBlock, self).__init__()

        if use_eql:
            self.conv_1 = _equalized_deconv2d(in_channels+embed_size, in_channels, (4, 4), bias=True)
            self.conv_2 = _equalized_conv2d(in_channels, in_channels, (3, 3),
                                            pad=1, bias=True)

        else:
            from torch.nn import Conv2d, ConvTranspose2d
            self.conv_1 = ConvTranspose2d(in_channels+embed_size, in_channels, (4, 4), bias=True)
            self.conv_2 = Conv2d(in_channels, in_channels, (3, 3), padding=1, bias=True)
   
        # Pixelwise feature vector normalization operation
        self.pixNorm = PixelwiseNorm()

        # leaky_relu:
        self.lrelu = LeakyReLU(0.2)
        
        self.embed = Embedding(num_classes, embed_size)

    def forward(self, x, labels):
        """
        forward pass of the block
        :param x: input to the module
        :return: y => output
        """
        # convert the tensor shape:
        y = th.unsqueeze(th.unsqueeze(x, -1), -1) # NxCx1x1
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        y = th.cat([y, embedding], dim = 1)
        # perform the forward computations:
        y = self.lrelu(self.conv_1(y))
        y = self.lrelu(self.conv_2(y))

        # apply pixel norm
        y = self.pixNorm(y)

        return y
    
# class CondGenerator(nn.Module):
#     def __init__(self, channels_noise, channels_img, features_g, num_classes, img_size, embed_size):
#         super(CondGenerator, self).__init__()
#         self.img_size = img_size
#         self.net = nn.Sequential(
#             # Input: N x channels_noise x 1 x 1
#             self._block(channels_noise + embed_size, features_g * 16, 4, 1, 0),  # img: 4x4
#             self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
#             self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
#             self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
#             nn.ConvTranspose2d(
#                 features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
#             ),
#             # Output: N x channels_img x 64 x 64
#             nn.Tanh(),
#         )
#         self.embed = nn.Embedding(num_classes, embed_size)

#     def _block(self, in_channels, out_channels, kernel_size, stride, padding):
#         return nn.Sequential(
#             nn.ConvTranspose2d(
#                 in_channels, out_channels, kernel_size, stride, padding, bias=False,
#             ),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#         )

#     def forward(self, x, labels):
#         # latent vector z: N x noise_dim x 1 x 1
#         embedding = self.embed(labels).unsqueeze(2).unsqueeze(3) # NxCx1x1
#         x = torch.cat([x, embedding], dim = 1)
#         return self.net(x)
    
class GenGeneralConvBlock(th.nn.Module):
    """ Module implementing a general convolutional block """

    def __init__(self, in_channels, out_channels, use_eql):
        """
        constructor for the class
        :param in_channels: number of input channels to the block
        :param out_channels: number of output channels required
        :param use_eql: whether to use equalized learning rate
        """
        from torch.nn import LeakyReLU
        from torch.nn.functional import interpolate

        super(GenGeneralConvBlock, self).__init__()

        self.upsample = lambda x: interpolate(x, scale_factor=2)

        if use_eql:
            self.conv_1 = _equalized_conv2d(in_channels, out_channels, (3, 3),
                                            pad=1, bias=True)
            self.conv_2 = _equalized_conv2d(out_channels, out_channels, (3, 3),
                                            pad=1, bias=True)
        else:
            from torch.nn import Conv2d
            self.conv_1 = Conv2d(in_channels, out_channels, (3, 3),
                                 padding=1, bias=True)
            self.conv_2 = Conv2d(out_channels, out_channels, (3, 3),
                                 padding=1, bias=True)

        # Pixelwise feature vector normalization operation
        self.pixNorm = PixelwiseNorm()

        # leaky_relu:
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the block
        :param x: input
        :return: y => output
        """
        y = self.upsample(x)
        y = self.pixNorm(self.lrelu(self.conv_1(y)))
        y = self.pixNorm(self.lrelu(self.conv_2(y)))

        return y


# function to calculate the Exponential moving averages for the Generator weights
# This function updates the exponential average weights based on the current training
def update_average(model_tgt, model_src, beta):
    """
    update the model_target using exponential moving averages
    :param model_tgt: target model
    :param model_src: source model
    :param beta: value of decay beta
    :return: None (updates the target model)
    """

    # utility function for toggling the gradient requirements of the models
    def toggle_grad(model, requires_grad):
        for p in model.parameters():
            p.requires_grad_(requires_grad)

    # turn off gradient calculation
    toggle_grad(model_tgt, False)
    toggle_grad(model_src, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert (p_src is not p_tgt)
        p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)

    # turn back on the gradient calculation
    toggle_grad(model_tgt, True)
    toggle_grad(model_src, True)


class MinibatchStdDev(th.nn.Module):
    """
    Minibatch standard deviation layer for the discriminator
    """

    def __init__(self):
        """
        derived class constructor
        """
        super(MinibatchStdDev, self).__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the layer
        :param x: input activation volume
        :param alpha: small number for numerical stability
        :return: y => x appended with standard deviation constant map
        """
        batch_size, _, height, width = x.shape

        # [B x C x H x W] Subtract mean over batch.
        y = x - x.mean(dim=0, keepdim=True)

        # [1 x C x H x W]  Calc standard deviation over batch
        y = th.sqrt(y.pow(2.).mean(dim=0, keepdim=False) + alpha)

        # [1]  Take average over feature_maps and pixels.
        y = y.mean().view(1, 1, 1, 1)

        # [B x 1 x H x W]  Replicate over group and pixels.
        y = y.repeat(batch_size, 1, height, width)

        # [B x C x H x W]  Append as new feature_map.
        y = th.cat([x, y], 1)

        # return the computed values:
        return y


class DisFinalBlock(th.nn.Module):
    """ Final block for the Discriminator """

    def __init__(self, in_channels, use_eql):
        """
        constructor of the class
        :param in_channels: number of input channels
        :param use_eql: whether to use equalized learning rate
        """
        from torch.nn import LeakyReLU

        super(DisFinalBlock, self).__init__()

        # declare the required modules for forward pass
        self.batch_discriminator = MinibatchStdDev()
        if use_eql:
            self.conv_1 = _equalized_conv2d(in_channels + 1, in_channels, (3, 3), pad=1, bias=True)
            self.conv_2 = _equalized_conv2d(in_channels, in_channels, (4, 4), bias=True)
            # final conv layer emulates a fully connected layer
            self.conv_3 = _equalized_conv2d(in_channels, 1, (1, 1), bias=True)
        else:
            from torch.nn import Conv2d
            self.conv_1 = Conv2d(in_channels + 1, in_channels, (3, 3), padding=1, bias=True)
            self.conv_2 = Conv2d(in_channels, in_channels, (4, 4), bias=True)
            # final conv layer emulates a fully connected layer
            self.conv_3 = Conv2d(in_channels, 1, (1, 1), bias=True)

        # leaky_relu:
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the FinalBlock
        :param x: input
        :return: y => output
        """
        # minibatch_std_dev layer
        y = self.batch_discriminator(x)

        # define the computations
        y = self.lrelu(self.conv_1(y))
        y = self.lrelu(self.conv_2(y))

        # fully connected layer
        y = self.conv_3(y)  # This layer has linear activation

        # flatten the output raw discriminator scores
        return y.view(-1)


class ConDisFinalBlock(th.nn.Module):
    """ Final block for the Conditional Discriminator
        Uses the Projection mechanism from the paper -> https://arxiv.org/pdf/1802.05637.pdf
    """

    def __init__(self, in_channels, num_classes, use_eql):
        """
        constructor of the class
        :param in_channels: number of input channels
        :param num_classes: number of classes for conditional discrimination
        :param use_eql: whether to use equalized learning rate
        """
        from torch.nn import LeakyReLU, Embedding

        super(ConDisFinalBlock, self).__init__()

        # declare the required modules for forward pass
        self.batch_discriminator = MinibatchStdDev()
        if use_eql:
            self.conv_1 = _equalized_conv2d(in_channels + 1, in_channels, (3, 3), pad=1, bias=True)
            self.conv_2 = _equalized_conv2d(in_channels, in_channels, (4, 4), bias=True)

            # final conv layer emulates a fully connected layer
            self.conv_3 = _equalized_conv2d(in_channels, 1, (1, 1), bias=True)
        else:
            from torch.nn import Conv2d
            self.conv_1 = Conv2d(in_channels + 1, in_channels, (3, 3), padding=1, bias=True)
            self.conv_2 = Conv2d(in_channels, in_channels, (4, 4), bias=True)

            # final conv layer emulates a fully connected layer
            self.conv_3 = Conv2d(in_channels, 1, (1, 1), bias=True)

        # we also need an embedding matrix for the label vectors
        self.label_embedder = Embedding(num_classes, in_channels, max_norm=1)

        # leaky_relu:
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x, labels):
        """
        forward pass of the FinalBlock
        :param x: input
        :param labels: samples' labels for conditional discrimination
                       Note that these are pure integer labels [Batch_size x 1]
        :return: y => output
        """
        # minibatch_std_dev layer
        y = self.batch_discriminator(x)  # [B x C x 4 x 4]

        # perform the forward pass
        y = self.lrelu(self.conv_1(y))  # [B x C x 4 x 4]

        # obtain the computed features
        y = self.lrelu(self.conv_2(y))  # [B x C x 1 x 1]

        # embed the labels
        labels = self.label_embedder(labels)  # [B x C]

        # compute the inner product with the label embeddings
        y_ = th.squeeze(th.squeeze(y, dim=-1), dim=-1)  # [B x C]
        projection_scores = (y_ * labels).sum(dim=-1)  # [B]

        # normal discrimination score
        y = self.lrelu(self.conv_3(y))  # This layer has linear activation

        # calculate the total score
        final_score = y.view(-1) + projection_scores

        # return the output raw discriminator scores
        return final_score


class DisGeneralConvBlock(th.nn.Module):
    """ General block in the discriminator  """

    def __init__(self, in_channels, out_channels, use_eql):
        """
        constructor of the class
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param use_eql: whether to use equalized learning rate
        """
        from torch.nn import AvgPool2d, LeakyReLU

        super(DisGeneralConvBlock, self).__init__()

        if use_eql:
            self.conv_1 = _equalized_conv2d(in_channels, in_channels, (3, 3), pad=1, bias=True)
            self.conv_2 = _equalized_conv2d(in_channels, out_channels, (3, 3), pad=1, bias=True)
        else:
            from torch.nn import Conv2d
            self.conv_1 = Conv2d(in_channels, in_channels, (3, 3), padding=1, bias=True)
            self.conv_2 = Conv2d(in_channels, out_channels, (3, 3), padding=1, bias=True)

        self.downSampler = AvgPool2d(2)

        # leaky_relu:
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the module
        :param x: input
        :return: y => output
        """
        # define the computations
        y = self.lrelu(self.conv_1(x))
        y = self.lrelu(self.conv_2(y))
        y = self.downSampler(y)

        return y
    
""" Module implementing GAN which will be trained using the Progressive growing
    technique -> https://arxiv.org/abs/1710.10196
"""
import datetime
import os
import time
import timeit
import copy
import numpy as np
import torch as th


# ========================================================================================
# Generator Module
# can be used with ProGAN, ConditionalProGAN or standalone (for inference)
# ========================================================================================

class Generator(th.nn.Module):
    """ Generator of the GAN network """

    def __init__(self, depth=7, latent_size=512, use_eql=True):
        """
        constructor for the Generator class
        :param depth: required depth of the Network
        :param latent_size: size of the latent manifold
        :param use_eql: whether to use equalized learning rate
        """
        from torch.nn import ModuleList
        from torch.nn.functional import interpolate

        super(Generator, self).__init__()

        assert latent_size != 0 and ((latent_size & (latent_size - 1)) == 0), \
            "latent size not a power of 2"
        if depth >= 4:
            assert latent_size >= np.power(2, depth - 4), "latent size will diminish to zero"

        # state of the generator:
        self.use_eql = use_eql
        self.depth = depth
        self.latent_size = latent_size

        # register the modules required for the GAN
        self.initial_block = GenInitialBlock(self.latent_size, use_eql=self.use_eql)

        # create a module list of the other required general convolution blocks
        self.layers = ModuleList([])  # initialize to empty list

        # create the ToRGB layers for various outputs:
        if self.use_eql:

            self.toRGB = lambda in_channels: \
                _equalized_conv2d(in_channels, IN_CHANNEL, (1, 1), bias=True)
        else:
            from torch.nn import Conv2d
            self.toRGB = lambda in_channels: Conv2d(in_channels, IN_CHANNEL, (1, 1), bias=True)

        self.rgb_converters = ModuleList([self.toRGB(self.latent_size)])

        # create the remaining layers
        for i in range(self.depth - 1):
            if i <= 2:
                layer = GenGeneralConvBlock(self.latent_size,
                                            self.latent_size, use_eql=self.use_eql)
                rgb = self.toRGB(self.latent_size)
            else:
                layer = GenGeneralConvBlock(
                    int(self.latent_size // np.power(2, i - 3)),
                    int(self.latent_size // np.power(2, i - 2)),
                    use_eql=self.use_eql
                )
                rgb = self.toRGB(int(self.latent_size // np.power(2, i - 2)))
            self.layers.append(layer)
            self.rgb_converters.append(rgb)

        # register the temporary upsampler
        self.temporaryUpsampler = lambda x: interpolate(x, scale_factor=2)

    def forward(self, x, depth, alpha):
        """
        forward pass of the Generator
        :param x: input noise
        :param depth: current depth from where output is required
        :param alpha: value of alpha for fade-in effect
        :return: y => output
        """

        assert depth < self.depth, "Requested output depth cannot be produced"

        y = self.initial_block(x)

        if depth > 0:
            for block in self.layers[:depth - 1]:
                y = block(y)

            residual = self.rgb_converters[depth - 1](self.temporaryUpsampler(y))
            straight = self.rgb_converters[depth](self.layers[depth - 1](y))

            out = (alpha * straight) + ((1 - alpha) * residual)

        else:
            out = self.rgb_converters[0](y)

        return out


# ========================================================================================
# Discriminator Module
# can be used with ProGAN or standalone (for inference).
# Note this cannot be used with ConditionalProGAN
# ========================================================================================

class Discriminator(th.nn.Module):
    """ Discriminator of the GAN """

    def __init__(self, height=7, feature_size=512, use_eql=True):
        """
        constructor for the class
        :param height: total height of the discriminator (Must be equal to the Generator depth)
        :param feature_size: size of the deepest features extracted
                             (Must be equal to Generator latent_size)
        :param use_eql: whether to use equalized learning rate
        """
        from torch.nn import ModuleList, AvgPool2d

        super(Discriminator, self).__init__()

        assert feature_size != 0 and ((feature_size & (feature_size - 1)) == 0), \
            "latent size not a power of 2"
        if height >= 4:
            assert feature_size >= np.power(2, height - 4), "feature size cannot be produced"

        # create state of the object
        self.use_eql = use_eql
        self.height = height
        self.feature_size = feature_size

        self.final_block = DisFinalBlock(self.feature_size, use_eql=self.use_eql)

        # create a module list of the other required general convolution blocks
        self.layers = ModuleList([])  # initialize to empty list

        # create the fromRGB layers for various inputs:
        if self.use_eql:
            self.fromRGB = lambda out_channels: \
                _equalized_conv2d(IN_CHANNEL, out_channels, (1, 1), bias=True)
        else:
            from torch.nn import Conv2d
            self.fromRGB = lambda out_channels: Conv2d(IN_CHANNEL, out_channels, (1, 1), bias=True)

        self.rgb_to_features = ModuleList([self.fromRGB(self.feature_size)])

        # create the remaining layers
        for i in range(self.height - 1):
            if i > 2:
                layer = DisGeneralConvBlock(
                    int(self.feature_size // np.power(2, i - 2)),
                    int(self.feature_size // np.power(2, i - 3)),
                    use_eql=self.use_eql
                )
                rgb = self.fromRGB(int(self.feature_size // np.power(2, i - 2)))
            else:
                layer = DisGeneralConvBlock(self.feature_size,
                                            self.feature_size, use_eql=self.use_eql)
                rgb = self.fromRGB(self.feature_size)

            self.layers.append(layer)
            self.rgb_to_features.append(rgb)

        # register the temporary downSampler
        self.temporaryDownsampler = AvgPool2d(2)

    def forward(self, x, height, alpha):
        """
        forward pass of the discriminator
        :param x: input to the network
        :param height: current height of operation (Progressive GAN)
        :param alpha: current value of alpha for fade-in
        :return: out => raw prediction values (WGAN-GP)
        """

        assert height < self.height, "Requested output depth cannot be produced"

        if height > 0:
            residual = self.rgb_to_features[height - 1](self.temporaryDownsampler(x))

            straight = self.layers[height - 1](
                self.rgb_to_features[height](x)
            )

            y = (alpha * straight) + ((1 - alpha) * residual)

            for block in reversed(self.layers[:height - 1]):
                y = block(y)
        else:
            y = self.rgb_to_features[0](x)

        out = self.final_block(y)

        return out
    
# class ConDiscriminator(th.nn.Module):
#     """ Discriminator of the GAN """

#     def __init__(self, height=7, feature_size=512, use_eql=True):
#         """
#         constructor for the class
#         :param height: total height of the discriminator (Must be equal to the Generator depth)
#         :param feature_size: size of the deepest features extracted
#                              (Must be equal to Generator latent_size)
#         :param use_eql: whether to use equalized learning rate
#         """
#         from torch.nn import ModuleList, AvgPool2d

#         super(ConDiscriminator, self).__init__()

#         assert feature_size != 0 and ((feature_size & (feature_size - 1)) == 0), \
#             "latent size not a power of 2"
#         if height >= 4:
#             assert feature_size >= np.power(2, height - 4), "feature size cannot be produced"

#         # create state of the object
#         self.use_eql = use_eql
#         self.height = height
#         self.feature_size = feature_size

#         self.final_block = DisFinalBlock(self.feature_size, use_eql=self.use_eql)

#         # create a module list of the other required general convolution blocks
#         self.layers = ModuleList([])  # initialize to empty list

#         # create the fromRGB layers for various inputs:
#         if self.use_eql:
#             self.fromRGB = lambda out_channels: \
#                 _equalized_conv2d(IN_CHANNEL, out_channels, (1, 1), bias=True)
#         else:
#             from torch.nn import Conv2d
#             self.fromRGB = lambda out_channels: Conv2d(IN_CHANNEL, out_channels, (1, 1), bias=True)

#         self.rgb_to_features = ModuleList([self.fromRGB(self.feature_size)])

#         # create the remaining layers
#         for i in range(self.height - 1):
#             if i > 2:
#                 layer = DisGeneralConvBlock(
#                     int(self.feature_size // np.power(2, i - 2)),
#                     int(self.feature_size // np.power(2, i - 3)),
#                     use_eql=self.use_eql
#                 )
#                 rgb = self.fromRGB(int(self.feature_size // np.power(2, i - 2)))
#             else:
#                 layer = DisGeneralConvBlock(self.feature_size,
#                                             self.feature_size, use_eql=self.use_eql)
#                 rgb = self.fromRGB(self.feature_size)

#             self.layers.append(layer)
#             self.rgb_to_features.append(rgb)

#         # register the temporary downSampler
#         self.temporaryDownsampler = AvgPool2d(2)

#     def forward(self, x, height, alpha):
#         """
#         forward pass of the discriminator
#         :param x: input to the network
#         :param height: current height of operation (Progressive GAN)
#         :param alpha: current value of alpha for fade-in
#         :return: out => raw prediction values (WGAN-GP)
#         """

#         assert height < self.height, "Requested output depth cannot be produced"

#         if height > 0:
#             residual = self.rgb_to_features[height - 1](self.temporaryDownsampler(x))

#             straight = self.layers[height - 1](
#                 self.rgb_to_features[height](x)
#             )

#             y = (alpha * straight) + ((1 - alpha) * residual)

#             for block in reversed(self.layers[:height - 1]):
#                 y = block(y)
#         else:
#             y = self.rgb_to_features[0](x)

#         out = self.final_block(y)

#         return out

# ========================================================================================
# Generator Module
# can be used with ProGAN, ConditionalProGAN or standalone (for inference)
# ========================================================================================

class ConditionalGenerator(th.nn.Module):
    """ Generator of the GAN network """

    def __init__(self, num_classes, embed_size, depth=7, latent_size=512, use_eql=True):
        """
        constructor for the Generator class
        :param num_classes: number of classes for conditional discrimination
        :param depth: required depth of the Network
        :param latent_size: size of the latent manifold
        :param use_eql: whether to use equalized learning rate
        """
        from torch.nn import ModuleList
        from torch.nn.functional import interpolate

        super(ConditionalGenerator, self).__init__()

        assert latent_size != 0 and ((latent_size & (latent_size - 1)) == 0), \
            "latent size not a power of 2"
        if depth >= 4:
            assert latent_size >= np.power(2, depth - 4), "latent size will diminish to zero"

        # state of the generator:
        self.use_eql = use_eql
        self.depth = depth
        self.latent_size = latent_size

        # register the modules required for the GAN
        self.initial_block = ConditionalGenInitialBlock(self.latent_size, num_classes, embed_size, use_eql=self.use_eql)

        # create a module list of the other required general convolution blocks
        self.layers = ModuleList([])  # initialize to empty list

        # create the ToRGB layers for various outputs:
        if self.use_eql:

            self.toRGB = lambda in_channels: \
                _equalized_conv2d(in_channels, IN_CHANNEL, (1, 1), bias=True)
        else:
            from torch.nn import Conv2d
            self.toRGB = lambda in_channels: Conv2d(in_channels, IN_CHANNEL, (1, 1), bias=True)

        self.rgb_converters = ModuleList([self.toRGB(self.latent_size)])

        # create the remaining layers
        for i in range(self.depth - 1):
            if i <= 2:
                layer = GenGeneralConvBlock(self.latent_size,
                                            self.latent_size, use_eql=self.use_eql)
                rgb = self.toRGB(self.latent_size)
            else:
                layer = GenGeneralConvBlock(
                    int(self.latent_size // np.power(2, i - 3)),
                    int(self.latent_size // np.power(2, i - 2)),
                    use_eql=self.use_eql
                )
                rgb = self.toRGB(int(self.latent_size // np.power(2, i - 2)))
            self.layers.append(layer)
            self.rgb_converters.append(rgb)

        # register the temporary upsampler
        self.temporaryUpsampler = lambda x: interpolate(x, scale_factor=2)

    def forward(self, x, labels, depth, alpha):
        """
        forward pass of the Generator
        :param x: input noise
        :param depth: current depth from where output is required
        :param alpha: value of alpha for fade-in effect
        :return: y => output
        """

        assert depth < self.depth, "Requested output depth cannot be produced"

        y = self.initial_block(x, labels)

        if depth > 0:
            for block in self.layers[:depth - 1]:
                y = block(y)

            residual = self.rgb_converters[depth - 1](self.temporaryUpsampler(y))
            straight = self.rgb_converters[depth](self.layers[depth - 1](y))

            out = (alpha * straight) + ((1 - alpha) * residual)

        else:
            out = self.rgb_converters[0](y)

        return out
    
# ========================================================================================
# ConditionalDiscriminator Module
# uses the projection discrimination mechanism
# can be used with ConditionalProGAN or standalone (for inference)
# Note that this is not to be used with ProGAN
# ========================================================================================

class ConditionalDiscriminator(th.nn.Module):
    """ Discriminator of the GAN """

    def __init__(self, num_classes, height=7, feature_size=512, use_eql=True):
        """
        constructor for the class
        :param num_classes: number of classes for conditional discrimination
        :param height: total height of the discriminator (Must be equal to the Generator depth)
        :param feature_size: size of the deepest features extracted
                             (Must be equal to Generator latent_size)
        :param use_eql: whether to use equalized learning rate
        """
        from torch.nn import ModuleList, AvgPool2d

        super(ConditionalDiscriminator, self).__init__()

        assert feature_size != 0 and ((feature_size & (feature_size - 1)) == 0), \
            "latent size not a power of 2"
        if height >= 4:
            assert feature_size >= np.power(2, height - 4), "feature size cannot be produced"

        # create state of the object
        self.use_eql = use_eql
        self.height = height
        self.feature_size = feature_size
        self.num_classes = num_classes

        self.final_block = ConDisFinalBlock(self.feature_size, self.num_classes,
                                            use_eql=self.use_eql)

        # create a module list of the other required general convolution blocks
        self.layers = ModuleList([])  # initialize to empty list

        # create the fromRGB layers for various inputs:
        if self.use_eql:
            self.fromRGB = lambda out_channels: \
                _equalized_conv2d(IN_CHANNEL, out_channels, (1, 1), bias=True)
        else:
            from torch.nn import Conv2d
            self.fromRGB = lambda out_channels: Conv2d(IN_CHANNEL, out_channels, (1, 1), bias=True)

        self.rgb_to_features = ModuleList([self.fromRGB(self.feature_size)])

        # create the remaining layers
        for i in range(self.height - 1):
            if i > 2:
                layer = DisGeneralConvBlock(
                    int(self.feature_size // np.power(2, i - 2)),
                    int(self.feature_size // np.power(2, i - 3)),
                    use_eql=self.use_eql
                )
                rgb = self.fromRGB(int(self.feature_size // np.power(2, i - 2)))
            else:
                layer = DisGeneralConvBlock(self.feature_size,
                                            self.feature_size, use_eql=self.use_eql)
                rgb = self.fromRGB(self.feature_size)

            self.layers.append(layer)
            self.rgb_to_features.append(rgb)

        # register the temporary downSampler
        self.temporaryDownsampler = AvgPool2d(2)

    def forward(self, x, labels, height, alpha):
        """
        forward pass of the discriminator
        :param x: input to the network
        :param labels: labels required for conditional discrimination
                       note that these are pure integer labels of shape [B x 1]
        :param height: current height of operation (Progressive GAN)
        :param alpha: current value of alpha for fade-in
        :return: out => raw prediction values
        """

        assert height < self.height, "Requested output depth cannot be produced"

        if height > 0:
            residual = self.rgb_to_features[height - 1](self.temporaryDownsampler(x))

            straight = self.layers[height - 1](
                self.rgb_to_features[height](x)
            )

            y = (alpha * straight) + ((1 - alpha) * residual)

            for block in reversed(self.layers[:height - 1]):
                y = block(y)
        else:
            y = self.rgb_to_features[0](x)

        out = self.final_block(y, labels)

        return out


# ========================================================================================
# ProGAN Module (Unconditional)
# ========================================================================================

class ProGAN:
    """ Wrapper around the Generator and the Discriminator """

    def __init__(self, depth=7, latent_size=512, learning_rate=0.001, beta_1=0,
                 beta_2=0.99, eps=1e-8, drift=0.001, n_critic=1, use_eql=True,
                 loss="wgan-gp", use_ema=True, ema_decay=0.999,
                 device=th.device("cpu")):
        """
        constructor for the class
        :param depth: depth of the GAN (will be used for each generator and discriminator)
        :param latent_size: latent size of the manifold used by the GAN
        :param learning_rate: learning rate for Adam
        :param beta_1: beta_1 for Adam
        :param beta_2: beta_2 for Adam
        :param eps: epsilon for Adam
        :param n_critic: number of times to update discriminator per generator update
        :param drift: drift penalty for the
                      (Used only if loss is wgan or wgan-gp)
        :param use_eql: whether to use equalized learning rate
        :param loss: the loss function to be used
                     Can either be a string =>
                          ["wgan-gp", "wgan", "lsgan", "lsgan-with-sigmoid",
                          "hinge", "standard-gan" or "relativistic-hinge"]
                     Or an instance of GANLoss
        :param use_ema: boolean for whether to use exponential moving averages
        :param ema_decay: value of mu for ema
        :param device: device to run the GAN on (GPU / CPU)
        """

        from torch.optim import Adam
        from torch.nn import DataParallel

        # Create the Generator and the Discriminator
        self.gen = Generator(depth, latent_size, use_eql=use_eql).to(device)
        self.dis = Discriminator(depth, latent_size, use_eql=use_eql).to(device)

        # if code is to be run on GPU, we can use DataParallel:
        if device == th.device("cuda"):
            print("use DataParallel for Generator and Discriminator")
            self.gen = DataParallel(self.gen)
            self.dis = DataParallel(self.dis)

        # state of the object
        self.latent_size = latent_size
        self.depth = depth
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.n_critic = n_critic
        self.use_eql = use_eql
        self.device = device
        self.drift = drift

        # define the optimizers for the discriminator and generator
        self.gen_optim = Adam(self.gen.parameters(), lr=learning_rate,
                              betas=(beta_1, beta_2), eps=eps)

        self.dis_optim = Adam(self.dis.parameters(), lr=learning_rate,
                              betas=(beta_1, beta_2), eps=eps)

        # define the loss function used for training the GAN
        self.loss = self.__setup_loss(loss)

        if self.use_ema:

            # create a shadow copy of the generator
            self.gen_shadow = copy.deepcopy(self.gen)

            # updater function:
            self.ema_updater = update_average

            # initialize the gen_shadow weights equal to the
            # weights of gen
            self.ema_updater(self.gen_shadow, self.gen, beta=0)

    def __setup_loss(self, loss):

        if isinstance(loss, str):
            loss = loss.lower()  # lowercase the string
            if loss == "wgan":
                loss = WGAN_GP(self.dis, self.drift, use_gp=False)
                # note if you use just wgan, you will have to use weight clipping
                # in order to prevent gradient exploding
                # check the optimize_discriminator method where this has been
                # taken care of.

            elif loss == "wgan-gp":
                loss = WGAN_GP(self.dis, self.drift, use_gp=True)

            elif loss == "standard-gan":
                loss = StandardGAN(self.dis)

            elif loss == "lsgan":
                loss = LSGAN(self.dis)

            elif loss == "lsgan-with-sigmoid":
                loss = LSGAN_SIGMOID(self.dis)

            elif loss == "hinge":
                loss = HingeGAN(self.dis)

            elif loss == "relativistic-hinge":
                loss = RelativisticAverageHingeGAN(self.dis)

            else:
                raise ValueError("Unknown loss function requested")

        elif not isinstance(loss, GANLoss):
            raise ValueError("loss is neither an instance of GANLoss nor a string")

        return loss

    def __progressive_downsampling(self, real_batch, depth, alpha):
        """
        private helper for downsampling the original images in order to facilitate the
        progressive growing of the layers.
        :param real_batch: batch of real samples
        :param depth: depth at which training is going on
        :param alpha: current value of the fader alpha
        :return: real_samples => modified real batch of samples
        """

        from torch.nn import AvgPool2d
        from torch.nn.functional import interpolate

        # downsample the real_batch for the given depth
        down_sample_factor = int(np.power(2, self.depth - depth - 1))
        prior_downsample_factor = max(int(np.power(2, self.depth - depth)), 0)

        ds_real_samples = AvgPool2d(down_sample_factor)(real_batch)

        if depth > 0:
            prior_ds_real_samples = interpolate(AvgPool2d(prior_downsample_factor)(real_batch),
                                                scale_factor=2)
        else:
            prior_ds_real_samples = ds_real_samples

        # real samples are a combination of ds_real_samples and prior_ds_real_samples
        real_samples = (alpha * ds_real_samples) + ((1 - alpha) * prior_ds_real_samples)

        # return the so computed real_samples
        return real_samples

    def optimize_discriminator(self, noise, real_batch, depth, alpha):
        """
        performs one step of weight update on discriminator using the batch of data
        :param noise: input noise of sample generation
        :param real_batch: real samples batch
        :param depth: current depth of optimization
        :param alpha: current alpha for fade-in
        :return: current loss (Wasserstein loss)
        """

        real_samples = self.__progressive_downsampling(real_batch, depth, alpha)

        loss_val = 0
        for _ in range(self.n_critic):
            # generate a batch of samples
            fake_samples = self.gen(noise, depth, alpha).detach()

            loss = self.loss.dis_loss(real_samples, fake_samples, depth, alpha)

            # optimize discriminator
            self.dis_optim.zero_grad()
            loss.backward()
            self.dis_optim.step()

            loss_val += loss.item()

        return loss_val / self.n_critic

    def optimize_generator(self, noise, real_batch, depth, alpha):
        """
        performs one step of weight update on generator for the given batch_size
        :param noise: input random noise required for generating samples
        :param real_batch: batch of real samples
        :param depth: depth of the network at which optimization is done
        :param alpha: value of alpha for fade-in effect
        :return: current loss (Wasserstein estimate)
        """

        real_samples = self.__progressive_downsampling(real_batch, depth, alpha)

        # generate fake samples:
        fake_samples = self.gen(noise, depth, alpha)

        # TODO_complete:
        # Change this implementation for making it compatible for relativisticGAN
        loss = self.loss.gen_loss(real_samples, fake_samples, depth, alpha)

        # optimize the generator
        self.gen_optim.zero_grad()
        loss.backward()
        self.gen_optim.step()

        # if use_ema is true, apply ema to the generator parameters
        if self.use_ema:
            self.ema_updater(self.gen_shadow, self.gen, self.ema_decay)

        # return the loss value
        return loss.item()

    @staticmethod
    def create_grid(samples, scale_factor, img_file):
        """
        utility function to create a grid of GAN samples
        :param samples: generated samples for storing
        :param scale_factor: factor for upscaling the image
        :param img_file: name of file to write
        :return: None (saves a file)
        """
        from torchvision.utils import save_image
        from torch.nn.functional import interpolate

        # upsample the image
        if scale_factor > 1:
            samples = interpolate(samples, scale_factor=scale_factor)

        # save the images:
        save_image(samples, img_file, nrow=int(np.sqrt(len(samples))),
                   normalize=True, scale_each=True)

    def train(self, dataset, epochs, batch_sizes,
              fade_in_percentage, num_samples=16,
              start_depth=0, start_epoch=1, num_workers=3, feedback_factor=100,
              log_dir="./models/", sample_dir="./samples/", save_dir="./models/",
              checkpoint_factor=1):
        """
        Utility method for training the ProGAN. Note that you don't have to necessarily use this
        you can use the optimize_generator and optimize_discriminator for your own training routine.
        :param dataset: object of the dataset used for training.
                        Note that this is not the dataloader (we create dataloader in this method
                        since the batch_sizes for resolutions can be different)
        :param epochs: list of number of epochs to train the network for every resolution
        :param batch_sizes: list of batch_sizes for every resolution
        :param fade_in_percentage: list of percentages of epochs per resolution
                                   used for fading in the new layer
                                   not used for first resolution, but dummy value still needed.
        :param num_samples: number of samples generated in sample_sheet. def=36
        :param start_depth: start training from this depth. def=0
        :param num_workers: number of workers for reading the data. def=3
        :param feedback_factor: number of logs per epoch. def=100
        :param log_dir: directory for saving the loss logs. def="./models/"
        :param sample_dir: directory for saving the generated samples. def="./samples/"
        :param checkpoint_factor: save model after these many epochs.
                                  Note that only one model is stored per resolution.
                                  during one resolution, the checkpoint will be updated (Rewritten)
                                  according to this factor.
        :param save_dir: directory for saving the models (.pth files)
        :return: None (Writes multiple files to disk)
        """

        assert self.depth == len(batch_sizes), "batch_sizes not compatible with depth"

        #RELOAD
        gen_save_file = os.path.join(save_dir, "GAN_GEN_" + str(start_depth) + ".pth")

        dis_save_file = os.path.join(save_dir, "GAN_DIS_" + str(start_depth) + ".pth")
        gen_optim_save_file = os.path.join(save_dir,
                                            "GAN_GEN_OPTIM_" + str(start_depth)
                                            + ".pth")
        dis_optim_save_file = os.path.join(save_dir,
                                            "GAN_DIS_OPTIM_" + str(start_depth)
                                            + ".pth")
        is_reload = False
        if exists(gen_save_file):
            print("--reload model--")
            is_reload = True
 
        if is_reload:
            self.gen.load_state_dict(th.load(gen_save_file))
            self.dis.load_state_dict(th.load(dis_save_file))
            self.gen_optim.load_state_dict(th.load(gen_optim_save_file))
            self.dis_optim.load_state_dict(th.load(dis_optim_save_file))

        # # also save the shadow generator if use_ema is True
        if self.use_ema:
            gen_shadow_save_file = os.path.join(save_dir, "GAN_GEN_SHADOW_" +
                                                str(start_depth) + ".pth")
            if is_reload:
                self.gen_shadow.load_state_dict(th.load(gen_shadow_save_file))
        #RELOAD

        # turn the generator and discriminator into train mode
        self.gen.train()
        self.dis.train()
        if self.use_ema:
            self.gen_shadow.train()

        # create a global time counter
        global_time = time.time()

        # create fixed_input for debugging
        fixed_input = th.randn(num_samples, self.latent_size).to(self.device)

        print("Starting the training process ... ")
        for current_depth in range(start_depth, self.depth):


            print("\n\nCurrently working on Depth: ", current_depth)
            current_res = np.power(2, current_depth + 2)
            print("Current resolution: %d x %d" % (current_res, current_res))

            data = get_data_loader(dataset, batch_sizes[current_depth], num_workers)
            if (start_epoch > 1): ticker = len(range(1,start_epoch))* len(iter(data))
            else: ticker = 1
            print("Ticker", ticker)

            if start_depth != current_depth:
                start_epoch = 1

            for epoch in range(start_epoch, epochs[current_depth] + 1):
                start = timeit.default_timer()  # record time at the start of epoch

                print("\nEpoch: %d" % epoch)
                total_batches = len(iter(data))

                fader_point = int((fade_in_percentage[current_depth] / 100)
                                  * epochs[current_depth] * total_batches)

                step = 0  # counter for number of iterations

                for (i, batch) in enumerate(data, 1):
                    # calculate the alpha for fading in the layers
                    alpha = ticker / fader_point if ticker <= fader_point else 1

                    # extract current batch of data for training
                    images = batch[0].to(self.device)

                    gan_input = th.randn(images.shape[0], self.latent_size).to(self.device)

                    # optimize the discriminator:
                    dis_loss = self.optimize_discriminator(gan_input, images,
                                                           current_depth, alpha)

                    # optimize the generator:
                    gen_loss = self.optimize_generator(gan_input, images, current_depth, alpha)

                    # provide a loss feedback
                    if i % int(total_batches / feedback_factor) == 0 or i == 1:
                        elapsed = time.time() - global_time
                        elapsed = str(datetime.timedelta(seconds=elapsed))
                        print("Elapsed: [%s]  batch: %d  d_loss: %f  g_loss: %f"
                              % (elapsed, i, dis_loss, gen_loss))

                        # also write the losses to the log file:
                        os.makedirs(log_dir, exist_ok=True)
                        log_file = os.path.join(log_dir, "loss_" + str(current_depth) + ".log")
                        with open(log_file, "a") as log:
                            log.write(str(step) + "\t" + str(dis_loss) +
                                      "\t" + str(gen_loss) + "\n")
                    # increment the alpha ticker and the step
                    ticker += 1
                    step += 1
                # create a grid of samples and save it
                os.makedirs(sample_dir, exist_ok=True)
                gen_img_file = os.path.join(sample_dir, "gen_" + str(current_depth) +
                                            "_" + str(epoch) + "_" +
                                            str(i) + ".png")

                # this is done to allow for more GPU space
                with th.no_grad():
                    self.create_grid(
                        samples=self.gen(
                            fixed_input,
                            current_depth,
                            alpha
                        ).detach() if not self.use_ema
                        else self.gen_shadow(
                            fixed_input,
                            current_depth,
                            alpha
                        ).detach(),
                        scale_factor=int(np.power(2, self.depth - current_depth - 1)),
                        img_file=gen_img_file,
                    )

                stop = timeit.default_timer()
                print("Time taken for epoch: %.3f secs" % (stop - start))
                print("ticker = ", ticker)
                if epoch % checkpoint_factor == 0 or epoch == 1 or epoch%5 ==0 or epoch == epochs[current_depth]:
                    os.makedirs(save_dir, exist_ok=True)
                    gen_save_file = os.path.join(save_dir, "GAN_GEN_" + str(current_depth) + ".pth")
                    dis_save_file = os.path.join(save_dir, "GAN_DIS_" + str(current_depth) + ".pth")
                    gen_optim_save_file = os.path.join(save_dir,
                                                       "GAN_GEN_OPTIM_" + str(current_depth)
                                                       + ".pth")
                    dis_optim_save_file = os.path.join(save_dir,
                                                       "GAN_DIS_OPTIM_" + str(current_depth)
                                                       + ".pth")

                    th.save(self.gen.state_dict(), gen_save_file)
                    th.save(self.dis.state_dict(), dis_save_file)
                    th.save(self.gen_optim.state_dict(), gen_optim_save_file)
                    th.save(self.dis_optim.state_dict(), dis_optim_save_file)

                    # also save the shadow generator if use_ema is True
                    if self.use_ema:
                        gen_shadow_save_file = os.path.join(save_dir, "GAN_GEN_SHADOW_" +
                                                            str(current_depth) + ".pth")
                        th.save(self.gen_shadow.state_dict(), gen_shadow_save_file)

        # put the gen, shadow_gen and dis in eval mode
        self.gen.eval()
        self.dis.eval()
        if self.use_ema:
            self.gen_shadow.eval()

        print("Training completed ...")
    def reload(self, start_depth = 0, save_dir = './models/'):
        gen_save_file = os.path.join(save_dir, "GAN_GEN_" + str(start_depth) + ".pth")
        dis_save_file = os.path.join(save_dir, "GAN_DIS_" + str(start_depth) + ".pth")
        gen_optim_save_file = os.path.join(save_dir,
                                        "GAN_GEN_OPTIM_" + str(start_depth)
                                        + ".pth")
        dis_optim_save_file = os.path.join(save_dir,
                                        "GAN_DIS_OPTIM_" + str(start_depth)
                                        + ".pth")
        self.gen.load_state_dict(th.load(gen_save_file))
        self.dis.load_state_dict(th.load(dis_save_file))
        self.gen_optim.load_state_dict(th.load(gen_optim_save_file))
        self.dis_optim.load_state_dict(th.load(dis_optim_save_file))
        if self.use_ema:
            gen_shadow_save_file = os.path.join(save_dir, "GAN_GEN_SHADOW_" +
                                            str(start_depth) + ".pth")
            self.gen_shadow.load_state_dict(th.load(gen_shadow_save_file))
            
    def print(self, input, depth, alpha):
        return (self.gen(input, depth, alpha ).detach() 
                    if not self.use_ema else 
                        self.gen_shadow(
                            input,
                            depth,
                            alpha
                        ).detach())
        
import torch as th
import torchvision as tv
from torch.utils.data import TensorDataset
from PIL import Image

def custom_loader_rgba(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert("RGBA")

def setup_data(data_path, custom = True, size = 256, num = -1):

    transforms = tv.transforms.Compose([
        tv.transforms.Resize([size, size]),
        tv.transforms.ToTensor(),
    ])
    trainset = None
    if custom:
        trainset = tv.datasets.ImageFolder(root=data_path, loader= custom_loader_rgba,
                                   transform=transforms)
    else:
        
        trainset = tv.datasets.ImageFolder(root=data_path,
                                   transform=transforms)        
    # return trainset
    if num !=  -1: 
        return th.utils.data.Subset(trainset, np.random.choice(len(trainset), num, replace=False))
    else:
        return trainset
    
    
    
class ConditionalProGAN:
    """ Wrapper around the Generator and the Discriminator """

    def __init__(self, num_classes, embed_size = 100, depth=7, latent_size=512, learning_rate=0.001, beta_1=0,
                 beta_2=0.99, eps=1e-8, drift=0.001, n_critic=1, use_eql=True,
                 loss="wgan-gp", use_ema=True, ema_decay=0.999,
                 device=th.device("cpu")):
        """
        constructor for the class
        :param depth: depth of the GAN (will be used for each generator and discriminator)
        :param latent_size: latent size of the manifold used by the GAN
        :param learning_rate: learning rate for Adam
        :param beta_1: beta_1 for Adam
        :param beta_2: beta_2 for Adam
        :param eps: epsilon for Adam
        :param n_critic: number of times to update discriminator per generator update
        :param drift: drift penalty for the
                      (Used only if loss is wgan or wgan-gp)
        :param use_eql: whether to use equalized learning rate
        :param loss: the loss function to be used
                     Can either be a string =>
                          ["wgan-gp", "wgan", "lsgan", "lsgan-with-sigmoid",
                          "hinge", "standard-gan" or "relativistic-hinge"]
                     Or an instance of GANLoss
        :param use_ema: boolean for whether to use exponential moving averages
        :param ema_decay: value of mu for ema
        :param device: device to run the GAN on (GPU / CPU)
        """

        from torch.optim import Adam
        from torch.nn import DataParallel

        # Create the Generator and the Discriminator
        self.gen = ConditionalGenerator(num_classes, embed_size, depth, latent_size, use_eql=use_eql).to(device)
        self.dis = ConditionalDiscriminator(num_classes, depth, latent_size, use_eql=use_eql).to(device)

        # if code is to be run on GPU, we can use DataParallel:
        if device == th.device("cuda"):
            print("use DataParallel for Generator and Discriminator")
            self.gen = DataParallel(self.gen)
            self.dis = DataParallel(self.dis)

        # state of the object
        self.num_classes = num_classes
        self.embed_size = embed_size
        self.latent_size = latent_size
        self.depth = depth
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.n_critic = n_critic
        self.use_eql = use_eql
        self.device = device
        self.drift = drift

        # define the optimizers for the discriminator and generator
        self.gen_optim = Adam(self.gen.parameters(), lr=learning_rate,
                              betas=(beta_1, beta_2), eps=eps)

        self.dis_optim = Adam(self.dis.parameters(), lr=learning_rate,
                              betas=(beta_1, beta_2), eps=eps)

        # define the loss function used for training the GAN
        self.loss = self.__setup_loss(loss)

        if self.use_ema:

            # create a shadow copy of the generator
            self.gen_shadow = copy.deepcopy(self.gen)

            # updater function:
            self.ema_updater = update_average

            # initialize the gen_shadow weights equal to the
            # weights of gen
            self.ema_updater(self.gen_shadow, self.gen, beta=0)

    def __setup_loss(self, loss):

        if isinstance(loss, str):
            loss = loss.lower()  # lowercase the string
            if loss == "wgan":
                loss = CondWGAN_GP(self.dis, self.drift, use_gp=False)
                # note if you use just wgan, you will have to use weight clipping
                # in order to prevent gradient exploding
                # check the optimize_discriminator method where this has been
                # taken care of.

            elif loss == "wgan-gp":
                loss = CondWGAN_GP(self.dis, self.drift, use_gp=True)

            elif loss == "standard-gan":
                loss = CondStandardGAN(self.dis)

            elif loss == "lsgan":
                loss = CondLSGAN(self.dis)

            elif loss == "lsgan-with-sigmoid":
                loss = CondLSGAN_SIGMOID(self.dis)

            elif loss == "hinge":
                loss = CondHingeGAN(self.dis)

            elif loss == "relativistic-hinge":
                loss = CondRelativisticAverageHingeGAN(self.dis)

            else:
                raise ValueError("Unknown loss function requested")

        elif not isinstance(loss, ConditionalGANLoss):
            raise ValueError("loss is neither an instance of GANLoss nor a string")

        return loss

    def __progressive_downsampling(self, real_batch, depth, alpha):
        """
        private helper for downsampling the original images in order to facilitate the
        progressive growing of the layers.
        :param real_batch: batch of real samples
        :param depth: depth at which training is going on
        :param alpha: current value of the fader alpha
        :return: real_samples => modified real batch of samples
        """

        from torch.nn import AvgPool2d
        from torch.nn.functional import interpolate

        # downsample the real_batch for the given depth
        down_sample_factor = int(np.power(2, self.depth - depth - 1))
        prior_downsample_factor = max(int(np.power(2, self.depth - depth)), 0)
        ds_real_samples = AvgPool2d(down_sample_factor)(real_batch)

        if depth > 0:
            prior_ds_real_samples = interpolate(AvgPool2d(prior_downsample_factor)(real_batch),
                                                scale_factor=2)
        else:
            prior_ds_real_samples = ds_real_samples

        # real samples are a combination of ds_real_samples and prior_ds_real_samples
        real_samples = (alpha * ds_real_samples) + ((1 - alpha) * prior_ds_real_samples)

        # return the so computed real_samples
        return real_samples

    def optimize_discriminator(self, noise, labels, real_batch, depth, alpha):
        """
        performs one step of weight update on discriminator using the batch of data
        :param noise: input noise of sample generation
        :param real_batch: real samples batch
        :param depth: current depth of optimization
        :param alpha: current alpha for fade-in
        :return: current loss (Wasserstein loss)
        """

        real_samples = self.__progressive_downsampling(real_batch, depth, alpha)
        loss_val = 0
        for _ in range(self.n_critic):
            # generate a batch of samples
            fake_samples = self.gen(noise, labels, depth, alpha).detach()
            loss = self.loss.dis_loss(real_samples, fake_samples, labels, depth, alpha)

            # optimize discriminator
            self.dis_optim.zero_grad()
            loss.backward()
            self.dis_optim.step()

            loss_val += loss.item()

        return loss_val / self.n_critic

    def optimize_generator(self, noise, labels, real_batch, depth, alpha):
        """
        performs one step of weight update on generator for the given batch_size
        :param noise: input random noise required for generating samples
        :param real_batch: batch of real samples
        :param depth: depth of the network at which optimization is done
        :param alpha: value of alpha for fade-in effect
        :return: current loss (Wasserstein estimate)
        """

        real_samples = self.__progressive_downsampling(real_batch, depth, alpha)

        # generate fake samples:
        fake_samples = self.gen(noise, labels, depth, alpha)

        # TODO_complete:
        # Change this implementation for making it compatible for relativisticGAN
        loss = self.loss.gen_loss(real_samples, fake_samples, labels, depth, alpha)

        # optimize the generator
        self.gen_optim.zero_grad()
        loss.backward()
        self.gen_optim.step()

        # if use_ema is true, apply ema to the generator parameters
        if self.use_ema:
            self.ema_updater(self.gen_shadow, self.gen, self.ema_decay)

        # return the loss value
        return loss.item()

    @staticmethod
    def create_grid(samples, scale_factor, img_file):
        """
        utility function to create a grid of GAN samples
        :param samples: generated samples for storing
        :param scale_factor: factor for upscaling the image
        :param img_file: name of file to write
        :return: None (saves a file)
        """
        from torchvision.utils import save_image
        from torch.nn.functional import interpolate

        # upsample the image
        if scale_factor > 1:
            samples = interpolate(samples, scale_factor=scale_factor)

        # save the images:
        save_image(samples, img_file, nrow=int(np.sqrt(len(samples))),
                   normalize=True, scale_each=True)

    def train(self, dataset, epochs, batch_sizes,
              fade_in_percentage, num_samples=9,
              start_depth=0, start_epoch=1, num_workers=3, feedback_factor=100,
              log_dir="./models/", sample_dir="./samples/", save_dir="./models/",
              checkpoint_factor=1):
        """
        Utility method for training the ProGAN. Note that you don't have to necessarily use this
        you can use the optimize_generator and optimize_discriminator for your own training routine.
        :param dataset: object of the dataset used for training.
                        Note that this is not the dataloader (we create dataloader in this method
                        since the batch_sizes for resolutions can be different)
        :param epochs: list of number of epochs to train the network for every resolution
        :param batch_sizes: list of batch_sizes for every resolution
        :param fade_in_percentage: list of percentages of epochs per resolution
                                   used for fading in the new layer
                                   not used for first resolution, but dummy value still needed.
        :param num_samples: number of samples generated in sample_sheet. def=36
        :param start_depth: start training from this depth. def=0
        :param num_workers: number of workers for reading the data. def=3
        :param feedback_factor: number of logs per epoch. def=100
        :param log_dir: directory for saving the loss logs. def="./models/"
        :param sample_dir: directory for saving the generated samples. def="./samples/"
        :param checkpoint_factor: save model after these many epochs.
                                  Note that only one model is stored per resolution.
                                  during one resolution, the checkpoint will be updated (Rewritten)
                                  according to this factor.
        :param save_dir: directory for saving the models (.pth files)
        :return: None (Writes multiple files to disk)
        """

        assert self.depth == len(batch_sizes), "batch_sizes not compatible with depth"

        #RELOAD
        gen_save_file = os.path.join(save_dir, "GAN_GEN_" + str(start_depth) + ".pth")

        dis_save_file = os.path.join(save_dir, "GAN_DIS_" + str(start_depth) + ".pth")
        gen_optim_save_file = os.path.join(save_dir,
                                            "GAN_GEN_OPTIM_" + str(start_depth)
                                            + ".pth")
        dis_optim_save_file = os.path.join(save_dir,
                                            "GAN_DIS_OPTIM_" + str(start_depth)
                                            + ".pth")
        is_reload = False
        if exists(gen_save_file):
            print("--reload model--")
            is_reload = True
 
        if is_reload:
            self.gen.load_state_dict(th.load(gen_save_file))
            self.dis.load_state_dict(th.load(dis_save_file))
            self.gen_optim.load_state_dict(th.load(gen_optim_save_file))
            self.dis_optim.load_state_dict(th.load(dis_optim_save_file))

        # # also save the shadow generator if use_ema is True
        if self.use_ema:
            gen_shadow_save_file = os.path.join(save_dir, "GAN_GEN_SHADOW_" +
                                                str(start_depth) + ".pth")
            if is_reload:
                self.gen_shadow.load_state_dict(th.load(gen_shadow_save_file))
        #RELOAD

        # turn the generator and discriminator into train mode
        self.gen.train()
        self.dis.train()
        if self.use_ema:
            self.gen_shadow.train()

        # create a global time counter
        global_time = time.time()
        current_seed = th.initial_seed()
        th.manual_seed(20171614)
        # create fixed_input for debugging
        fixed_input = th.randn(num_samples, self.latent_size).to(self.device)
        fixed_label = th.as_tensor([i%self.num_classes for i in range(num_samples)]).to(self.device)
        th.manual_seed(current_seed)
        print("seed: ", current_seed)
        print("Starting the training process ... ")
        for current_depth in range(start_depth, self.depth):


            print("\n\nCurrently working on Depth: ", current_depth)
            current_res = np.power(2, current_depth + 2)
            print("Current resolution: %d x %d" % (current_res, current_res))

            data = get_data_loader(dataset, batch_sizes[current_depth], num_workers)
            if (start_epoch > 1): ticker = len(range(1,start_epoch))* len(iter(data))
            else: ticker = 1
            print("Ticker", ticker)

            if start_depth != current_depth:
                start_epoch = 1

            for epoch in range(start_epoch, epochs[current_depth] + 1):
                start = timeit.default_timer()  # record time at the start of epoch

                print("\nEpoch: %d" % epoch)
                total_batches = len(iter(data))

                fader_point = int((fade_in_percentage[current_depth] / 100)
                                  * epochs[current_depth] * total_batches)

                step = 0  # counter for number of iterations

                for (i, batch) in enumerate(data, 1):
                    # calculate the alpha for fading in the layers
                    alpha = ticker / fader_point if ticker <= fader_point else 1

                    # extract current batch of data for training
                    images = batch[0].to(self.device)
                    labels = batch[1].to(self.device)

                    gan_input = th.randn(images.shape[0], self.latent_size).to(self.device)

                    # optimize the discriminator:
                    dis_loss = self.optimize_discriminator(gan_input, labels, images,
                                                           current_depth, alpha)

                    # optimize the generator:
                    gen_loss = self.optimize_generator(gan_input, labels, images,  current_depth, alpha)

                    # provide a loss feedback
                    if i % int(total_batches / feedback_factor) == 0 or i == 1:
                        elapsed = time.time() - global_time
                        elapsed = str(datetime.timedelta(seconds=elapsed))
                        print("Elapsed: [%s]  batch: %d  d_loss: %f  g_loss: %f"
                              % (elapsed, i, dis_loss, gen_loss))

                        # also write the losses to the log file:
                        os.makedirs(log_dir, exist_ok=True)
                        log_file = os.path.join(log_dir, "loss_" + str(current_depth) + ".log")
                        with open(log_file, "a") as log:
                            log.write(str(step) + "\t" + str(dis_loss) +
                                      "\t" + str(gen_loss) + "\n")
                    # increment the alpha ticker and the step
                    ticker += 1
                    step += 1
                # create a grid of samples and save it
                os.makedirs(sample_dir, exist_ok=True)
                gen_img_file = os.path.join(sample_dir, "gen_" + str(current_depth) +
                                            "_" + str(epoch) + "_" +
                                            str(i) + ".png")

                # this is done to allow for more GPU space
                with th.no_grad():
                    self.create_grid(
                        samples=self.gen(
                            fixed_input,fixed_label,
                            current_depth,
                            alpha
                        ).detach() if not self.use_ema
                        else self.gen_shadow(
                            fixed_input, fixed_label,
                            current_depth,
                            alpha
                        ).detach(),
                        scale_factor=int(np.power(2, self.depth - current_depth - 1)),
                        img_file=gen_img_file,
                    )

                stop = timeit.default_timer()
                print("Time taken for epoch: %.3f secs" % (stop - start))
                print("ticker = ", ticker)
                if epoch % checkpoint_factor == 0 or epoch == 1 or epoch%5 ==0 or epoch == epochs[current_depth]:
                    os.makedirs(save_dir, exist_ok=True)
                    gen_save_file = os.path.join(save_dir, "GAN_GEN_" + str(current_depth) + ".pth")
                    dis_save_file = os.path.join(save_dir, "GAN_DIS_" + str(current_depth) + ".pth")
                    gen_optim_save_file = os.path.join(save_dir,
                                                       "GAN_GEN_OPTIM_" + str(current_depth)
                                                       + ".pth")
                    dis_optim_save_file = os.path.join(save_dir,
                                                       "GAN_DIS_OPTIM_" + str(current_depth)
                                                       + ".pth")

                    th.save(self.gen.state_dict(), gen_save_file)
                    th.save(self.dis.state_dict(), dis_save_file)
                    th.save(self.gen_optim.state_dict(), gen_optim_save_file)
                    th.save(self.dis_optim.state_dict(), dis_optim_save_file)

                    # also save the shadow generator if use_ema is True
                    if self.use_ema:
                        gen_shadow_save_file = os.path.join(save_dir, "GAN_GEN_SHADOW_" +
                                                            str(current_depth) + ".pth")
                        th.save(self.gen_shadow.state_dict(), gen_shadow_save_file)

        # put the gen, shadow_gen and dis in eval mode
        self.gen.eval()
        self.dis.eval()
        if self.use_ema:
            self.gen_shadow.eval()

        print("Training completed ...")
        
    def reload(self, start_depth = 0, save_dir = './models/'):
        gen_save_file = os.path.join(save_dir, "GAN_GEN_" + str(start_depth) + ".pth")
        dis_save_file = os.path.join(save_dir, "GAN_DIS_" + str(start_depth) + ".pth")
        gen_optim_save_file = os.path.join(save_dir,
                                        "GAN_GEN_OPTIM_" + str(start_depth)
                                        + ".pth")
        dis_optim_save_file = os.path.join(save_dir,
                                        "GAN_DIS_OPTIM_" + str(start_depth)
                                        + ".pth")
        self.gen.load_state_dict(th.load(gen_save_file))
        self.dis.load_state_dict(th.load(dis_save_file))
        self.gen_optim.load_state_dict(th.load(gen_optim_save_file))
        self.dis_optim.load_state_dict(th.load(dis_optim_save_file))
        if self.use_ema:
            gen_shadow_save_file = os.path.join(save_dir, "GAN_GEN_SHADOW_" +
                                            str(start_depth) + ".pth")
            self.gen_shadow.load_state_dict(th.load(gen_shadow_save_file))
            
    def print(self, input, labels, depth, alpha):
        with th.no_grad():
            return (self.gen(input, labels, depth, alpha ).detach() 
                    if not self.use_ema else 
                        self.gen_shadow(
                            input,
                            labels,
                            depth,
                            alpha
                        ).detach())