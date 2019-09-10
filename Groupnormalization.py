import os

os.environ["KERAS_BACKEND"] = "tensorflow"

from keras.engine import Layer, InputSpec
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import backend as K

from keras.utils.generic_utils import get_custom_objects


class GroupNormalization(Layer):
    """Group normalization layer

    Group Normalization divides the channels into groups and computes within each group
    the mean and variance for normalization. GN's computation is independent of batch sizes,
    and its accuracy is stable in a wide range of batch sizes

    # Arguments
        groups: Integer, the number of groups for Group Normalization.
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `BatchNormalization`.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.

    # References
        - [Group Normalization](https://arxiv.org/abs/1803.08494)
    """

    def __init__(self,
                 groups=32,
                 axis=-1,
                 momentum=0.99,
                 epsilon=1e-5,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(GroupNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = initializers.get(moving_variance_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        dim = input_shape[self.axis]

        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                                                        'input tensor should have a defined dimension '
                                                        'but the layer received an input with shape ' +
                             str(input_shape) + '.')

        if dim < self.groups:
            raise ValueError('Number of groups (' + str(self.groups) + ') cannot be '
                                                                       'more than the number of channels (' +
                             str(dim) + ').')

        if dim % self.groups != 0:
            raise ValueError('Number of groups (' + str(self.groups) + ') must be a '
                                                                       'multiple of the number of channels (' +
                             str(dim) + ').')

        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        shape_ = (1, dim, 1, 1)
        shape = (self.groups,)
        broadcast_shape = [-1, self.groups, 1, 1, 1]

        if self.scale:
            self.gamma = self.add_weight(shape=shape_,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)

        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape_,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)

        else:
            self.beta = None

        self.moving_mean = self.add_weight(
            shape=shape,
            name="moving_mean",
            initializer=self.moving_mean_initializer,
            trainable=False)
        self.moving_mean = K.reshape(self.moving_mean, broadcast_shape)
        self.moving_mean = K.variable(value=self.moving_mean)

        self.moving_variance = self.add_weight(
            shape=shape,
            name="moving_variance",
            initializer=self.moving_variance_initializer,
            trainable=False)
        self.moving_variance = K.reshape(self.moving_variance, broadcast_shape)
        self.moving_variance = K.variable(value=self.moving_variance)

        self.built = True

    def call(self, inputs, training=None, **kwargs):

        G = self.groups

        # transpose:[ba,h,w,c] -> [bs,c,h,w]
        if self.axis in {-1, 3}:
            inputs = K.permute_dimensions(inputs, (0, 3, 1, 2))

        input_shape = K.int_shape(inputs)
        N, C, H, W = input_shape
        inputs = K.reshape(inputs, (-1, G, C // G, H, W))
        # inputs.assign_sub()

        # compute group-channel mean & variance
        gn_mean = K.mean(inputs, axis=[2, 3, 4], keepdims=True)
        gn_variance = K.var(inputs, axis=[2, 3, 4], keepdims=True)

        # compute group-normalization in different state
        def gn_inference():
            # when in test phase, just return moving_mean & moving_var
            mean, variance = self.moving_mean, self.moving_variance
            outputs = (inputs - mean) / (K.sqrt(variance + self.epsilon))
            outputs = K.reshape(outputs, [-1, C, H, W]) * self.gamma + self.beta
            # transpose: [bs,c,h,w] -> [ba,h,w,c]
            if self.axis in {-1, 3}:
                outputs = K.permute_dimensions(outputs, (0, 2, 3, 1))

            return outputs

        if training in {0, False}:
            return gn_inference()

        outputs = (inputs - gn_mean) / (K.sqrt(gn_variance + self.epsilon))
        outputs = K.reshape(outputs, [-1, C, H, W]) * self.gamma + self.beta

        # transpose: [bs,c,h,w] -> [ba,h,w,c]
        if self.axis in {-1, 3}:
            outputs = K.permute_dimensions(outputs, (0, 2, 3, 1))

        self.add_update([K.moving_average_update(self.moving_mean,
                                                 gn_mean,
                                                 self.momentum),
                         K.moving_average_update(self.moving_variance,
                                                 gn_variance,
                                                 self.momentum)],
                        inputs)

        # print("moving_mean shape : ",K.int_shape(self.moving_mean))
        # print("moving_mean: ",K.eval(self.moving_mean))
        # print("moving_variance shape: ",K.int_shape(self.moving_variance))
        # print("moving_variance: ",K.eval(self.moving_variance))

        return K.in_train_phase(outputs,
                                gn_inference,
                                training=training)

    def get_config(self):
        config = {
            'groups': self.groups,
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'moving_mean_initializer': initializers.serialize(self.moving_mean_initializer),
            'moving_variance_initializer': initializers.serialize(self.moving_variance_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(GroupNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


get_custom_objects().update({'GroupNormalization': GroupNormalization})