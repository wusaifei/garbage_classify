import tensorflow as tf


class DualTransform:

    identity_param = None

    def prepare(self, params):
        if isinstance(params, tuple):
            params = list(params)
        elif params is None:
            params = []
        elif not isinstance(params, list):
            params = [params]

        if not self.identity_param in params:
            params.append(self.identity_param)
        return params

    def forward(self, image, param):
        raise NotImplementedError

    def backward(self, image, param):
        raise NotImplementedError


class SingleTransform(DualTransform):

    def backward(self, image, param):
        return image


class HFlip(DualTransform):

    identity_param = 0

    def prepare(self, params):
        if params == False:
            return [0]
        if params == True:
            return [1, 0]

    def forward(self, image, param):
        return tf.image.flip_left_right(image) if param else image

    def backward(self, image, param):
        return self.forward(image, param)


class VFlip(DualTransform):

    identity_param = 0

    def prepare(self, params):
        if params == False:
            return [0]
        if params == True:
            return [1, 0]

    def forward(self, image, param):
        return tf.image.flip_up_down(image) if param else image

    def backward(self, image, param):
        return self.forward(image, param)


class Rotate(DualTransform):

    identity_param = 0

    def forward(self, image, angle):
        k = angle // 90 if angle >= 0 else (angle + 360) // 90
        return tf.image.rot90(image, k)

    def backward(self, image, angle):
        return self.forward(image, -angle)


class HShift(DualTransform):

    identity_param = 0

    def forward(self, image, param):
        return tf.manip.roll(image, param, axis=0)

    def backward(self, image, param):
        return tf.manip.roll(image, -param, axis=0)


class VShift(DualTransform):

    identity_param = 0

    def forward(self, image, param):
        return tf.manip.roll(image, param, axis=1)

    def backward(self, image, param):
        return tf.manip.roll(image, -param, axis=1)


class Contrast(SingleTransform):

    identity_param = 1

    def forward(self, image, param):
        return tf.image.adjust_contrast(image, param)


class Add(SingleTransform):

    identity_param = 0

    def forward(self, image, param):
        return image + param


class Multiply(SingleTransform):

    identity_param = 1

    def forward(self, image, param):
        return image * param


def gmean(x):
    g_pow = 1 / x.get_shape().as_list()[0]
    x = tf.reduce_prod(x, axis=0, keepdims=True)
    x = tf.pow(x, g_pow)
    return x


def mean(x):
    return tf.reduce_mean(x, axis=0, keepdims=True)


def max(x):
    return tf.reduce_max(x, axis=0, keepdims=True)
