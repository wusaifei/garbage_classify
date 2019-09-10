from keras.models import Model
from keras.layers import Input

from .layers import Repeat, TTA, Merge
from .augmentation import Augmentation


doc = """
    IMPORTANT constraints:
        1) model has to have 1 input and 1 output
        2) inference batch_size = 1
        3) image height == width if rotate augmentation is used

    Args:
        model: instance of Keras model
        h_flip: (bool) horizontal flip
        v_flip: (bool) vertical flip
        h_shifts: (list of int) list of horizontal shifts (e.g. [10, -10])
        v_shifts: (list of int) list of vertical shifts (e.g. [10, -10])
        rotation: (list of int) list of angles (deg) for rotation in range [0, 360),
            should be divisible by 90 deg (e.g. [90, 180, 270])
        contrast: (list of float) values for contrast adjustment
        add: (list of int or float) values to add on image (e.g. [-10, 10])
        mul: (list of float) values to multiply image on (e.g. [0.9, 1.1])
        merge: one of 'mean', 'gmean' and 'max' - mode of merging augmented
            predictions together.

    Returns:
        Keras Model instance

"""

def tta_segmentation(model,
                     h_flip=False,
                     v_flip=False,
                     h_shift=None,
                     v_shift=None,
                     rotation=None,
                     contrast=None,
                     add=None,
                     mul=None,
                     merge='mean'):

    """
    Segmentation model test time augmentation wrapper.
    """
    tta = Augmentation(h_flip=h_flip,
                       v_flip=v_flip,
                       h_shift=h_shift,
                       v_shift=v_shift,
                       rotation=rotation,
                       contrast=contrast,
                       add=add,
                       mul=mul,
                       )

    input_shape = (1, *model.input.shape.as_list()[1:])

    inp = Input(batch_shape=input_shape)
    x = Repeat(tta.n_transforms)(inp)
    x = TTA(*tta.forward)(x)
    x = model(x)
    x = TTA(*tta.backward)(x)
    x = Merge(merge)(x)
    tta_model = Model(inp, x)

    return tta_model


def tta_classification(model,
                       h_flip=False,
                       v_flip=False,
                       h_shift=None,
                       v_shift=None,
                       rotation=None,
                       contrast=None,
                       add=None,
                       mul=None,
                       merge='mean'):
    """
    Classification model test time augmentation wrapper.
    """

    tta = Augmentation(h_flip=h_flip,
                       v_flip=v_flip,
                       h_shift=h_shift,
                       v_shift=v_shift,
                       rotation=rotation,
                       contrast=contrast,
                       add=add,
                       mul=mul,
                       )

    input_shape = (1, *model.input.shape.as_list()[1:])

    inp = Input(batch_shape=input_shape)
    x = Repeat(tta.n_transforms)(inp)
    x = TTA(*tta.forward)(x)
    x = model(x)
    x = Merge(merge)(x)
    tta_model = Model(inp, x)

    return tta_model


tta_classification.__doc__ += doc
tta_segmentation.__doc__ += doc
