from imgaug import augmenters as iaa
import imgaug as ia

def augumentor(image):
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential(
        [
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Affine(rotate=(-10, 10)),
            sometimes(iaa.Crop(percent=(0, 0.1), keep_size=True)),
        ],
        random_order=True
    )


    image_aug = seq.augment_image(image)

    return image_aug