from torchvision import transforms
from PIL import Image


def normalize_image(img, scaling=None):
    list_of_transformations = []
    if scaling:
        list_of_transformations.append(transforms.Resize(scaling))

    list_of_transformations.append(transforms.ToTensor())
    list_of_transformations.append(transforms.Normalize(
        [0.67040706, 0.67040706, 0.67040706],
        [0.23386002, 0.23386002, 0.23386002],
        )
    )

    normalized = transforms.Compose(list_of_transformations)(Image.fromarray(img).convert('RGB'))

    return normalized


def generate_means_and_stds(path):
    from skimage import io
    from glob import glob
    import numpy as np
    img_list = glob(path)
    means = []
    stds = []
    for i in img_list:
        img = io.imread(i)
        t_img = transforms.ToTensor()(np.asarray(Image.fromarray(np.asarray(img)).convert('RGB')))
        np_t_img = t_img.numpy()
        means.append([np.mean(np_t_img[i, :, :]) for i in range(np_t_img.shape[0])])
        stds.append([np.std(np_t_img[i, :, :]) for i in range(np_t_img.shape[0])])

    means = np.asarray(means)
    stds = np.asarray(stds)
    print(np.mean(means, axis=0))
    print(np.mean(stds, axis=0))


if __name__ == '__main__':
    generate_means_and_stds('../pngs/*/*1*')
