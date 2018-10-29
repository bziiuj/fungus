from torchvision import transforms
from PIL import Image
import numpy as np


def normalize_image(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            [0.7328962, 0.73332304, 0.7587824],
            [0.24182673, 0.25079936, 0.21770595]
        ),
    ])
    return transform(img)


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
