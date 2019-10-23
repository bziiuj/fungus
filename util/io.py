import imageio


def read_img(path):
    assert path.suffix == '.png', path.suffix
    img = imageio.imread(path, 'PNG-FI')
    assert img.dtype == 'uint16'
    # ensure image is RGB
    assert img.shape[2] == 3, img.shape
    img = img / (2 ** 16)
    return img


def read_mask(path):
    assert path.suffix == '.png', path.suffix
    mask = imageio.imread(path, 'PNG-FI')
    # ensure mask has correct content
    assert mask.max() == 2
    assert mask.min() == 0
    return mask
