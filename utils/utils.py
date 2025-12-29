import torch

NO_PATCH = 0
TOP_LEFT = 1
TOP_RIGHT = 2
BOTTOM_LEFT = 3
BOTTOM_RIGHT = 4


def add_one_patch_mnist(image, patch_location, patch_size=5):
    """ 
    Add patch to single MNIST image.
    
    Returns:
        torch.Tensor: The modified image
    """
    assert image.shape == (28, 28), f'Invalid image shape: {image.shape}'
    image = image.clone().detach() 
    patch = torch.ones(patch_size, patch_size)
    q, r = patch_size, 28 - patch_size

    if patch_location == NO_PATCH:
        pass 
    elif patch_location == TOP_LEFT:
        image[:q, :q] = patch
    elif patch_location == TOP_RIGHT:
        image[:q, r:] = patch
    elif patch_location == BOTTOM_LEFT:
        image[r:, :q] = patch
    elif patch_location == BOTTOM_RIGHT:
        image[r:, r:] = patch
    else:
        raise ValueError(f'Invalid {patch_location=}')
    return image