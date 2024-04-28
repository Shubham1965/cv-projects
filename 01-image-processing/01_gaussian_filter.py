# Learned how to implement gaussian filter using scipy.ndimage.convolve and from scratch

import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio
from scipy.ndimage import convolve, correlate

def plot_multiple(images, titles, colormap='gray', max_columns=np.inf, share_axes=True):
    """Plot multiple images as subplots on a grid."""
    assert len(images) == len(titles)
    n_images = len(images)
    n_cols = min(max_columns, n_images)
    n_rows = int(np.ceil(n_images / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4),
        squeeze=False, sharex=share_axes, sharey=share_axes)

    axes = axes.flat
    # Hide subplots without content
    for ax in axes[n_images:]:
        ax.axis('off')
        
    if not isinstance(colormap, (list,tuple)):
        colormaps = [colormap]*n_images
    else:
        colormaps = colormap

    for ax, image, title, cmap in zip(axes, images, titles, colormaps):
        ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        
    fig.tight_layout()
    plt.show()

# define a 1D gaussian kernel: 
def gauss(x, sigma):
    g = np.exp(-x**2/(2*sigma**2)) / (np.sqrt(2*np.pi*sigma**2))
    return g

def gaussian_filter_scratch(image, sigma, padding):

    """Returns the Gaussian filtered image."""

    # Generate 1D kernel
    ks_half = int(np.ceil(3 * sigma))
    x = np.linspace(-ks_half, ks_half, 2*ks_half + 1, dtype=float)
    kernel = gauss(x, sigma)
    kernel = kernel / np.sum(kernel)

    # Expand kernel to handle 3-channel image
    kernel = np.expand_dims(kernel, axis=-1)

    # Add border
    if padding:
        image = cv2.copyMakeBorder(
            image, ks_half, ks_half, ks_half, ks_half,
            cv2.BORDER_DEFAULT)

    # Create an image to store intermediate result
    # of the row-wise filtering
    image_tmp = np.empty_like(image)
    image_result = np.empty_like(image)

    # Apply row filter
    for i in range(image.shape[0]):
        for j in range(ks_half, image.shape[1] - ks_half):
            image_roi = image[i, j-ks_half:j+ks_half+1]
            image_tmp[i, j] = np.sum(image_roi * kernel, axis=0)

    # Apply column filter
    for i in range(ks_half, image.shape[0] - ks_half):
        for j in range(image.shape[1]):
            image_roi = image_tmp[i-ks_half:i+ks_half+1, j]
            image_result[i, j] = np.sum(image_roi * kernel, axis=0)

    # Remove previously added border
    image_result = image_result[ks_half:-ks_half, ks_half:-ks_half]
    ### END SOLUTION
    return image_result


def gaussian_filter_scipy(image, sigma):
    """Returns the Gaussian filtered image."""

    # Generate 1D kernel
    ks_half = int(np.ceil(3 * sigma))
    x = np.linspace(-ks_half, ks_half, 2*ks_half + 1, dtype=float)
    kernel = gauss(x, sigma)
    kernel = kernel / np.sum(kernel)

    # Expand kernel to handle 3-channel image
    kernel = np.expand_dims(kernel, axis=-1)

    # Using scipy.ndimage.convolve():
    # Convolve the kernel along row:
    filtered_image_row = [convolve(image[:,:,i], kernel, mode='constant', cval=0.0) for i in range(image.shape[-1])]
    filtered_image_row_combined = np.stack(filtered_image_row, axis=2)

    # Convolve the kernel along column:
    filtered_image_col = [convolve(filtered_image_row_combined[:,:,i], kernel.T, mode='constant', cval=0.0) for i in range(image.shape[-1])]
    filtered_image = np.stack(filtered_image_col, axis=2)

    return filtered_image

if __name__ == '__main__':
    
    # testing the gaussian filter:
    x = np.linspace(-5, 5, 100)
    y = gauss(x, sigma=2)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    fig.tight_layout()
    plt.show()
    
    # testing the gaussian filters build from scratch and scipy.ndimage on an image:
    image = imageio.v2.imread('01-image-processing/images/graf_small.png')

    sigmas = [2, 4, 8]
    blurred_images = [gaussian_filter_scipy(image, s) for s in sigmas]
    titles = [f'sigma={s}' for s in sigmas]
    plot_multiple(blurred_images, titles)

    blurred_images = [gaussian_filter_scratch(image, s, padding=True) for s in sigmas]
    plot_multiple(blurred_images, titles)

    



