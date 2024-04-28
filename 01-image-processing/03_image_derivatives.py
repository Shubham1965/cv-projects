import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from scipy import ndimage

def convolve_with_two(image, kernel1, kernel2):
    """Apply two filters, one after the other."""
    image = ndimage.convolve(image, kernel1)
    image = ndimage.convolve(image, kernel2)   
    return image

def imread_gray(filename):
    """Read grayscale image."""
    return cv2.imread(filename, cv2.IMREAD_GRAYSCALE).astype(np.float32)

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

def gauss(x, sigma):
    g = np.exp(-x**2/(2*sigma**2)) / (np.sqrt(2*np.pi*sigma**2))
    return g

def gaussdx(x, sigma):
    
    ks_half = int(np.ceil(3*sigma))
    x = np.linspace(-ks_half, ks_half, 2*ks_half + 1, dtype=float)
    gauss_kernel = gauss(x, sigma)
    gauss_kernel = gauss_kernel / np.sum(gauss_kernel)
    gauss_kernel = np.expand_dims(gauss_kernel, axis=-1).T

    gdx = ndimage.convolve(gauss_kernel, np.array([[1,0,-1]]), mode='constant', cval=0.0)
    return gdx

def gauss_derivs(image, sigma):

    ks_half = int(np.ceil(3*sigma))
    x = np.linspace(-ks_half, ks_half, 2*ks_half + 1, dtype=float)

    image_dx = ndimage.convolve(image, gaussdx(x,sigma), mode='constant', cval=0.0)
    image_dy = ndimage.convolve(image, gaussdx(x,sigma).T, mode='constant', cval=0.0)
    return image_dx, image_dy

def gauss_second_derivs(image, sigma):
    kernel_radius = int(3.0 * sigma)
    ks_half = int(np.ceil(kernel_radius))
    x = np.linspace(-ks_half, ks_half, 2*ks_half + 1, dtype=float)

    image_dx, image_dy = gauss_derivs(image, sigma)
    
    image_dxx = ndimage.convolve(image_dx, gaussdx(x,sigma), mode='constant', cval=0.0)
    image_dxy = ndimage.convolve(image_dy, gaussdx(x, sigma), mode='constant', cval=0.0)
    image_dyy = ndimage.convolve(image_dy, gaussdx(x, sigma).T, mode='constant', cval=0.0)
    
    return image_dxx, image_dxy, image_dyy


def image_gradients_polar(image, sigma):

    image_dx, image_dy = gauss_derivs(image, sigma)
    magnitude = np.sqrt(image_dx**2 + image_dy**2)
    direction = np.arctan2(image_dy, image_dx)

    return magnitude, direction

def laplace(image, sigma):
    image_dxx, _, image_dyy = gauss_second_derivs(image, sigma)
    return image_dxx + image_dyy

if __name__ == '__main__':

    # Create an test immage with only the center pixel set to high value and rest 0
    impulse = np.zeros((64, 64))
    impulse[32, 32] = 1.0

    sigma = 6.0
    kernel_radius = int(3.0 * sigma)
    x = np.arange(-kernel_radius, kernel_radius + 1)[np.newaxis]
    G = gauss(x, sigma)
    D = gaussdx(x, sigma)

    images = [
        impulse,
        convolve_with_two(impulse, G, G.T),
        convolve_with_two(impulse, G, D.T),
        convolve_with_two(impulse, D, G.T),
        convolve_with_two(impulse, G.T, D),
        convolve_with_two(impulse, D.T, G)]

    titles = [
        'original',
        'first G, then G^T',
        'first G, then D^T',
        'first D, then G^T',
        'first G^T, then D',
        'first D^T, then G']

    plot_multiple(images, titles, max_columns=3)

    # Testing the first derivatives:
    image = imread_gray('01-image-processing/images/tomatoes.png')
    grad_dx, grad_dy = gauss_derivs(image, sigma=5.0)
    plot_multiple([image, grad_dx, grad_dy], ['Image', 'Derivative in x-direction', 'Derivative in y-direction'])

    # Testing the second derivatives:
    image = imread_gray('01-image-processing/images/coins1.jpg')
    grad_dxx, grad_dxy, grad_dyy = gauss_second_derivs(image, sigma=2.0)
    plot_multiple([image, grad_dxx, grad_dxy, grad_dyy], ['Image', 'Dxx', 'Dxy','Dyy'])

    image = imread_gray('01-image-processing/images/circuit.png')
    grad_dxx, grad_dxy, grad_dyy = gauss_second_derivs(image, sigma=2.0)
    plot_multiple([image, grad_dxx, grad_dxy, grad_dyy], ['Image', 'Dxx', 'Dxy','Dyy'])

    # Testing the magnitude and direction: (# Note: the twilight colormap only works since Matplotlib 3.0, use 'gray' in earlier versions.)
    image = imread_gray('01-image-processing/images/coins1.jpg')
    grad_mag, grad_dir = image_gradients_polar(image, sigma=2.0)
    plot_multiple([image, grad_mag, grad_dir], ['Image', 'Magnitude', 'Direction'], colormap=['gray', 'gray', 'twilight'])

    image = imread_gray('01-image-processing/images/circuit.png')
    grad_mag, grad_theta = image_gradients_polar(image, sigma=2.0)
    plot_multiple([image, grad_mag, grad_theta], ['Image', 'Magnitude', 'Direction'], colormap=['gray', 'gray', 'twilight'])

    # Testing the Laplacian:
    image = imread_gray('01-image-processing/images/coins1.jpg')
    lap = laplace(image, sigma=2.0)
    plot_multiple([image, lap], ['Image', 'Laplace'])

    image = imread_gray('01-image-processing/images/circuit.png')
    lap = laplace(image, sigma=2.0)
    plot_multiple([image, lap], ['Image', 'Laplace'])

