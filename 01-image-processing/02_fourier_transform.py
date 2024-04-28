# Implemented a gaussian and a box filter, saw the difference between them, 
# and checked how to downsample an image without gaps and with gaps and how gaussian filter helps us to avoid aliasing/artifacts. 

import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.signal
import numpy as np
from scipy import ndimage
import cv2

def imread_gray(filename):
    """Read grayscale image from our data directory."""
    return cv2.imread(filename, cv2.IMREAD_GRAYSCALE).astype(np.float32)

def convolve_with_two(image, kernel1, kernel2):
    """Apply two filters, one after the other."""
    image = ndimage.convolve(image, kernel1, mode='wrap')
    image = ndimage.convolve(image, kernel2, mode='wrap')
    return image

def fourier_spectrum(im):
    normalized_im = im / np.sum(im)
    f = np.fft.fft2(normalized_im)
    return np.fft.fftshift(f)

def log_magnitude_spectrum(im):
    return np.log(np.abs(fourier_spectrum(im))+1e-8)

def plot_with_spectra(images, titles):
    """Plots a list of images in the first column and the logarithm of their
    magnitude spectrum in the second column."""
    
    assert len(images) == len(titles)
    n_cols = 2
    n_rows = len(images)
    fig, axes = plt.subplots(
        n_rows, 2, figsize=(n_cols * 4, n_rows * 4),
        squeeze=False)

    spectra = [log_magnitude_spectrum(im) for im in images]

    lower = min(np.percentile(s, 0.1) for s in spectra)
    upper = min(np.percentile(s, 99.999) for s in spectra)
    normalizer = mpl.colors.Normalize(vmin=lower, vmax=upper)
    
    for ax, image, spectrum, title in zip(axes, images, spectra, titles):
        ax[0].imshow(image, cmap='gray')
        ax[0].set_title(title)
        ax[0].set_axis_off()
        c = ax[1].imshow(spectrum, norm=normalizer, cmap='viridis')
        ax[1].set_title('Log magnitude spectrum')
        ax[1].set_axis_off()
        
    fig.tight_layout()
    plt.show()


def generate_pattern():
    """
    Generates a pattern using sinusoidal functions.

    Returns:
        numpy.ndarray: The generated pattern.
    """
    x = np.linspace(0, 1, 256, endpoint=False)
    y = np.sin(x**2 * 16 * np.pi)
    return np.outer(y,y)/2+0.5

def gauss(x, sigma):
    """
    Calculate the Gaussian function value for a given input `x` and standard deviation `sigma`.

    Parameters:
        x (float): The input value.
        sigma (float): The standard deviation.

    Returns:
        float: The calculated Gaussian function value.
    """

    g = np.exp(-x**2/(2*sigma**2)) / (np.sqrt(2*np.pi*sigma**2))
    return g


def filter_gauss(image, kernel_factor, sigma):
    """
    Apply a Gaussian filter to an image using a specified kernel factor and sigma.

    Parameters:
        image (numpy.ndarray): The input image to be filtered.
        kernel_factor (int): The factor to determine the size of the kernel. 
                             This parameter defines half of the kernel size.
        sigma (float): The standard deviation of the Gaussian distribution.

    Returns:
        numpy.ndarray: The filtered image.
    """
    # kernel_factor = 3 for the best results, but this is a parameter which defines half of the kernel size

    # Generate 1D kernel
    ks_half = int(np.ceil(kernel_factor * sigma))
    x = np.linspace(-ks_half, ks_half, 2*ks_half + 1, dtype=float)
    kernel = gauss(x, sigma)
    kernel = kernel / np.sum(kernel)

    # Expand kernel to handle 3-channel image
    kernel = np.expand_dims(kernel, axis=-1)
    kernel2D = np.matmul(kernel, kernel.T)

    filtered_image = ndimage.convolve(image, kernel, mode='constant', cval=0.0)

    return filtered_image

def sample_with_gaps(im, period):
    """
    Sample an image at equal intervals along rows and columns.

    Parameters:
    - im: Input image (numpy array).
    - period: Sampling period.

    Returns:
    - sampled_im: Sampled image.
    """
    # Get the shape of the input image, its a single channel image, if it a 4 channel RGB image, the shape will be (rows, cols, channels)
    rows, cols = im.shape 

    # Initialize an empty array for the sampled image
    sampled_im = np.zeros_like(im)

    # Sample along rows
    for i in range(0, rows, period):
        # Sample along columns within the current row
        for j in range(0, cols, period):
            # Copy the pixel values to the sampled image
            sampled_im[i:i+period, j:j+period] = im[i, j]

    return sampled_im

def sample_without_gaps(im, period):
    """
    Downsample an image without gaps.

    Parameters:
    - im: Input image (numpy array).
    - period: Downsampling period.

    Returns:
    - downsampled_im: Downsampled image.
    """
    # Use numpy slicing with stride to downsample the image
    downsampled_im = im[::period, ::period]

    return downsampled_im


def filter_box(image, sigma):
    side = int(round(sigma * 12**0.5))
    B = np.full([1, side], 1/side)
    B_2d = scipy.signal.convolve2d(B, B.T, mode='full')
    return scipy.signal.convolve2d(image, B_2d, mode='same', boundary='wrap')

if __name__ == '__main__':
    
    im_grass = imread_gray('01-image-processing/images/grass.jpg')
    im_zebras = imread_gray('01-image-processing/images/zebras.jpg')
    im_pattern = generate_pattern()
    plot_with_spectra([im_grass, im_zebras, im_pattern], ['Grass image', 'Zebra image', 'Pattern image'])

    sigma = 3
    im = im_grass
    gauss_filtered = filter_gauss(im, kernel_factor=3, sigma=sigma)
    box_filtered = filter_box(im, sigma=sigma)
    plot_with_spectra([im, box_filtered, gauss_filtered], ['Image', 'Box filtered', 'Gauss filtered'])

    N=4
    im = im_zebras
    sampled_gaps = sample_with_gaps(im, N)
    sampled = sample_without_gaps(im, N)

    blurred = filter_gauss(im, kernel_factor=6, sigma=4)
    blurred_and_sampled_gaps = sample_with_gaps(blurred, N)
    blurred_and_sampled = sample_without_gaps(blurred, N)

    plot_with_spectra([im, sampled_gaps, sampled], ['Original', 'Sampled (w/ gaps)', 'Sampled'])
    plot_with_spectra([blurred, blurred_and_sampled_gaps, blurred_and_sampled], ['Gauss blurred', 'Blurred and s. (w/ gaps)', 'Blurred and s.'])

    # Testing on the pattern image to check at what values of sigma to avoid alaising, i.e., avoid artifacts
    N=4
    s = 2

    image = im_pattern
    downsampled_gaps = sample_with_gaps(im_pattern, N)
    downsampled = sample_without_gaps(im_pattern, N)

    blurred = filter_gauss(image, kernel_factor=3, sigma=s)
    blurred_and_downsampled_gaps = sample_with_gaps(blurred, N)
    blurred_and_downsampled = sample_without_gaps(blurred, N)

    plot_with_spectra([im_pattern, downsampled_gaps, downsampled], ['Original', 'Downsampled (w/ gaps)', 'Downsampled (no gaps)'])
    plot_with_spectra([blurred, blurred_and_downsampled_gaps, blurred_and_downsampled], ['Gauss blurred', 'Blurred and ds. (w/ gaps)', 'Blurred and downs. (no gaps)'])