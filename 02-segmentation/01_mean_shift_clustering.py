import numpy as np
import matplotlib.pyplot as plt
import time
import imageio
import skimage
from mpl_toolkits.mplot3d import Axes3D
import cv2
import os

def find_peak(data, query, radius):
    
    # Get the data points which are within the search radius:
    data_search_radius = data[np.linalg.norm(data - query, axis=1) <= radius]

    # Find the mean of the data points
    old_peak = np.mean(data_search_radius, axis=0)

    while True:
        # Get the data points which are within the search radius:
        data_search_radius = data[np.linalg.norm(data - old_peak, axis=1) <= radius]

        # Find the mean of the data points
        new_peak = np.mean(data_search_radius, axis=0)

        if np.linalg.norm(new_peak - old_peak) < 1e-5:
            break

        old_peak = new_peak

    return new_peak

def mean_shift(data, radius):
    labels = np.full(len(data), fill_value=-1, dtype=int)

    # Find the peak and label for the first data point:
    peaks = [find_peak(data, data[0], radius)]
    labels[0] = 0
    
    # Iterate over each data point: 
    for i in range(1,len(data)):

        next_peak = find_peak(data, data[i], radius)

        # if the next peak is too close to the current peak, assign it to the current peak
        if any(np.linalg.norm(np.array(peaks) - next_peak, axis=1) <= radius/2):
            labels[i] = np.where(np.linalg.norm(np.array(peaks) - next_peak, axis=1) <= radius/2)[0][0]
        else:
            peaks.append(next_peak)
            labels[i] = len(peaks)

    peaks = np.array(peaks)        
    return peaks, labels


# Speedup - 1
def mean_shift_opt(data, radius):
    labels = np.full(len(data), fill_value=-1, dtype=int)
    
    # Find the peak and label for the first data point:
    peaks = [find_peak(data, data[0], radius)]
    labels[0] = 0

    # data points within the radius of the peak must get the same label as the peak
    labels[np.where(np.linalg.norm(data - peaks[0], axis=1) <= radius)] = 0
    
    # Iterate over each data point: 
    for i in range(1,len(data)):

        if labels[i] == -1:
            next_peak = find_peak(data, data[i], radius)

            # if the next peak is too close to the current peak, assign it to the current peak
            if any(np.linalg.norm(np.array(peaks) - next_peak, axis=1) <= radius/2):
                labels[i] = np.where(np.linalg.norm(np.array(peaks) - next_peak, axis=1) <= radius/2)[0][0]
            else:
                peaks.append(next_peak)
                labels[i] = len(peaks)

                # data points within the radius of the peak must get the same label as the peak
                labels[np.where(np.linalg.norm(data - next_peak, axis=1) <= radius)] = labels[i]
        else:
            continue

    peaks = np.array(peaks) 

    return peaks, labels

# Speedup - 2
def find_peak_opt(data, query, radius, c=3):
    is_near_search_path = np.zeros(len(data), dtype=bool)
    
    shift = np.inf
    while shift > 0.1:
        dist = np.linalg.norm(data - query, axis=1)
        query_old = query
        query = np.mean(data[dist <= radius], axis=0)
        shift = np.linalg.norm(query - query_old)
        # SPEEDUP 2:
        is_near_search_path[dist <= radius/c] = True
   
    return query, is_near_search_path

def mean_shift_opt2(data, radius):
    labels = np.full(len(data), fill_value=-1, dtype=int)
 
    peaks = np.empty(data.shape)
    n_peaks = 0
    
    for idx, query in enumerate(data):
        # Skip point if it already has a valid label assigned
        if labels[idx] != -1:
            continue
            
        peak, is_near_search_path = find_peak_opt(data, query, radius)
        label = None
        
        # Compare found peak to existing peaks
        if n_peaks > 0:
            dist = np.linalg.norm(peaks[:n_peaks] - peak, axis=1)
            label_of_nearest_peak = np.argmin(dist)
            
            # If the nearest existing peak is near enough, take its label
            if dist[label_of_nearest_peak] <= radius / 2:
                label = label_of_nearest_peak
        
        # No existing peak was near enough, create new one
        if label is None:
            label = n_peaks
            peaks[label] = peak
            n_peaks += 1
            
            # SPEEDUP 1: give same label to points near the peak
            dist = np.linalg.norm(data - peak, axis=1)
            labels[dist <= radius] = label
            
        # SPEEDUP 2: give same label to points that were near the search path
        labels[is_near_search_path] = label
    
    peaks = peaks[:n_peaks]

    return peaks, labels

def mean_shift_segment(im, radius):
    
    data = im.reshape(-1, 3).astype(np.float64)
    peaks, labels = mean_shift_opt2(data, radius)
    segmented_im = peaks[labels].reshape(im.shape).astype(np.uint8)

    return data, peaks, labels, segmented_im

def mean_shift_segment_luv(im, radius):
    
    luv_image = cv2.cvtColor(im, cv2.COLOR_RGB2LUV).reshape(-1, 3).astype(np.float64)
    peaks, labels = mean_shift_opt2(luv_image, radius)
    segmented_luv_im = peaks[labels].reshape(im.shape).astype(np.uint8)
    segmented_im = cv2.cvtColor(segmented_luv_im, cv2.COLOR_LUV2RGB)

    return data, peaks, labels, segmented_im
    
def mean_shift_segment_luv_pos(im, radius, pos_factor=1):
    
    im_luv = cv2.cvtColor(im, cv2.COLOR_RGB2LUV)
    y, x = np.mgrid[:im.shape[0],:im.shape[1]]
    xy = np.stack([x, y], axis=-1)
    feature_im = np.concatenate([im_luv, pos_factor*xy], axis=-1)
    data = feature_im.reshape(-1, 5).astype(float)

    peaks, labels = mean_shift_opt2(data, radius)
    segmented_im = peaks[labels][..., :3].reshape(im.shape).astype(np.uint8)
    segmented_im = cv2.cvtColor(segmented_im, cv2.COLOR_LUV2RGB)

    return data, peaks, labels, segmented_im

def plot_3d_clusters(ax, data, labels, peaks, peak_colors=None, colors=None, axis_names='xyz'):
        
        """Plots a set of point clusters in 3D, each with different color."""
        def luv2rgb(color):
            expanded = color[np.newaxis, np.newaxis]
            rgb = cv2.cvtColor(expanded.astype(np.uint8), cv2.COLOR_LUV2RGB)
            return rgb[0,0]/255
        
        if peak_colors is None:
            peak_colors = peaks
        
        for label in range(len(peaks)):
            if colors=='rgb':
                cluster_color = color = peak_colors[label]/255
            elif colors=='luv':
                cluster_color = luv2rgb(peak_colors[label])
            else:
                cluster_color=None

            cluster = data[labels==label]
            ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2],
                    alpha=0.15, color=cluster_color)
            ax.scatter(peaks[label, 0], peaks[label, 1], peaks[label, 2], 
                    color='black', marker='x', s=150, linewidth=3)

        ax.set_xlabel(axis_names[0])
        ax.set_ylabel(axis_names[1])
        ax.set_zlabel(axis_names[2])

def make_label_colormap():
    """Create a color map for visualizing the labels themselves,
    such that the segment boundaries become more visible, unlike
    in the visualization using the cluster peak colors.
    """
    import matplotlib.colors
    rng = np.random.RandomState(2)
    values = np.linspace(0, 1, 20)
    colors = plt.cm.get_cmap('hsv')(values)
    rng.shuffle(colors)
    return matplotlib.colors.ListedColormap(colors)

if __name__ == '__main__':
    
    # This tests out the peak finding algorithm:
    data = np.genfromtxt(f'02-segmentation/data/gaussian_mixture_samples_3d.csv', delimiter=',')
    query_ids = [0, 5, 1500]
    radius = 2

    fig, axes = plt.subplots(1, len(query_ids), figsize=(9.5,3.5))
    for query_id, ax in zip(query_ids, axes):
        query = data[query_id]
        peak = find_peak(data, query, radius)
        print('Found peak', peak)
        
        ax.scatter(data[:, 0], data[:, 1], marker='.', color='gray')
        ax.scatter(query[0], query[1], s=150, linewidth=5,
                color='blue', marker='x', label='starting point')
        ax.scatter(peak[0], peak[1], color='orange', marker='x',
                s=150, linewidth=5, label='found peak')
        ax.legend()
    fig.tight_layout()


    radii = [1, 1.25, 2, 8]
    fig, axes = plt.subplots(
        1, len(radii), figsize=(15,4), subplot_kw={'projection': '3d'})

    for radius, ax in zip(radii, axes): 
        start_time = time.time()
        peaks, labels = mean_shift(data, radius)
        plot_3d_clusters(ax, data, labels, peaks)
        duration = time.time()-start_time
        ax.set_title(
            f'Found {len(peaks)} peaks using radius={radius:.2f}\n'
            f'Computation took {duration:.4f} s\n')
        
    fig.tight_layout()

    radii = [1, 1.25, 2, 8]
    fig, axes = plt.subplots(
        1, len(radii), figsize=(15,4), subplot_kw={'projection': '3d'})

    for radius, ax in zip(radii, axes): 
        start_time = time.time()
        peaks, labels = mean_shift_opt(data, radius)
        plot_3d_clusters(ax, data, labels, peaks)
        duration = time.time()-start_time
        ax.set_title(
            f'Found {len(peaks)} peaks using radius={radius:.2f}\n'
            f'Computation took {duration:.4f} s\n')
        
    fig.tight_layout()

    radii = [1, 1.25, 2, 8]
    fig, axes = plt.subplots(
        1, len(radii), figsize=(15,4), subplot_kw={'projection': '3d'})

    for radius, ax in zip(radii, axes):
        start_time = time.time()
        peaks, labels = mean_shift_opt2(data, radius)
        plot_3d_clusters(ax, data, labels, peaks)
        duration = time.time()-start_time
        ax.set_title(f'Found {len(peaks)} peaks using radius={radius:.2f}\n'
                    f'Computation took {duration:.4f} s\n')
        
    fig.tight_layout()

    label_cmap = make_label_colormap()

    im = imageio.v2.imread('02-segmentation/data/terrain_small.png')
    start_time = time.time()
    data, peaks, labels, segmented_im = mean_shift_segment(im, radius=15)
    duration= time.time()-start_time
    print(f'Took {duration:.2f} s')

    fig = plt.figure(figsize=(9.5,8))
    ax = fig.add_subplot(2, 2, 1)
    ax.set_title('Original Image')
    ax.imshow(im)

    ax = fig.add_subplot(2, 2, 2)
    ax.set_title('Segmented')
    ax.imshow(segmented_im)

    ax = fig.add_subplot(2, 2, 3)
    ax.set_title('Labels')
    ax.imshow(labels.reshape(im.shape[:2]), cmap=label_cmap)

    ax = fig.add_subplot(2, 2, 4, projection='3d')
    ax.set_title(f'RGB space')
    plot_3d_clusters(ax, data, labels, peaks, colors='rgb', axis_names='RGB')
    fig.tight_layout()

    im = imageio.v2.imread('02-segmentation/data/terrain_small.png')
    data, peaks, labels, segmented_im = mean_shift_segment_luv(im, radius=10)
    fig = plt.figure(figsize=(12,8))

    ax = fig.add_subplot(2, 3, 1)
    ax.set_title('Segmented (LUV)')
    ax.imshow(segmented_im)

    ax = fig.add_subplot(2, 3, 2)
    ax.set_title('Labels (LUV)')
    ax.imshow(labels.reshape(im.shape[:2]), cmap=label_cmap)

    ax = fig.add_subplot(2, 3, 3, projection='3d')
    ax.set_title(f'LUV space')
    plot_3d_clusters(ax, data, labels, peaks, colors='luv', axis_names='LUV')

    data, peaks, labels, segmented_im = mean_shift_segment_luv_pos(im, radius=20)
    ax = fig.add_subplot(2, 3, 4)
    ax.set_title('Segmented (LUV+pos)')
    ax.imshow(segmented_im)

    ax = fig.add_subplot(2, 3, 5)
    ax.set_title('Labels (LUV+pos)')
    ax.imshow(labels.reshape(im.shape[:2]), cmap=label_cmap)

    ax = fig.add_subplot(2, 3, 6, projection='3d')
    ax.set_title(f'VXY space')
    plot_3d_clusters(
        ax, data[:, 2:], labels, peaks[:, 2:], 
        peak_colors=peaks[:, :3], colors='luv', axis_names='VXY')
    ax.invert_zaxis()
    ax.view_init(azim=20, elev=15)

    fig.tight_layout()

    radii = [5, 10, 20]
    fig, axes = plt.subplots(len(radii), 3, figsize=(15, 15))
    for radius, ax in zip(radii, axes):
        segmented_im = mean_shift_segment(im, radius)[-1]
        ax[0].imshow(segmented_im)
        ax[0].set_title(f'Radius {radius} RGB')
        
        segmented_im = mean_shift_segment_luv(im, radius)[-1]
        ax[1].imshow(segmented_im)
        ax[1].set_title(f'Radius {radius} LUV')

        segmented_im = mean_shift_segment_luv_pos(im, radius)[-1]
        ax[2].imshow(segmented_im)
        ax[2].set_title(f'Radius {radius} LUV+pos')
    fig.tight_layout()

    # astronaut image testing
    im = skimage.data.astronaut()
    im = cv2.resize(im, (256,256))
    data, peaks, labels, segmented_im = mean_shift_segment_luv(im, radius=15)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].set_title('Original image')
    axes[0].imshow(im)
    axes[1].set_title('Segmented image')
    axes[1].imshow(segmented_im)
    axes[2].set_title('Labels')
    axes[2].imshow(labels.reshape(im.shape[:2]), cmap=label_cmap)
    fig.tight_layout()
    plt.show()



