import sys, argparse
import numpy as np
from scipy import ndimage
import scipy.signal
from scipy.ndimage import label
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def main(argv):

    parser = argparse.ArgumentParser(description='Script for calculating CD_X, CD_Y and LCDU for vias.')
    parser.add_argument('-i', dest='input', help = 'input image')
    parser.add_argument('-s', dest='scale', default=1,help = 'pixel scale')
    args=parser.parse_args()

    image = binary(apply_gaussian_blur(grescale_image(read_image(args.input)), kernel_size=15, sigma=15))

    # Create a figure and axes for plotting
    fig, ax = plt.subplots()
    ax.imshow(read_image("holes-2.jpg"), cmap='gray')
    ax.imshow(get_contours(image), cmap='gray', alpha=0.5)

    labeled_image, num_holes = label((image == 0))
    x_bounds = find_x_bounds(labeled_image)
    y_bounds = find_y_bounds(labeled_image)

    cd_x = []
    for label_value, (min_x, max_x) in x_bounds.items():
        cd_x +=  [max_x - min_x]

    cd_y = []
    for label_value, (min_y, max_y) in y_bounds.items():
        cd_y +=  [max_y - min_y]

    
    print('CD_X:{} CD_Y {} LCDU_X {} LCDU_Y {}'.format(np.mean(cd_x)*args.scale,np.mean(cd_y)*args.scale,3*np.std(cd_x)*args.scale,3*np.std(cd_y)*args.scale))

    return


def read_image(file):
    return np.array(plt.imread(file))

def grescale_image(image):
    return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

def gaussian_kernel(size=5, sigma=1):
    
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / np.sum(kernel)  # Normalize

def apply_gaussian_blur(image, kernel_size=5, sigma=1):
    
    kernel = gaussian_kernel(kernel_size, sigma)
    blurred = scipy.signal.convolve2d(image, kernel, mode='same', boundary='symm')
    return blurred

def binary(image,threshold=100):
    return (image > threshold) * 255 

def get_contours(image):
    pixels = image.reshape(-1, 1)

    # Apply K-Means clustering with k=2 (black and white)
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(pixels)

    # Get cluster labels and reshape them back to the original image shape
    clustered_image = kmeans.labels_.reshape(image.shape)

    return clustered_image.astype(np.uint8)

def find_x_bounds(labeled_image):

    unique_labels = np.unique(labeled_image)
    unique_labels = unique_labels[unique_labels > 0]  # Ignore background (0)

    component_bounds = {}
    
    for label_value in unique_labels:
        # Get all the pixels of the current component
        indices = np.argwhere(labeled_image == label_value)
        min_x = np.min(indices[:, 1])  # Min column index (X)
        max_x = np.max(indices[:, 1])  # Max column index (X)
        component_bounds[label_value] = (min_x, max_x)

    return component_bounds

def find_y_bounds(labeled_image):

    unique_labels = np.unique(labeled_image)
    unique_labels = unique_labels[unique_labels > 0]  # Ignore background (0)

    component_bounds = {}

    for label_value in unique_labels:
        # Get all the pixels of the current component
        indices = np.argwhere(labeled_image == label_value)
        min_y = np.min(indices[1,:])  # Min column index (y)
        max_y = np.max(indices[1,:])  # Max column index (y)
        component_bounds[label_value] = (min_y, max_y)

    return component_bounds

if __name__ == "__main__":
    main(sys.argv[1:])
