import cv2
import numpy as np
from skimage.feature import hog

import matplotlib.image as mpimg


def convert_RGB(img, to='RGB'):
    """Convert from RGB to another color space.

    :param img: image to convert
    :param to: color space to convert the image to. Can be RGB, HSV, LUV, HLS, YUV, YCrCb.
    :returns: a new instance of the image in the new color space"""
    if to != 'RGB':
        if to == 'HSV':
            conv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif to == 'LUV':
            conv_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif to == 'HLS':
            conv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif to == 'YUV':
            conv_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif to == 'YCrCb':
            conv_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        conv_image = np.copy(img)
    return conv_image


def get_spatial_features(img, size=(42, 42)):
    """Extract spatial binning of color features

    :param img: image to extract features from
    :param size: resize size
    :returns: features vector"""
    return cv2.resize(img, size).ravel()


# Define a function to compute color histogram features
def get_hist_features(img, nbins=32, bins_range=(0, 256)):
    """Extract histogram of color features

    :param img: image to extract features from
    :param nbins: number of bins
    :param bins_range: range of each bin
    :returns: features vector"""
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    return np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))


# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    """Extract hog features

    :param img:  image to extract features from
    :param orient: number of orientation bins
    :param pix_per_cell: number of pixels per cell
    :param cell_per_block: number of cells per block
    :param channel: list of channels to apply hog to
    :param vis: True to have the hog image returned
    :param feature_vec: True to have the feature vector returned
    """
    # Call with two outputs if vis==True
    if vis:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features


# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB',
                        spatial_features=True, spatial_size=(32, 32),
                        hist_features=True, hist_bins=32,
                        hog_features=True, hog_orient=9, hog_pix_per_cell=8, hog_cell_per_block=2, hog_channel=0):
    """Extract features of a single image.

        :param img: image's filename to extract features from
        :param color_space: color space used for feature extraction. Can be RGB, HSV, LUV, HLS, YUV, YCrCb.
        :param spatial_features: boolean to toggle spatial binning of color features extraction status
        :param spatial_size: resize size
        :param hist_features: boolean to toggle Histogram of color features extraction status
        :param hist_bins: number of color bins
        :param hog_features: boolean to toggle Hog features extraction status
        :param hog_orient: number of orientation bins
        :param hog_pix_per_cell: number of pixels per cell
        :param hog_cell_per_block: number of cells per block
        :param hog_channel: list of channels to apply hog to
        :returns: a vector that contains the extracted features"""
    img_features = []
    # Apply color conversion
    feature_image = convert_RGB(img, color_space)
    # Extract spatial features
    if spatial_features:
        s_features = get_spatial_features(feature_image, size=spatial_size)
        img_features.append(s_features)
    # Extract hist features
    if hist_features:
        h_features = get_hist_features(feature_image, nbins=hist_bins)
        img_features.append(h_features)
    # Extract hog features
    if hog_features:
        h_features = []
        if hog_channel == 'ALL':
            channels = list(range(feature_image.shape[2]))
        else:
            channels = [hog_channel]

        for channel in channels:
            h_features.extend(get_hog_features(feature_image[:, :, channel],
                                               hog_orient, hog_pix_per_cell, hog_cell_per_block,
                                               vis=False, feature_vec=True))
        img_features.append(h_features)

    # Concatenate all features to have a vector
    return np.concatenate(img_features)


def imgs_features(files, color_space='RGB',
                  # Spatial features
                 spatial_features=True, spatial_size=(32, 32),
                  # Hist features
                 hist_features=True, hist_bins=32,
                  # Hog features
                 hog_features=True, hog_orient=9, hog_pix_per_cell=8, hog_cell_per_block=2, hog_channel=0):
    """Extract features of each image from the image list.

        :param files: list of image's filename to extract features from
        :param color_space: color space used for feature extraction. Can be RGB, HSV, LUV, HLS, YUV, YCrCb.
        :param spatial_features: boolean to toggle spatial binning of color features extraction status
        :param spatial_size: resize size
        :param hist_features: boolean to toggle Histogram of color features extraction status
        :param hist_bins: number of color bins
        :param hog_features: boolean to toggle Hog features extraction status
        :param hog_orient: number of orientation bins
        :param hog_pix_per_cell: number of pixels per cell
        :param hog_cell_per_block: number of cells per block
        :param hog_channel: list of channels to apply hog to
        :returns: list of numpy arrays, each containing the extracted features"""
    features = []
    for file in files:
        image = mpimg.imread(file)
        feature = single_img_features(image, color_space=color_space,
                                      spatial_features=spatial_features, spatial_size=spatial_size,
                                      hist_features=hist_features, hist_bins=hist_bins,
                                      hog_features=hog_features, hog_orient=hog_orient,
                                      hog_pix_per_cell=hog_pix_per_cell,
                                      hog_cell_per_block=hog_cell_per_block, hog_channel=hog_channel)
        features.append(feature)
    return features
