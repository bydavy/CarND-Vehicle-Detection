import argparse
import glob
import pickle
import time
from common import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.svm import LinearSVC

# Path of vehicles images
vehicles_data = "data/vehicles/**/*.png"
# Path of non vehicles images
non_vehicles_data = "data/non-vehicles/**/*.png"
# Output file to save the classifier and features extraction parameters
save_file = "model.p"
# Color space used for features extraction
color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
# Spatial binning of color features parameters
spatial_features = True  # Enable spatial features
spatial_size = (32, 32)
# Histogram of color features parameters
hist_features = True  # Enable hist features
hist_bins = 32
# Hog features parameters
hog_features = True  # Enable/disable hog features
hog_orient = 9
hog_pix_per_cell = 8
hog_cell_per_block = 2
hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train classifier to detect cars')
    parser.add_argument('-o', '--output', default=save_file,
                        help='path to save the model')
    args = parser.parse_args()

    # Read images filename
    cars = glob.glob(vehicles_data, recursive=True)
    non_cars = glob.glob(non_vehicles_data, recursive=True)

    if len(cars) == 0:
        print("Empty set of cars images")
        exit(-1)
    if len(non_cars) == 0:
        print("Empty set of non cars images")
        exit(-1)

    print(len(cars), "vehicles images")
    print(len(non_cars), "non vehicles images")

    # Extract features
    car_features = imgs_features(cars, color_space=color_space,
                                 # Spatial features
                                 spatial_features=spatial_features, spatial_size=spatial_size,
                                 # Hist features
                                 hist_features=hist_features, hist_bins=hist_bins,
                                 # Hog features
                                 hog_features=hog_features, hog_orient=hog_orient, hog_pix_per_cell=hog_pix_per_cell,
                                 hog_cell_per_block=hog_cell_per_block, hog_channel=hog_channel)
    non_car_features = imgs_features(non_cars, color_space=color_space,
                                     # Spatial features
                                     spatial_features=spatial_features, spatial_size=spatial_size,
                                     # Hist features
                                     hist_features=hist_features, hist_bins=hist_bins,
                                     # Hog features
                                     hog_features=hog_features, hog_orient=hog_orient, hog_pix_per_cell=hog_pix_per_cell,
                                     hog_cell_per_block=hog_cell_per_block, hog_channel=hog_channel)

    # Create an array stack of feature vectors
    X = np.vstack((car_features, non_car_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(non_car_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    clf = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    clf.fit(X_train, y_train)
    print(round(time.time() - t, 2), 'secs to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t = time.time()
    n_predict = 10
    print('My SVC predicts:\t', clf.predict(X_test[0:n_predict]))
    print('For these', n_predict, 'labels: ', y_test[0:n_predict])
    print(round(time.time() - t, 5), 'secs to predict', n_predict, 'labels with SVC')

    # Persist classifier and features extraction parameters
    data = dict()
    data['clf'] = clf
    data['X_scaler'] = X_scaler
    data['color_space'] = color_space
    data['hog_features'] = hog_features
    if hog_features:
        data['hog_orient'] = hog_orient
        data['hog_pix_per_cell'] = hog_pix_per_cell
        data['hog_cell_per_block'] = hog_cell_per_block
        data['hog_channel'] = hog_channel
    data['spatial_features'] = spatial_features
    if spatial_features:
        data['spatial_size'] = spatial_size
    data['hist_features'] = hist_features
    if hist_features:
        data['hist_bins'] = hist_bins
    pickle.dump(data, open(args.output, "wb"))
    print("Model saved to", args.output)
