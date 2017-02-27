import argparse
import glob
import os
import pickle
import cv2
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label

from common import *

output_dir = "output_images"  # output directory
y_start = 400  # start y coordinate for scanning
y_stop = 656  # stop y coordinate for scanning
scale = 1.5


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    """Draw boxes"""
    imcopy = np.copy(img)
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    return imcopy


def find_cars(img, y_start, y_stop, scale, clf, X_scaler, color_space='RGB',
              spatial_features=True, spatial_size=(32, 32),
              hist_features=True, hist_bins=32,
              hog_features=True, hog_orient=9, hog_pix_per_cell=8, hog_cell_per_block=2, hog_channel=0):
    """Extract features and make predictions"""
    box_list = []
    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255

    img_tosearch = img[y_start:y_stop, :, :]
    ctrans_tosearch = convert_RGB(img_tosearch, color_space)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // hog_pix_per_cell) - 1
    nyblocks = (ch1.shape[0] // hog_pix_per_cell) - 1
    nfeat_per_block = hog_orient * hog_cell_per_block ** 2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // hog_pix_per_cell) - 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, hog_orient, hog_pix_per_cell, hog_cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, hog_orient, hog_pix_per_cell, hog_cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, hog_orient, hog_pix_per_cell, hog_cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            xpos = xb * cells_per_step
            ypos = yb * cells_per_step
            xleft = xpos * hog_pix_per_cell
            ytop = ypos * hog_pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_feat = np.empty(0)
            if spatial_features:
                spatial_feat = get_spatial_features(subimg, size=spatial_size)
            hist_feat = np.empty(0)
            if hist_features:
                hist_feat = get_hist_features(subimg, nbins=hist_bins)
            # Extract HOG for this patch
            hog_feat = np.empty(0)
            if hog_features:
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            # Scale features
            test_features = X_scaler.transform(np.hstack((spatial_feat, hist_feat, hog_feat)).reshape(1, -1))
            # Make prediction
            test_prediction = clf.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                pt1 = (xbox_left, ytop_draw + y_start)
                pt2 = (xbox_left + win_draw, ytop_draw + win_draw + y_start)
                box_list.append((pt1, pt2))
                cv2.rectangle(draw_img, pt1, pt2, (0, 0, 255), 6)

    return draw_img, box_list


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


class CarTracker:
    """ Tracks cars in a video stream. This is used as a container to hold all parameters relative to a tracking
    session for a given video. """

    def __init__(self, y_start, y_stop, scale, clf, X_scaler, color_space='RGB',
                 spatial_features=True, spatial_size=(32, 32),
                 hist_features=True, hist_bins=32,
                 hog_features=True, hog_orient=9, hog_pix_per_cell=8, hog_cell_per_block=2, hog_channel=0):
        self.y_start = y_start
        self.y_stop = y_stop
        self.scale = scale
        self.clf = clf
        self.X_scaler = X_scaler
        self.color_space = color_space
        self.spatial_features = spatial_features
        self.spatial_size = spatial_size
        self.hist_features = hist_features
        self.hist_bins = hist_bins
        self.hog_features = hog_features
        self.orient = hog_orient
        self.pix_per_cell = hog_pix_per_cell
        self.cell_per_block = hog_cell_per_block
        self.hog_channel = hog_channel

    def next_image(self, img):
        """Invoked for each image composing the video.

        :param img: next image in the video stream
        :returns: new image with bounding boxes around detected cars"""
        heat = np.zeros_like(img[:, :, 0]).astype(np.float)

        out_img, box_list = find_cars(img, self.y_start, self.y_stop, self.scale, self.clf, self.X_scaler,
                                      color_space=self.color_space,
                                      # Spatial features
                                      spatial_features=self.spatial_features, spatial_size=self.spatial_size,
                                      # Hist features
                                      hist_features=self.hist_features, hist_bins=self.hist_bins,
                                      # Hog features
                                      hog_features=self.hog_features, hog_orient=self.orient,
                                      hog_pix_per_cell=self.pix_per_cell,
                                      hog_cell_per_block=self.cell_per_block, hog_channel=self.hog_channel)

        # Add heat to each box in box list
        heat = add_heat(heat, box_list)
        # Apply threshold to help remove false positives
        heat = apply_threshold(heat, 1)
        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)
        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(np.copy(img), labels)

        return draw_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detects cars in image or video')
    parser.add_argument('-m', '--model', default="model.p",
                        help='path of the model to load')
    parser.add_argument("file", help="image, video or directory to scan")
    args = parser.parse_args()

    data = pickle.load(open(args.model, "rb"))

    files = glob.glob(args.file, recursive=True)
    for file in files:
        output_file = output_dir + os.sep + os.path.basename(file)

        carTracker = CarTracker(y_start, y_stop, scale, data['clf'], data['X_scaler'], color_space=data['color_space'],
                                spatial_features=data['spatial_features'], spatial_size=data['spatial_size'],
                                hist_features=data['hist_features'], hist_bins=data['hist_bins'],
                                hog_features=data['hog_features'], hog_orient=data['hog_orient'],
                                hog_pix_per_cell=data['hog_pix_per_cell'],
                                hog_cell_per_block=data['hog_cell_per_block'],
                                hog_channel=data['hog_channel'])

        _, file_extension = os.path.splitext(file)
        if ".jpg" == file_extension.lower():
            img = mpimg.imread(file)
            out_img = carTracker.next_image(img)
            cv2.imwrite(output_file, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
        elif ".mp4" == file_extension.lower():
            clip = VideoFileClip(file)
            output_clip = clip.fl_image(carTracker.next_image)
            output_clip.write_videofile(output_file, audio=False)
        else:
            print("Unknown file format: " + args.file)
            continue
