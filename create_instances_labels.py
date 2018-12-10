import argparse
import glob
import os
import csv
import json

import numpy as np
import cv2
from tqdm import tqdm


def export_bounding_box(args, contour, filename, mask, subdir):
    x, y, w, h = cv2.boundingRect(contour)
    x_min = x
    x_max = x + w
    y_min = y
    y_max = y_min + h
    if args.debug:
        rect_on_mask = mask.copy()
        cv2.rectangle(rect_on_mask, (x_min, y_min), (x_max, y_max), thickness=3, color=255)
        cv2.imshow("bbox", rect_on_mask)
        cv2.waitKey(0)
    with open(os.path.join(args.output_dir, subdir, filename[:-3] + "json"), "w") as json_file:
        json.dump({"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}, fp=json_file)


def export_instance_mask(args, contour_idx, imagefile, label, single_instance, subdir):
    if args.debug:
        print("instance " + label["label"] + "_" + str(contour_idx))
        cv2.imshow("instance", single_instance)
        cv2.waitKey(0)

    filename = os.path.basename(imagefile[:-4] + "_" + label["label"] +
                                "_" + str(contour_idx) + ".png")
    cv2.imwrite(os.path.join(args.output_dir, subdir, filename), single_instance)
    return filename


def do_parsing():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="PNG instance labels creation script")
    parser.add_argument("--dataset_dir", required=True, type=str, help="Train labels directory")
    parser.add_argument("--labels_path", required=True, type=str,
                        help="Path to pbtxt labels file description which depends on the dataset")
    parser.add_argument("--output_dir", required=False, type=str, help="Export directory for PNG labels")
    parser.add_argument("--export_bbox", action="store_true", help="Export bounding boxes in json format")
    parser.add_argument("--debug", action="store_true", help="Show image")
    args = parser.parse_args()
    return args


def main():
    """
    Script to create png instances labels images compatible with Tensorflow Object Detection API
    https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/instance_segmentation.md
    """
    args = do_parsing()
    print(args)

    assert os.path.exists(args.dataset_dir), "Dataset directory does not exist"

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load labels
    labels = list()
    with open(args.labels_path, "r") as labels_file:
        reader = csv.reader(labels_file, delimiter=";")
        print("Classes CSV Header: " + str(next(reader)))

        for index, row in enumerate(reader):
            color_list = [int(a) for a in row[1].split(",")]
            labels.append({"label": row[0], "color": np.array(color_list, dtype=np.uint8), "index": index})

    # Retrieve list of test images
    extensions = ["bmp"]
    imagefiles = []
    _, dirs, _ = next(os.walk(args.dataset_dir))
    for dir in dirs:
        for extension in extensions:
            imagefiles.extend(glob.glob(os.path.join(args.dataset_dir, dir) + "/*." + extension))

    # Create PNG Labels
    for imagefile in tqdm(imagefiles, desc="image"):
        image_bgr = cv2.imread(imagefile, cv2.IMREAD_COLOR)

        fulldir = os.path.dirname(imagefile)
        subdir = fulldir[fulldir.rfind("/") + 1:]
        os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)

        if args.debug:
            cv2.imshow("gt", image_bgr)
            cv2.waitKey(0)

        # Isolate object with connected components

        # We skip background
        for label in labels[1:]:

            if args.debug:
                print("Label: " + label["label"])

            mask = cv2.inRange(image_bgr, label["color"], label["color"])

            ret, thresh = cv2.threshold(mask, 127, 255, 0)
            _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if args.debug:
                cv2.imshow("debug", mask)
                cv2.waitKey(0)

            if label["label"] != "face":

                contour_idx = 0

                for contour in contours:
                    single_instance = np.zeros_like(mask)
                    cv2.fillPoly(single_instance, pts=[contour], color=255)

                    area = cv2.contourArea(contour)

                    if area / float(mask.shape[0] * mask.shape[1]) > 0.001:

                        contour_idx = contour_idx + 1

                        filename = export_instance_mask(args, contour_idx, imagefile, label, single_instance, subdir)

                        if args.export_bbox:
                            export_bounding_box(args, contour, filename, mask, subdir)

            else:

                # Face has inner shapes, ugly solution use directly with if-else
                contour_idx = 1

                filename = export_instance_mask(args, contour_idx, imagefile, label, mask, subdir)

                # Find biggest contour (face contour)
                max_area = 0
                biggest_contour = None
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > max_area:
                        biggest_contour = contour
                        max_area = area

                if args.export_bbox:
                    export_bounding_box(args, biggest_contour, filename, mask, subdir)

    print("Success")


if __name__ == "__main__":
    main()
