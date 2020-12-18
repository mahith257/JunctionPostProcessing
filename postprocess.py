import image_slicer
import cv2
import glob
import os
import numpy as np
from pathlib import Path
from PIL import Image
import math
import pandas as pd
from sklearn.metrics import r2_score
import config


# function for distance between two points
def dist(x, y, a, b):
    distance = math.sqrt((a - x) ** 2 + (b - y) ** 2)
    return distance


# function to get lines after applying houghLinesP method
def get_lines(lines_in):
    return [l[0] for l in lines_in]


def subtract_arrays(outer, inner):
    final = []
    if len(inner) == 0:
        final = outer
    else:
        for i in range(len(outer)):
            for j in range(len(inner)):
                if np.array_equal(outer[i], inner[j]):
                    break
                else:
                    if j == len(inner) - 1:
                        final.append(outer[i])

    return final


def get_file_names(path):
    file_names = []
    for img in glob.glob(path):
        m = Path(img).stem
        file_names.append(m)
    return file_names


def erosion(path, read_path, write_path):
    kernel_1 = np.ones((5, 5), np.uint8)
    for n in range(len(get_file_names(path))):
        image = cv2.imread(read_path + file_names_bm[n]+'.png')
        erode_image = cv2.erode(image, kernel_1, iterations=1)
        cv2.imwrite(write_path + file_names_bm[n]+'.jpg', erode_image)


def skeleton(path, read_path, write_path):
    for n in range(len(get_file_names(path))):
        img = cv2.imread(read_path + file_names_bm[n]+'.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, img = cv2.threshold(img, 50, 255, 0)
        size = np.size(img)
        skel = np.zeros(img.shape, np.uint8)
        # cross shaped kernel
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

        while True:
            # opening of image or erosion followed by dilation
            open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
            temp = cv2.subtract(img, open)
            eroded = cv2.erode(img, element)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()
            if cv2.countNonZero(img) == 0:
                break
        cv2.imwrite(write_path + file_names_bm[n]+'.jpg', skel)


def make_directory(path):
    os.mkdir(path)


def slicing(img_path, save_path, no_of_slices):
    # slicing the binary mask images into 49 slices(7*7) using image_slicer
    tiles = image_slicer.slice(img_path, no_of_slices, save=False)
    # saving the sliced images of binary mask in the respective folders
    image_slicer.save_tiles(tiles, directory=save_path, prefix='slice', format="jpeg")

    return tiles


def read_images(path):
    slices = []
    for img in glob.glob(path):
        y = cv2.imread(img)
        slices.append(y)

    return slices


def points_at_interval(lines, interval):
    """
    to get the points at specified interval on the lines
    :param lines: lines
    :param interval: interval of 10px
    :return: points
    """
    points = []
    for line in lines:
        x1, y1, x2, y2 = line
        points.append([x1, y1])
        points.append([x2, y2])
        if dist(x1, y1, x2, y2) > interval:
            i = 1
            while i > 0:
                s = ((2 * (y1 - y2) * (x2 * y1 - y2 * x1)) + (
                        x1 ** 2 + y1 ** 2 - x2 ** 2 - y2 ** 2 - (interval * i) ** 2 + (
                        (dist(x1, y1, x2, y2) - (interval * i)) ** 2)) * (x1 - x2)) / (
                            2 * ((x2 - x1) ** 2 + (y2 - y1) ** 2))
                t = ((2 * (x2 - x1) * (x2 * y1 - y2 * x1)) + (
                        x1 ** 2 + y1 ** 2 - x2 ** 2 - y2 ** 2 - (interval * i) ** 2 + (
                        (dist(x1, y1, x2, y2) - (interval * i)) ** 2)) * (y1 - y2)) / (
                            2 * ((x2 - x1) ** 2 + (y2 - y1) ** 2))
                points.append([s, t])
                i = i + 1
                if dist(s, t, x2, y2) < interval:
                    i = 0

    return points


def add_arrays_of_lines(final, lines):
    for j in range(len(lines)):
        x1, y1, x2, y2 = lines[j]
        final.append(lines[j])

    return final


def parallel_lines_in_slices(lines):
    """
    if there are two lines in the same slice within 10% slope then create a median
    :param lines: lines detected in the slice
    :return: lines without the parallel lines but with median lines
    """
    all_lines = []
    parallel_lines = []
    parallel_lines_drawn = []
    if len(get_lines(lines)) == 1:
        x1, y1, x2, y2 = get_lines(lines)[0]
        all_lines.append(np.array(get_lines(lines)[0]))
    elif len(get_lines(lines)) == 2:
        x1, y1, x2, y2 = get_lines(lines)[0]
        x3, y3, x4, y4 = get_lines(lines)[1]
        all_lines.append(np.array(get_lines(lines)[0]))
        all_lines.append(np.array(get_lines(lines)[1]))
        diff = abs(((y2 - y1) / (x2 - x1)) - ((y4 - y3) / (x4 - x3)))
        if diff <= 0.1:
            x5 = round((x1 + x3) / 2)
            y5 = round((y1 + y3) / 2)
            x6 = round((x2 + x4) / 2)
            y6 = round((y2 + y4) / 2)
            parallel_lines_drawn.append([x5, y5, x6, y6])
            parallel_lines.append(np.array(get_lines(lines)[0]))
            parallel_lines.append(np.array(get_lines(lines)[1]))
    else:
        for line in get_lines(lines):
            x1, y1, x2, y2 = line
            all_lines.append(line)
            for l in get_lines(lines):
                x3, y3, x4, y4 = l
                if (x3 != x1) and (y3 != y1) and (x4 != x2) and (y4 != y2):
                    diff = abs(((y2 - y1) / (x2 - x1)) - ((y4 - y3) / (x4 - x3)))
                    if diff <= 0.1:
                        x5 = round((x1 + x3) / 2)
                        y5 = round((y1 + y3) / 2)
                        x6 = round((x2 + x4) / 2)
                        y6 = round((y2 + y4) / 2)
                        parallel_lines_drawn.append([x5, y5, x6, y6])
                        parallel_lines.append(l)
                        parallel_lines.append(line)

    parallel_lines_drawn = np.unique(parallel_lines_drawn, axis=0)
    parallel_lines = np.unique(parallel_lines, axis=0)
    rest_lines = subtract_arrays(all_lines, parallel_lines)

    rest_lines = add_arrays_of_lines(rest_lines, parallel_lines_drawn)

    return rest_lines


def polyfit_in_slices(all_points, model):
    """
    applying polyfit in each slice
    :param all_points:
    :param model:
    :return: lines fitted
    """
    lines_fitted = []
    m = model[0]
    c = model[1]
    max_num = 0
    x8 = 0
    y8 = 0
    x9 = 0
    y9 = 0
    for p in all_points:
        x1, y1 = p
        for j in all_points:
            x2, y2 = j
            if dist(x1, y1, x2, y2) >= max_num:
                max_num = dist(x1, y1, x2, y2)
                x8 = x1
                y8 = y1
                x9 = x2
                y9 = y2
    h1 = round((((-m) * (m * x8 - y8 + c)) / (m ** 2 + 1)) + x8)
    k1 = round(((m * x8 - y8 + c) / (m ** 2 + 1)) + y8)
    h2 = round((((-m) * (m * x9 - y9 + c)) / (m ** 2 + 1)) + x9)
    k2 = round(((m * x9 - y9 + c) / (m ** 2 + 1)) + y9)
    lines_fitted.append([h1, k1, h2, k2])

    return lines_fitted


def polyfit_after_slicing(lines):
    """
    after assembling the slices, if there are two lines within 10% slope and distance between them(end points) is less
    than 10px then fit a line through them using polyfit
    :param lines:
    :return:
    """
    fit_lines = []
    fit_lines_drawn = []
    for s in range(len(lines)):
        x7, y7, x8, y8 = lines[s]
        for h in range(len(lines)):
            x9, y9, x10, y10 = lines[h]
            if (x7 != x9) and (y7 != y9) and (x8 != x10) and (y8 != y10):
                if (x8 - x7) != 0 and (x10 - x9) != 0:
                    diff = abs(((y8 - y7) / (x8 - x7)) - ((y10 - y9) / (x10 - x9)))
                    if diff <= 0.1:
                        if (max(dist(x7, y7, x9, y9), dist(x7, y7, x10, y10), dist(x8, y8, x9, y9),
                                dist(x8, y8, x10, y10))) == dist(x7, y7, x9, y9):
                            if (min(dist(x7, y7, x9, y9), dist(x7, y7, x10, y10), dist(x8, y8, x9, y9),
                                    dist(x8, y8, x10, y10))) <= 10:
                                fit_lines.append([x7, y7, x8, y8])
                                fit_lines.append([x9, y9, x10, y10])
                                data = {'x': [x7, x8, x9, x10], 'y': [y7, y8, y9, y10]}
                                data_frame = pd.DataFrame(data=data)
                                x = data_frame.x
                                y = data_frame.y
                                model = np.polyfit(x, y, 1)
                                m = model[0]
                                c = model[1]
                                h1 = round((((-m) * (m * x7 - y7 + c)) / (m ** 2 + 1)) + x7)
                                k1 = round(((m * x7 - y7 + c) / (m ** 2 + 1)) + y7)
                                h2 = round((((-m) * (m * x9 - y9 + c)) / (m ** 2 + 1)) + x9)
                                k2 = round(((m * x9 - y9 + c) / (m ** 2 + 1)) + y9)
                                fit_lines_drawn.append([h1, k1, h2, k2])
                        elif (max(dist(x7, y7, x9, y9), dist(x7, y7, x10, y10), dist(x8, y8, x9, y9),
                                  dist(x8, y8, x10, y10))) == dist(x7, y7, x10, y10):
                            if (min(dist(x7, y7, x9, y9), dist(x7, y7, x10, y10), dist(x8, y8, x9, y9),
                                    dist(x8, y8, x10, y10))) <= 10:
                                fit_lines.append([x7, y7, x8, y8])
                                fit_lines.append([x9, y9, x10, y10])
                                data = {'x': [x7, x8, x9, x10], 'y': [y7, y8, y9, y10]}
                                data_frame = pd.DataFrame(data=data)
                                x = data_frame.x
                                y = data_frame.y
                                model = np.polyfit(x, y, 1)
                                m = model[0]
                                c = model[1]
                                h1 = round((((-m) * (m * x7 - y7 + c)) / (m ** 2 + 1)) + x7)
                                k1 = round(((m * x7 - y7 + c) / (m ** 2 + 1)) + y7)
                                h2 = round((((-m) * (m * x10 - y10 + c)) / (m ** 2 + 1)) + x10)
                                k2 = round(((m * x10 - y10 + c) / (m ** 2 + 1)) + y10)
                                fit_lines_drawn.append([h1, k1, h2, k2])
                        elif (max(dist(x7, y7, x9, y9), dist(x7, y7, x10, y10), dist(x8, y8, x9, y9),
                                  dist(x8, y8, x10, y10))) == dist(x8, y8, x9, y9):
                            if (min(dist(x7, y7, x9, y9), dist(x7, y7, x10, y10), dist(x8, y8, x9, y9),
                                    dist(x8, y8, x10, y10))) <= 10:
                                fit_lines.append([x7, y7, x8, y8])
                                fit_lines.append([x9, y9, x10, y10])
                                data = {'x': [x7, x8, x9, x10], 'y': [y7, y8, y9, y10]}
                                data_frame = pd.DataFrame(data=data)
                                x = data_frame.x
                                y = data_frame.y
                                model = np.polyfit(x, y, 1)
                                m = model[0]
                                c = model[1]
                                h1 = round((((-m) * (m * x8 - y8 + c)) / (m ** 2 + 1)) + x8)
                                k1 = round(((m * x8 - y8 + c) / (m ** 2 + 1)) + y8)
                                h2 = round((((-m) * (m * x9 - y9 + c)) / (m ** 2 + 1)) + x9)
                                k2 = round(((m * x9 - y9 + c) / (m ** 2 + 1)) + y9)
                                fit_lines_drawn.append([h1, k1, h2, k2])
                        elif (max(dist(x7, y7, x9, y9), dist(x7, y7, x10, y10), dist(x8, y8, x9, y9),
                                  dist(x8, y8, x10, y10))) == dist(x8, y8, x10, y10):
                            if (min(dist(x7, y7, x9, y9), dist(x7, y7, x10, y10), dist(x8, y8, x9, y9),
                                    dist(x8, y8, x10, y10))) <= 10:
                                fit_lines.append([x7, y7, x8, y8])
                                fit_lines.append([x9, y9, x10, y10])
                                data = {'x': [x7, x8, x9, x10], 'y': [y7, y8, y9, y10]}
                                data_frame = pd.DataFrame(data=data)
                                x = data_frame.x
                                y = data_frame.y
                                model = np.polyfit(x, y, 1)
                                m = model[0]
                                c = model[1]
                                h1 = round((((-m) * (m * x8 - y8 + c)) / (m ** 2 + 1)) + x8)
                                k1 = round(((m * x8 - y8 + c) / (m ** 2 + 1)) + y8)
                                h2 = round((((-m) * (m * x10 - y10 + c)) / (m ** 2 + 1)) + x10)
                                k2 = round(((m * x10 - y10 + c) / (m ** 2 + 1)) + y10)
                                fit_lines_drawn.append([h1, k1, h2, k2])

    fit_lines = np.unique(fit_lines, axis=0)
    fit_lines_drawn = np.unique(fit_lines_drawn, axis=0)
    final_lines_all = subtract_arrays(rest_lines_all, fit_lines)

    for l in fit_lines_drawn:
        x1, y1, x2, y2 = l
        final_lines_all.append([x1, y1, x2, y2])

    return final_lines_all


instance_mask = config.INSTANCE_MASK_PATH
visualized_image = config.VISUALIZED_IMAGES_PATH
source_image = config.SOURCE_IMAGE_PATH
concatenated_images = config.CONCATENATED_IMAGES_PATH
slicing_images = config.SLICING_IMAGES_PATH
sliced_tiles = config.SLICED_TILES_PATH
file_names_bm = get_file_names(instance_mask)

erosion(instance_mask, 'instance_mask_new/', 'erode_instance_images_new/')
skeleton(instance_mask, 'instance_mask_new/', 'skeleton_instance_images_new/')
for n in range(len(file_names_bm)):
    # for making a directory with filenames of binary mask images if directory already exists it shows an error
    # make_directory('erode_instance_sliced_tiles_new/'+file_names_bm[n])
    # slicing and saving the binary mask images into 49 slices(7*7) using image_slicer
    tiles = slicing(slicing_images +file_names_bm[n]+'.jpg', sliced_tiles+file_names_bm[n], 64)
    rest_lines_all = []
    # appending the filenames of the sliced images of a corresponding binary mask image in an empty list
    file_names = get_file_names(sliced_tiles + file_names_bm[n] + '/*.jpg')
    sliced_images = read_images(sliced_tiles + file_names_bm[n] + '/*.jpg')

    for z in range(len(sliced_images)):
        poly_fit_lines = []
        img = sliced_images[z]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=10, minLineLength=10, maxLineGap=60)
        if np.any(lines):
            # R1
            rest_lines = parallel_lines_in_slices(lines)
            all_points = points_at_interval(rest_lines, 10)
            all_points = np.unique(all_points, axis=0)
            if np.any(all_points):
                if len(all_points) > 1:
                    data = {'x': [], 'y': []}
                    for point in all_points:
                        x1, y1 = point
                        data['x'].append(x1)
                        data['y'].append(y1)
                    data_frame = pd.DataFrame(data=data)
                    x = data_frame.x
                    y = data_frame.y
                    model = np.polyfit(x, y, 1)
                    predict = np.poly1d(model)
                    accuracy = r2_score(y, predict(x))
                    # if r2_score is less then the accuracy of the model is less
                    # if r2_score is less than 0.5 then divide the slice into four quadrants and apply the same pipeline
                    if accuracy > 0.5:
                        poly_fit_lines = polyfit_in_slices(all_points, model)
                    else:
                        # make_directory(sliced_tiles + file_names_bm[n] + '/' + file_names[z])
                        tiles_2 = slicing(sliced_tiles + file_names_bm[n] + '/' + file_names[z] + '.jpg',
                                          sliced_tiles + file_names_bm[n] + '/' + file_names[z], 4)
                        sliced_images_2 = read_images(sliced_tiles + file_names_bm[n] + '/' + file_names[z] + '/*.jpg')
                        for b in range(len(sliced_images_2)):
                            img_2 = sliced_images_2[b]
                            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
                            edges = cv2.Canny(gray, 50, 150)
                            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=10, minLineLength=10,
                                                    maxLineGap=60)
                            if np.any(lines):
                                rest_lines = parallel_lines_in_slices(lines)
                                all_points = points_at_interval(rest_lines, 10)
                                all_points = np.unique(all_points, axis=0)
                                if np.any(all_points):
                                    if len(all_points) > 1:
                                        data = {'x': [], 'y': []}
                                        for point in all_points:
                                            x1, y1 = point
                                            data['x'].append(x1)
                                            data['y'].append(y1)
                                        data_frame = pd.DataFrame(data=data)
                                        x = data_frame.x
                                        y = data_frame.y
                                        model = np.polyfit(x, y, 1)
                                        poly_fit_lines_2 = polyfit_in_slices(all_points, model)

                                for k in range(2):
                                    if b == (2 * k):
                                        for m in range(len(poly_fit_lines_2)):
                                            x1, y1, x2, y2 = poly_fit_lines_2[m]
                                            x1 = x1
                                            y1 = y1 + (img_2.shape[0] * k)
                                            x2 = x2
                                            y2 = y2 + (img_2.shape[0] * k)
                                            poly_fit_lines_2[m] = [x1, y1, x2, y2]
                                            poly_fit_lines.append(poly_fit_lines_2[m])
                                    elif b == ((2 * k) + 1):
                                        for m in range(len(poly_fit_lines_2)):
                                            x1, y1, x2, y2 = poly_fit_lines_2[m]
                                            x1 = x1 + img_2.shape[1]
                                            y1 = y1 + (img_2.shape[0] * k)
                                            x2 = x2 + img_2.shape[1]
                                            y2 = y2 + (img_2.shape[0] * k)
                                            poly_fit_lines_2[m] = [x1, y1, x2, y2]
                                            poly_fit_lines.append(poly_fit_lines_2[m])

                # converting the parameters of points with respect to the concatenated image
                # as we have done 64 slices(square root of 64 is 8)
                for k in range(8):
                    if z == (8 * k):
                        for m in range(len(poly_fit_lines)):
                            x1, y1, x2, y2 = poly_fit_lines[m]
                            x1 = x1
                            y1 = y1 + (img.shape[0] * k)
                            x2 = x2
                            y2 = y2 + (img.shape[0] * k)
                            poly_fit_lines[m] = [x1, y1, x2, y2]
                            rest_lines_all.append(poly_fit_lines[m])
                    elif z == ((8 * k) + 1):
                        for m in range(len(poly_fit_lines)):
                            x1, y1, x2, y2 = poly_fit_lines[m]
                            x1 = x1 + img.shape[1]
                            y1 = y1 + (img.shape[0] * k)
                            x2 = x2 + img.shape[1]
                            y2 = y2 + (img.shape[0] * k)
                            poly_fit_lines[m] = [x1, y1, x2, y2]
                            rest_lines_all.append(poly_fit_lines[m])
                    elif z == ((8 * k) + 2):
                        for m in range(len(poly_fit_lines)):
                            x1, y1, x2, y2 = poly_fit_lines[m]
                            x1 = x1 + (img.shape[1] * 2)
                            y1 = y1 + (img.shape[0] * k)
                            x2 = x2 + (img.shape[1] * 2)
                            y2 = y2 + (img.shape[0] * k)
                            poly_fit_lines[m] = [x1, y1, x2, y2]
                            rest_lines_all.append(poly_fit_lines[m])
                    elif z == ((8 * k) + 3):
                        for m in range(len(poly_fit_lines)):
                            x1, y1, x2, y2 = poly_fit_lines[m]
                            x1 = x1 + (img.shape[1] * 3)
                            y1 = y1 + (img.shape[0] * k)
                            x2 = x2 + (img.shape[1] * 3)
                            y2 = y2 + (img.shape[0] * k)
                            poly_fit_lines[m] = [x1, y1, x2, y2]
                            rest_lines_all.append(poly_fit_lines[m])
                    elif z == ((8 * k) + 4):
                        for m in range(len(poly_fit_lines)):
                            x1, y1, x2, y2 = poly_fit_lines[m]
                            x1 = x1 + (img.shape[1] * 4)
                            y1 = y1 + (img.shape[0] * k)
                            x2 = x2 + (img.shape[1] * 4)
                            y2 = y2 + (img.shape[0] * k)
                            poly_fit_lines[m] = [x1, y1, x2, y2]
                            rest_lines_all.append(poly_fit_lines[m])
                    elif z == ((8 * k) + 5):
                        for m in range(len(poly_fit_lines)):
                            x1, y1, x2, y2 = poly_fit_lines[m]
                            x1 = x1 + (img.shape[1] * 5)
                            y1 = y1 + (img.shape[0] * k)
                            x2 = x2 + (img.shape[1] * 5)
                            y2 = y2 + (img.shape[0] * k)
                            poly_fit_lines[m] = [x1, y1, x2, y2]
                            rest_lines_all.append(poly_fit_lines[m])
                    elif z == ((8 * k) + 6):
                        for m in range(len(poly_fit_lines)):
                            x1, y1, x2, y2 = poly_fit_lines[m]
                            x1 = x1 + (img.shape[1] * 6)
                            y1 = y1 + (img.shape[0] * k)
                            x2 = x2 + (img.shape[1] * 6)
                            y2 = y2 + (img.shape[0] * k)
                            poly_fit_lines[m] = [x1, y1, x2, y2]
                            rest_lines_all.append(poly_fit_lines[m])
                    elif z == ((8 * k) + 7):
                        for m in range(len(poly_fit_lines)):
                            x1, y1, x2, y2 = poly_fit_lines[m]
                            x1 = x1 + (img.shape[1] * 7)
                            y1 = y1 + (img.shape[0] * k)
                            x2 = x2 + (img.shape[1] * 7)
                            y2 = y2 + (img.shape[0] * k)
                            poly_fit_lines[m] = [x1, y1, x2, y2]
                            rest_lines_all.append(poly_fit_lines[m])

    # concatenating the slices
    im_tile = image_slicer.join(tiles)
    im_tile = cv2.cvtColor(np.array(im_tile), cv2.COLOR_RGB2BGR)

    rest_lines_all = np.unique(rest_lines_all, axis=0)

    final_lines = polyfit_after_slicing(rest_lines_all)

    for s in range(len(final_lines)):
        x1, y1, x2, y2 = final_lines[s]
        cv2.line(im_tile, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # writing the concatenated slices into a folder
    cv2.imwrite(concatenated_images + file_names_bm[n] + '.jpg', im_tile)
    # path to the source images
    path_bg = source_image + file_names_bm[n] + '.jpg'
    bg = cv2.imread(path_bg)
    background = Image.open(path_bg)
    # resizing the concatenated image to the size of the source image
    fg_resize = cv2.resize(im_tile, background.size)
    # superimposing the concatenated image on the source image
    result = cv2.addWeighted(bg, 0.8, fg_resize, 0.5, 0.8)
    # writing the superimposed images into the folder
    cv2.imwrite(visualized_image + file_names_bm[n] + '.jpg', result)

k = cv2.waitKey(0)
cv2.destroyAllWindows()









