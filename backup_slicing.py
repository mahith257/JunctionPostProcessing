import image_slicer
import cv2
import glob
import os
import numpy as np
from pathlib import Path
from PIL import Image
import math
import pandas as pd


# function for distance between two points
def dist(x, y, a, b):
    distance = math.sqrt((a - x) ** 2 + (b - y) ** 2)
    return distance


def get_lines(lines_in):
    return [l[0] for l in lines_in]


# function for concatenating the sliced images
def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])


kernel_1 = np.ones((5, 5), np.uint8)
images = []
# for appending the filenames of binary mask images(without extensions) in a list
file_names_bm = []
for img in glob.glob('binary_mask/*.jpg'):
    m = Path(img).stem
    file_names_bm.append(m)

# eroding the image and writing the eroded images into a folder
for n in range(len(file_names_bm)):
    image = cv2.imread('binary_mask/'+file_names_bm[n]+'.jpg')
    erode_image = cv2.erode(image, kernel_1, iterations=1)
    cv2.imwrite('erode_images/'+file_names_bm[n]+'.jpg', erode_image)

# skeletonization of images and writing them into a folder
for n in range(len(file_names_bm)):
    img = cv2.imread('binary_mask/'+file_names_bm[n]+'.jpg')
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
    cv2.imwrite('skeleton_images/'+file_names_bm[n]+'.jpg', skel)

for n in range(len(file_names_bm)):
    # for making a directory with filenames of binary mask images if directory already exists it shows an error
    # os.mkdir('skeleton_instance_sliced/'+file_names_bm[n])
    # slicing the binary mask images into 49 slices(7*7) using image_slicer
    tiles = image_slicer.slice('skeleton_images/'+file_names_bm[n]+'.jpg', 49, save=False)
    # saving the sliced images of binary mask in the respective folders
    image_slicer.save_tiles(tiles, directory="skeleton_slice/"+file_names_bm[n], prefix='slice', format="jpeg")
    rest_lines_all = []
    # appending the read sliced images in an empty list
    sliced_images = []
    # appending the filenames of the sliced images of a corresponding binary mask image in an empty list
    file_names = []
    for img in glob.glob('skeleton_slice/'+file_names_bm[n]+'/*.jpg'):
        z = Path(img).stem
        file_names.append(z)
        y = cv2.imread(img)
        sliced_images.append(y)

    # applying HoughLinesP on each and every sliced image of a corresponding binary mask or instance image
    for z in range(len(sliced_images)):
        img = sliced_images[z]
        all_lines = []
        parallel_lines = []
        rest_lines = []
        parallel_lines_drawn = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=20, maxLineGap=10)
        if np.any(lines):
            if len(get_lines(lines)) == 1:
                x1, y1, x2, y2 = get_lines(lines)[0]
                all_lines.append(np.array(get_lines(lines)[0]))
                # cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
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
                    # cv2.line(img, (x5, y5), (x6, y6), (255, 0, 0), 2)
                    parallel_lines_drawn.append([x5, y5, x6, y6])
                    parallel_lines.append(np.array(get_lines(lines)[0]))
                    parallel_lines.append(np.array(get_lines(lines)[1]))
                # else:
                #     cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                #     cv2.line(img, (x3, y3), (x4, y4), (0, 0, 255), 2)
            else:
                for line in get_lines(lines):
                    x1, y1, x2, y2 = line
                    all_lines.append(line)
                    for l in get_lines(lines):
                        x3, y3, x4, y4 = l
                        if (x3 != x1) and (y3 != y1) and (x4 != x2) and (y4 != y2):
                            diff = abs(((y2 - y1) / (x2 - x1)) - ((y4 - y3) / (x4 - x3)))
                            # if the lines are within 10% slope then create a median
                            if diff <= 0.1:
                                if (math.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)) <= (
                                math.sqrt((x4 - x1) ** 2 + (y4 - y1) ** 2)):
                                    x5 = round((x1 + x3) / 2)
                                    y5 = round((y1 + y3) / 2)
                                    x6 = round((x2 + x4) / 2)
                                    y6 = round((y2 + y4) / 2)
                                    # cv2.line(img, (x5, y5), (x6, y6), (255, 0, 0), 2)
                                    parallel_lines_drawn.append([x5, y5, x6, y6])
                                    parallel_lines.append(l)
                                    parallel_lines.append(line)
                                else:
                                    x5 = round((x1 + x4) / 2)
                                    y5 = round((y1 + y4) / 2)
                                    x6 = round((x2 + x3) / 2)
                                    y6 = round((y2 + y3) / 2)
                                    # cv2.line(img, (x5, y5), (x6, y6), (255, 0, 0), 2)
                                    parallel_lines_drawn.append([x5, y5, x6, y6])
                                    parallel_lines.append(l)
                                    parallel_lines.append(line)

            # all_lines = np.unique(all_lines, axis=0)
            # print(all_lines)
            parallel_lines_drawn = np.unique(parallel_lines_drawn, axis=0)
            parallel_lines = np.unique(parallel_lines, axis=0)

            # if there are no parallel lines
            if len(parallel_lines) == 0:
                rest_lines = all_lines
            else:
                for i in range(len(all_lines)):
                    for j in range(len(parallel_lines)):
                        if np.array_equal(all_lines[i], parallel_lines[j]):
                            break
                        else:
                            if j == len(parallel_lines) - 1:
                                rest_lines.append(all_lines[i])
            # rest_lines = np.unique(rest_lines, axis=0)
            # rest_lines_all = np.unique(rest_lines_all, axis=0)
            # print(rest_lines)
            # appending median lines(parallel_lines_drawn) into rest_lines
            for j in range(len(parallel_lines_drawn)):
                x1, y1, x2, y2 = parallel_lines_drawn[j]
                rest_lines.append(parallel_lines_drawn[j])

            # converting the parameters of points with respect to the concatenated image
            for k in range(7):
                if z == (7 * k):
                    for m in range(len(rest_lines)):
                        x1, y1, x2, y2 = rest_lines[m]
                        x1 = x1
                        y1 = y1 + (36 * k)
                        x2 = x2
                        y2 = y2 + (36 * k)
                        rest_lines[m] = [x1, y1, x2, y2]
                        rest_lines_all.append(rest_lines[m])
                elif z == ((7 * k) + 1):
                    for m in range(len(rest_lines)):
                        x1, y1, x2, y2 = rest_lines[m]
                        x1 = x1 + 73
                        y1 = y1 + (36 * k)
                        x2 = x2 + 73
                        y2 = y2 + (36 * k)
                        rest_lines[m] = [x1, y1, x2, y2]
                        rest_lines_all.append(rest_lines[m])
                elif z == ((7 * k) + 2):
                    for m in range(len(rest_lines)):
                        x1, y1, x2, y2 = rest_lines[m]
                        x1 = x1 + (73 * 2)
                        y1 = y1 + (36 * k)
                        x2 = x2 + (73 * 2)
                        y2 = y2 + (36 * k)
                        rest_lines[m] = [x1, y1, x2, y2]
                        rest_lines_all.append(rest_lines[m])
                elif z == ((7 * k) + 3):
                    for m in range(len(rest_lines)):
                        x1, y1, x2, y2 = rest_lines[m]
                        x1 = x1 + (73 * 3)
                        y1 = y1 + (36 * k)
                        x2 = x2 + (73 * 3)
                        y2 = y2 + (36 * k)
                        rest_lines[m] = [x1, y1, x2, y2]
                        rest_lines_all.append(rest_lines[m])
                elif z == ((7 * k) + 4):
                    for m in range(len(rest_lines)):
                        x1, y1, x2, y2 = rest_lines[m]
                        x1 = x1 + (73 * 4)
                        y1 = y1 + (36 * k)
                        x2 = x2 + (73 * 4)
                        y2 = y2 + (36 * k)
                        rest_lines[m] = [x1, y1, x2, y2]
                        rest_lines_all.append(rest_lines[m])
                elif z == ((7 * k) + 5):
                    for m in range(len(rest_lines)):
                        x1, y1, x2, y2 = rest_lines[m]
                        x1 = x1 + (73 * 5)
                        y1 = y1 + (36 * k)
                        x2 = x2 + (73 * 5)
                        y2 = y2 + (36 * k)
                        rest_lines[m] = [x1, y1, x2, y2]
                        rest_lines_all.append(rest_lines[m])
                elif z == ((7 * k) + 6):
                    for m in range(len(rest_lines)):
                        x1, y1, x2, y2 = rest_lines[m]
                        x1 = x1 + (73 * 6)
                        y1 = y1 + (36 * k)
                        x2 = x2 + (73 * 6)
                        y2 = y2 + (36 * k)
                        rest_lines[m] = [x1, y1, x2, y2]
                        rest_lines_all.append(rest_lines[m])

    # writing the sliced images with detected lines into the corresponding binary mask or instance mask image folders
    for k in range(len(sliced_images)):
        path = 'skeleton_slice/'+file_names_bm[n]+'/'+file_names[k] + '.jpg'
        # cv_img[k] = cv2.cvtColor(cv_img[k], cv2.COLOR_BGR2RGB)
        cv2.imwrite(path, sliced_images[k])

    # concatenating the slices
    im_tile = concat_tile([[sliced_images[0], sliced_images[1], sliced_images[2], sliced_images[3], sliced_images[4], sliced_images[5], sliced_images[6]],
                           [sliced_images[7], sliced_images[8], sliced_images[9], sliced_images[10], sliced_images[11], sliced_images[12], sliced_images[13]],
                           [sliced_images[14], sliced_images[15], sliced_images[16], sliced_images[17], sliced_images[18], sliced_images[19], sliced_images[20]],
                           [sliced_images[21], sliced_images[22], sliced_images[23], sliced_images[24], sliced_images[25], sliced_images[26], sliced_images[27]],
                           [sliced_images[28], sliced_images[29], sliced_images[30], sliced_images[31], sliced_images[32], sliced_images[33], sliced_images[34]],
                           [sliced_images[35], sliced_images[36], sliced_images[37], sliced_images[38], sliced_images[39], sliced_images[40], sliced_images[41]],
                           [sliced_images[42], sliced_images[43], sliced_images[44], sliced_images[45], sliced_images[46], sliced_images[47], sliced_images[48]]])

    fit_lines = []
    final_lines_all = []
    for s in range(len(rest_lines_all)):
        x7, y7, x8, y8 = rest_lines_all[s]
        for h in range(len(rest_lines_all)):
            x9, y9, x10, y10 = rest_lines_all[h]
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
                                cv2.line(im_tile, (h1, k1), (h2, k2), (0, 255, 255), 2)
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
                                cv2.line(im_tile, (h1, k1), (h2, k2), (0, 255, 255), 2)
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
                                cv2.line(im_tile, (h1, k1), (h2, k2), (0, 255, 255), 2)
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
                                cv2.line(im_tile, (h1, k1), (h2, k2), (0, 255, 255), 2)

    fit_lines = np.unique(fit_lines, axis=0)
    rest_lines_all = np.unique(rest_lines_all, axis=0)
    for i in range(len(rest_lines_all)):
        for j in range(len(fit_lines)):
            if np.array_equal(rest_lines_all[i], fit_lines[j]):
                break
            else:
                if j == len(fit_lines) - 1:
                    final_lines_all.append(rest_lines_all[i])
    # print(len(rest_lines_all))
    # print(len(fit_lines))
    # print(len(final_lines_all))
    for s in range(len(final_lines_all)):
        x1, y1, x2, y2 = final_lines_all[s]
        cv2.line(im_tile, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # writing the concatenated slices into a folder
    cv2.imwrite('concatenated_skeleton_images/'+file_names_bm[n]+'.jpg', im_tile)
    # path to the source images
    path_bg = 'source_image/' + file_names_bm[n] + '.jpg'
    bg = cv2.imread(path_bg)
    background = Image.open(path_bg)
    # resizing the concatenated image to the size of the source image
    fg_resize = cv2.resize(im_tile, background.size)
    # superimposing the concatenated image on the source image
    result = cv2.addWeighted(bg, 0.8, fg_resize, 0.5, 0.8)
    # writing the superimposed images into the folder
    cv2.imwrite('visualized_skeleton_images/'+file_names_bm[n]+'.jpg', result)

k = cv2.waitKey(0)
cv2.destroyAllWindows()
