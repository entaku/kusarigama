__author__ = 'team-entaku'

import cv2
import sys
import selectivesearch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    file_name = sys.argv[1]
    im = plt.imread(file_name)

    print "here1"
    img_lbl, regions = selectivesearch.selective_search(
        im, scale=500, sigma=0.9, min_size=10)

    print "here2"
    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # distorted rects
        x, y, w, h = r['rect']
        if h < 1 or w < 1:
            continue
        if w / h > 1.2 or h / w > 1.2:
            continue
        candidates.add(r['rect'])

    print "here"
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(im)



