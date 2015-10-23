__author__ = 'team-entaku'

import cv2
import numpy as np


def process_mser(orig, delta, min_area, max_area, max_variation, min_diversity, max_evolution, area_threshold,
                 min_margin, edge_blur_size):
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    mser = cv2.MSER(delta, min_area, max_area, max_variation, min_diversity, max_evolution, area_threshold,
                    min_margin, edge_blur_size)
    regions = mser.detect(gray, None)
    rects = [cv2.minAreaRect(r.reshape(-1, 1, 2)) for r in regions]
    boxes = [np.int0(cv2.cv.BoxPoints(rect)) for rect in rects]
    return boxes


def run_mser_video(
        delta,
        min_area,
        max_area,
        max_variation,
        min_diversity,
        max_evolution,
        area_threshold,
        min_margin,
        edge_blur_size):
    cap = cv2.VideoCapture(0)

    while True:
        res, orig = cap.read()
        boxes = process_mser(orig,
                             delta,
                             min_area,
                             max_area,
                             max_variation,
                             min_diversity,
                             max_evolution,
                             area_threshold,
                             min_margin, edge_blur_size)
        im = orig

        cv2.drawContours(im, boxes, -1, (0, 255, 0), 1)
        cv2.imshow('img', im)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_video():
    delta = 5
    min_area = 60
    max_area = 2000
    max_variation = 0.25
    min_diversity = 0.2
    max_evolution = 200
    area_threshold = 1.01
    min_margin = 0.003
    edge_blur_size = 5
    run_mser_video(delta, min_area, max_area, max_variation, min_diversity, max_evolution, area_threshold,
                   min_margin,
                   edge_blur_size)


if __name__ == "__main__":
    run_video()
