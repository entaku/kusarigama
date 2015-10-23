__author__ = 'team-entaku'

import cv2
import sys

if __name__ == "__main__":
    file_name = sys.argv[1]
    im = cv2.imread(file_name, 0)
    cv2.imshow('im', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
