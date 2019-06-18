'''
Created on Tue Jun 18 11:29:07 2019
@coded by: yudhiprabowo
'''

import numpy as np
import cv2
import calculation as calc

inp = "F:\\latihan\\hazeremoval\\1.jpg"
out = "F:\\latihan\\hazeremoval\\1_hr.jpg"

imgi = cv2.imread(inp)
row, col, band = imgi.shape
sw = 7
rad = 7
eps = 0.001

imgo = np.asarray(calc.darkchannelprior(imgi, row, col, band, sw, rad, eps))
cv2.imwrite(out, imgo)
