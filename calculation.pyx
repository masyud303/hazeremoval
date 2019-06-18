#cython: language_level=3
'''
Created on Tue Jun 18 11:29:07 2019
@coded by: yudhiprabowo
'''

import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef unsigned char[:,::1] darkchannel(unsigned char[:,:,::1] img, unsigned char[:,::1] dark,
                  int row, int col, int band, int sw):
    cdef int i, j, k, i1, i2, j1, j2, m, n, minv, pad
    
    pad = sw // 2
    for i in range(row):
        i1 = i - pad
        if(i1 < 0): i1 = 0
        i2 = i + pad
        if(i2 >= row): i2 = row
        for j in range(col):
            j1 = j - pad
            if(j1 < 0): j1 = 0
            j2 = j + pad
            if(j2 >= col): j2 = col
            minv = 255
            for k in range(band):
                for m in range(i1, i2):
                    for n in range(j1, j2):
                        if(img[m, n, k] < minv):
                            minv = img[m, n, k]
            dark[i, j] = minv
            
    return dark
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef unsigned char[::1] atmlight(unsigned char[:,:,::1] img, unsigned char[:,::1] dark,
                  unsigned long[::1] hist, unsigned char[::1] atm, int row, int col, int band):
    cdef int i, j, k, thd, cum, jmlh0, toplist
    
    for i in range(row):
        for j in range(col):
            hist[dark[i, j]] += 1

    cum = 0
    toplist = (row * col) // 1000
    for i in range(255, 0, -1):
        cum += hist[i]
        if(cum >= toplist): 
            thd = i+1
            break
        
    jmlh0 = 0
    for i in range(row):
        for j in range(col):
            if(dark[i, j] >= thd):
                jmlh = img[i, j, 0] + img[i, j, 1] + img[i, j, 2]
                if(jmlh > jmlh0):
                    jmlh0 = jmlh
                    for k in range(band):
                        atm[k] = img[i, j, k]
    
    return atm

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef float[:,::1] transmissionmap(unsigned char[:,:,::1] img, unsigned char[::1] atm,
                  float[:,::1] tmap, int row, int col, int band, int sw):
    cdef:
        int i, j, k, i1, i2, j1, j2, m, n, pad
        float minv, omega, dn0
        
    omega = 0.85
    pad = sw // 2
    for i in range(row):
        i1 = i - pad
        if(i1 < 0): i1 = 0
        i2 = i + pad
        if(i2 >= row): i2 = row
        for j in range(col):
            j1 = j - pad
            if(j1 < 0): j1 = 0
            j2 = j + pad
            if(j2 >= col): j2 = col
            minv = 255
            for k in range(band):
                dn0 = float(img[m, n, k]) / atm[k]
                for m in range(i1, i2):
                    for n in range(j1, j2):
                        if(dn0 < minv):
                            minv = dn0
            tmap[i, j] = 1 - (omega * minv)
    
    return tmap

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef float[:,:,::1] guidedfilter(unsigned char[:,:,::1] img, float[:,::1] tmap, float[:,:,::1] reftmap,
          float[:,::1] coef_a, float[:,::1] coef_b, int row, int col, int band, int rad, int eps):
    cdef:
        int i, j, k, m, n, num
        float sum_i, sum_p, sum_ii, sum_ip, sum_a, sum_b
        float mean_i, mean_p, mean_ii, mean_ip, cov_ip, var_i, mean_a, mean_b
    
    for k in range(band):
        for i in range(row):
            i1 = i - rad
            if(i1 < 0): i1 = 0
            i2 = i + rad
            if(i2 >= row): i2 = row
            for j in range(col):
                j1 = j - rad
                if(j1 < 0): j1 = 0
                j2 = j + rad
                if(j2 >= col): j2 = col
                sum_i = 0
                sum_p = 0
                sum_ii = 0
                sum_ip = 0
                num = 0
                for m in range(i1, i2):
                    for n in range(j1, j2):
                        sum_i += img[m, n, k]
                        sum_p += tmap[m, n]
                        sum_ii += img[m, n, k] ** 2
                        sum_ip += img[m, n, k] * tmap[m, n]
                        num += 1
                mean_i = sum_i / num
                mean_p = sum_p / num
                mean_ii = sum_ii / num
                mean_ip = sum_ip / num
                cov_ip = mean_ip - (mean_i * mean_p)
                var_i = mean_ii - (mean_i * mean_i)
                coef_a[i, j] = cov_ip / (var_i + eps)
                coef_b[i, j] = mean_p - coef_a[i, j] * mean_i
        
        for i in range(row):
            i1 = i - rad
            if(i1 < 0): i1 = 0
            i2 = i + rad
            if(i2 >= row): i2 = row
            for j in range(col):
                j1 = j - rad
                if(j1 < 0): j1 = 0
                j2 = j + rad
                if(j2 >= col): j2 = col
                sum_a = 0
                sum_b = 0
                num = 0
                for m in range(i1, i2):
                    for n in range(j1, j2):
                        sum_a += coef_a[m, n]
                        sum_b += coef_b[m, n]
                        num += 1
                mean_a = sum_a / num
                mean_b = sum_b / num
                reftmap[i, j, k] = mean_a * img[i, j, k] + mean_b
            
    return reftmap

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef unsigned char[:,:,::1] recoverimg(unsigned char[:,:,::1] img, unsigned char[::1] atm, float[:,:,::1] reftmap,
                  unsigned char[:,:,::1] hzremov, int row, int col, int band):
    cdef:
        int i, j, k
        float t0, tmap, hzrem
        
    t0 = 0.1
    for i in range(row):
        for j in range(col):
            for k in range(band):
                if(reftmap[i, j, k] >= t0):
                    tmap = reftmap[i, j, k]
                else:
                    tmap = t0
                hzrem = ((float(img[i, j, k]) - atm[k]) / tmap) + atm[k]
                if(hzrem < 0):
                    hzrem = 0
                if(hzrem > 255):
                    hzrem = 255
                hzremov[i, j, k] = int(hzrem)
                
    return hzremov        
    
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef unsigned char[:,:,::1] darkchannelprior(unsigned char[:,:,::1] img, int row, int col, int band,
                   int sw, int rad, int eps):
    cdef:
        unsigned char[::1] atm
        unsigned char[:,::1] dark
        unsigned char[:,:,::1] hzremov
        unsigned long[::1] hist
        float[:,::1] tmap, coef_a, coef_b
        float[:,:,::1] reftmap
        
    atm = np.zeros(band, dtype=np.uint8)
    dark = np.zeros((row, col), dtype=np.uint8)
    hist = np.zeros(256, dtype=np.uint32)
    tmap = np.zeros((row, col), dtype=np.float32)
    coef_a = np.zeros((row, col), dtype=np.float32)
    coef_b = np.zeros((row, col), dtype=np.float32)
    reftmap = np.zeros((row, col, band), dtype=np.float32)
    hzremov = np.zeros((row, col, band), dtype=np.uint8)
    
    dark = darkchannel(img, dark, row, col, band, sw)
    atm = atmlight(img, dark, hist, atm, row, col, band)
    tmap = transmissionmap(img, atm, tmap, row, col, band, sw)
    reftmap = guidedfilter(img, tmap, reftmap, coef_a, coef_b, row, col, band, rad, eps)
    hzremov = recoverimg(img, atm, reftmap, hzremov, row, col, band)
    
    return hzremov
