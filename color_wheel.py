#!/usr/bin/env python3
# Licensed under the MIT license.
# Color wheel generation code adapted from http://www.ficml.org/jemimap/style/color/hsvwheel.phps , listed as "Free for any use"

# Setup:
# Install pip
# python3 -m pip install pypng
# python3 -m pip install numpy
# python3 -m pip install scikit-image


import png
import numpy as np
import math
import csv
import skimage
import argparse
import random

### TRANSFORMATION FUNCTIONS

#Color transformation functions
def hsb2rgb(h, s, b):
    maxi = round(b * 51/20)
    mini = round(maxi * (1 - s/100))
    if maxi == mini:
        return (maxi, maxi, maxi)
    d = maxi - mini
    h6 = h / 60
    if h6 <= 1:
        return (maxi, round(mini + h6 * d), mini)
    if h6 <= 2:
        return (round(mini - (h6 - 2) * d), maxi, mini)
    if h6 <= 3:
        return (mini, maxi, round(mini + (h6 - 2) * d))
    if h6 <= 4:
        return (mini, round(mini - (h6 - 4) * d), maxi)
    if h6 <= 5:
        return (round(mini + (h6 - 4) * d), mini, maxi)
    return (maxi, mini, round(mini - (h6 - 6) * d))

def rgb2hsb(r, g, b):
    maxi = 0
    mini = 0
    h6 = 0
    d = 0
    if r == g and r == b:
        maxi = r
        mini = r
        h6 = 0
    elif r >= g and g >= b:
        # 0 <= h6 <= 1
        maxi = r
        mini = b
        d = maxi - mini
        h6 = (g - mini)/d
    elif g >= r and r >= b:
        # 1 <= h6 <= 2
        maxi = g
        mini = b
        d = maxi - mini
        h6 = ((mini - r)/d) + 2
    elif g >= b and b >= r:
        # 2 <= h6 <= 3
        maxi = g
        mini = r
        d = maxi - mini
        h6 = ((b - mini)/d) + 2
    elif b >= g and g >= r:
        # 3 <= h6 <= 4
        maxi = b
        mini = r
        d = maxi - mini
        h6 = ((mini - g)/d) + 4
    elif b >= r and r >= g:
        # 4<= h6 <= 5
        maxi = b
        mini = g
        d = maxi - mini
        h6 = ((r - mini)/d) + 4
    else:
        # r > b and b > g:
        # 5 <= h6 <= 6
        maxi = r
        mini = g
        d = maxi - mini
        h6 = ((mini - b)/d) + 6

    if maxi == 0:
        return (h6 * 60, 0, 0)
    b = round(maxi * 20 / 51)
    s = round((1 - (mini/maxi)) * 100)
    return (round(h6 * 60), s, b)

#Color Wheel Transformation Functions
def polar2hsb(rn, ad):
    if rn >= .5:
        rs = (rn - .5) * 2
        return (ad, 100 - (rs * 100), 100)
    rb = rn * 2
    return (ad, 100, rb * 100)

def hsb2polar(h, s, b):
    if s < b:
        rs = (100 - s)/100
        rn = (rs / 2) + .5
        return (rn, h)
    rb = b / 100
    rn = rb / 2
    return (rn, h)

#Coordinate Transformation Functions
def xy2polar(x, y, mid):
    if x == 0 and y == 0:
        return (0, 0)
    x2 = x*x
    y2 = y*y
    rr = math.sqrt(x2 + y2)
    rn = rr/mid
    ar = math.acos(x/rr)
    arc = ar if y >= 0 else 2*math.pi - ar
    ad = math.degrees(arc)
    return (rn, ad)

def polar2xy(rn, ad, mid):
    arc = math.radians(ad)
    x = rn * math.cos(arc) * mid
    y = rn * math.sin(arc) * mid
    return (round(x), round(y))


### IMAGE PROCESSING FUNCTIONS

# Supersample img into an image of size outsize
# Requires that img's size is outsize * ss_rate
def supersample(img, outsize, ss_rate):
    out_img = [[[0,0,0,0] for i in range(outsize)] for i in range(outsize)]
    ss2 = ss_rate * ss_rate
    for x in range(outsize):
        for y in range(outsize):
            r = 0
            g = 0
            b = 0
            a = 0
            for xi in range(ss_rate):
                for yi in range(ss_rate):
                    pixel = img[y * ss_rate + yi][x * ss_rate + xi]
                    r = r + pixel[0]
                    g = g + pixel[1]
                    b = b + pixel[2]
                    a = a + pixel[3]
            out_img[y][x] = [r // ss2, g // ss2, b // ss2, a // ss2]
    return out_img

# Flatten img from a 3d array to 2d packed
def flatten(img, size):
    out_img = []
    for y in range(size):
        out_img.append([img[y][x // 4][x % 4] for x in range(size * 4)])
    return out_img

def draw_diamond_impl(img, size, center_x, center_y, s, color):
    rows = [0 for i in range(4)]
    cols = [0 for i in range(4)]
    rows[0] = center_x + s
    cols[0] = center_y

    rows[1] = center_x
    cols[1] = center_y + s

    rows[2] = center_x - s
    cols[2] = center_y

    rows[3] = center_x
    cols[3] = center_y - s

    rr, cc = skimage.draw.polygon(rows, cols)
    img[cc, rr] = color

def draw_diamond(img, size, center_x, center_y, r, t):
    draw_diamond_impl(img, size, center_x, center_y, r, [0, 0, 0, 255])
    draw_diamond_impl(img, size, center_x, center_y, r - t, [255, 255, 255, 255])

#Draw a circle on the image centered at location center_x,center_y with radius r
def draw_circle_impl(img, size, center_x, center_y, r, color):
    rr, cc = skimage.draw.disk((center_x, center_y), r, shape = img.shape)
    img[cc, rr] = color

def draw_circle(img, size, center_x, center_y, r, t):
    draw_circle_impl(img, size, center_x, center_y, r, [0, 0, 0, 255])
    draw_circle_impl(img, size, center_x, center_y, r - t, [255, 255, 255, 255])

#Draw a box on the image centered at location center_x,center_y with side length 2s
def draw_box_impl(img, size, center_x, center_y, s, color):
    bottom_left = (center_x - s, center_y - s)
    top_right = (center_x + s, center_y + s)
    rr, cc = skimage.draw.rectangle(bottom_left, end = top_right, shape = img.shape)
    img[cc, rr] = color

def draw_box(img, size, center_x, center_y, s, t):
    draw_box_impl(img, size, center_x, center_y, s, [0, 0, 0, 255])
    draw_box_impl(img, size, center_x, center_y, s - t, [255, 255, 255, 255])

#Draw a cross on the image centered at location center_x,center_y with size s
def draw_cross_impl(img, size, center_x, center_y, s, correction, color):
    rows = [0 for i in range(12)]
    cols = [0 for i in range(12)]

    cardinal_off = s // 2 - correction
    large_off = s - correction
    small_off = s // 2
    rows[0] = center_x + cardinal_off
    cols[0] = center_y
    rows[1] = center_x + large_off
    cols[1] = center_y + small_off
    rows[2] = center_x + small_off
    cols[2] = center_y + large_off

    rows[3] = center_x
    cols[3] = center_y + cardinal_off
    rows[4] = center_x - small_off
    cols[4] = center_y + large_off
    rows[5] = center_x - large_off
    cols[5] = center_y + small_off

    rows[6] = center_x - cardinal_off
    cols[6] = center_y
    rows[7] = center_x - large_off
    cols[7] = center_y - small_off
    rows[8] = center_x - small_off
    cols[8] = center_y - large_off

    rows[9] = center_x
    cols[9] = center_y - cardinal_off
    rows[10] = center_x + small_off
    cols[10] = center_y - large_off
    rows[11] = center_x + large_off
    cols[11] = center_y - small_off

    rr, cc = skimage.draw.polygon(rows, cols)
    img[cc, rr] = color

def draw_cross(img, size, center_x, center_y, s, t):
    draw_cross_impl(img, size, center_x, center_y, s, 0, [0, 0, 0, 255])
    draw_cross_impl(img, size, center_x, center_y, s, round(4*t/5), [255, 255, 255, 255])

#Draw a cross on the image centered at location center_x,center_y with size s
def draw_plus_impl(img, size, center_x, center_y, s, correction, color):
    rows = [0 for i in range(12)]
    cols = [0 for i in range(12)]

    large_off = s - correction
    small_off = s // 2 - correction
    rows[0] = center_x + large_off
    cols[0] = center_y + small_off
    rows[1] = center_x + small_off
    cols[1] = center_y + small_off
    rows[2] = center_x + small_off
    cols[2] = center_y + large_off

    rows[3] = center_x - small_off
    cols[3] = center_y + large_off
    rows[4] = center_x - small_off
    cols[4] = center_y + small_off
    rows[5] = center_x - large_off
    cols[5] = center_y + small_off

    rows[6] = center_x - large_off
    cols[6] = center_y - small_off
    rows[7] = center_x - small_off
    cols[7] = center_y - small_off
    rows[8] = center_x - small_off
    cols[8] = center_y - large_off

    rows[9] = center_x + small_off
    cols[9] = center_y - large_off
    rows[10] = center_x + small_off
    cols[10] = center_y - small_off
    rows[11] = center_x + large_off
    cols[11] = center_y - small_off

    rr, cc = skimage.draw.polygon(rows, cols)
    img[cc, rr] = color

def draw_plus(img, size, center_x, center_y, s, t):
    draw_plus_impl(img, size, center_x, center_y, s, 0, [0, 0, 0, 255])
    draw_plus_impl(img, size, center_x, center_y, s, round(4*t/5), [255, 255, 255, 255])

display_funcs = [draw_box, draw_cross, draw_plus, draw_diamond]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = 'color_wheel',
        description = 'Generates a color wheel and places symbols on it at points cooresponding to colors')
    parser.add_argument('infile', help='input CSV file with data points')
    parser.add_argument('outfile', help='output PNG file')
    parser.add_argument('-s', '--image-size', type=int, default=512, help='output image size in pixels')
    parser.add_argument('-q', '--quality', type=int, default=4, help='supersampling rate; decrease to speed up program, increase to make image smoother')
    parser.add_argument('-c', '--column', default='condition', help='name of column to use to differentiate symbols; case sensitive')
    parser.add_argument('-d', '--symbol-size', type=int, default=4, help='size to make symbols for each data point')
    parser.add_argument('-t', '--symbol-thickness', type=int, default=2, help='thickness of borders for each symbol')
    parser.add_argument('-v', '--verbose', default=0, action='count', help='print verbose runtime information')
    parser.add_argument('-x', '--hex-colors', default=False, action='store_true', help='use hex codes instead of r g b columns')
    parser.add_argument('-j', '--no-jitter', dest='jitter', default=True, action='store_false', help="don't jitter identicial values to make display easier")
    parser.add_argument('-w', '--no-white', default=False, action='store_true', help='exclude white or near-white values')
    args = parser.parse_args()

    size = args.image_size * args.quality
    symbol_size = args.symbol_size * args.quality
    symbol_thickness = args.symbol_thickness * args.quality
    mid = (size - (symbol_size * 3 + symbol_thickness) - 1) // 2

    print("Generating wheel..")
    img = np.zeros((size, size, 4), dtype=np.uint8)
    for rx in range(size):
        for ry in range(size):
            x = rx - mid
            y = mid - ry
            if x == 0 and y == 0:
                img[ry][rx] = [0,0,0,255]
                continue
            (rn, ad) = xy2polar(x, y, mid)
            if rn > 1:
                img[ry][rx] = [255,255,255,0]
                continue
            (h, s, b) = polar2hsb(rn, ad)
            (r, g, b) = hsb2rgb(h, s, b)
            img[ry][rx] = [r, g, b, 255]
            

    print("Processing data...")
    conditionmap = {}
    idx = 0
    with open(args.infile, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        if args.verbose >= 2:
            print("Field names: %s" % reader.fieldnames)
        if 'R' in reader.fieldnames:
            red = 'R'
        else:
            red = 'r'
        if 'G' in reader.fieldnames:
            green = 'G'
        else:
            green = 'g'
        if 'B' in reader.fieldnames:
            blue = 'B'
        else:
            blue = 'b'
        jitterset = set({})
        for row in reader:
            if row[args.column] not in conditionmap:
                if idx == len(display_funcs):
                    print("Error: Too many different values for condition, only %d shapes supported" % len(display_funcs))
                    exit(1)
                conditionmap[row[args.column]] = display_funcs[idx]
                idx = idx + 1
            func = conditionmap[row[args.column]]
            if args.hex_colors:
                (r, g, b) = tuple(int(row['hex'][i:i+2], 16) for i in (0, 2, 4))
                if args.verbose >= 2:
                    print("Mapping %s to %d %d %d" % (row['hex'], r, g, b), end='')
            else:
                r = int(row[red])
                g = int(row[green])
                b = int(row[blue])
                if args.verbose >= 2:
                    print("Mapping %d %d %d" % (r, g, b), end='')
            (h, s, b) = rgb2hsb(r, g, b)
            if args.verbose >= 3:
                print(" to %d %d %d" % (h, s, b), end='')
            (rn, ad) = hsb2polar(h, s, b)
            if args.verbose >= 3:
                print(" to %d %d" % (rn, ad), end='')
            (x, y) = polar2xy(rn, ad, mid)
            if args.verbose >=2:
                print(" to %d %d" % (x, y))
            if args.jitter:
                if (x, y) in jitterset:
                    dist = random.uniform(.8 * symbol_size, 2 * symbol_size)
                    ang = random.uniform(0, 2 * math.pi)
                    x += round(math.cos(ang) * dist)
                    y += round(math.sin(ang) * dist)
                    if args.verbose >= 2:
                        print("jittering by %d,%d" % (math.cos(ang) * dist, math.sin(ang) * dist))
                else:
                    jitterset.add((x,y))
            if rn >= .95 and args.no_white:
                if args.verbose >= 2:
                    print("Skipping due to whiteness")
                continue
            func(img, size, x + mid, mid - y, symbol_size, symbol_thickness)
        if args.verbose >= 1:
            for (k, v) in conditionmap.items():
                print("%s maps to %s" % (k, v.__name__))
    
    print("Downscaling...")
    if args.quality == 1:
        downscaled = img
    else:
        downscaled = supersample(img, args.image_size, args.quality)

    print("Flattening...")
    flattened = flatten(downscaled, args.image_size)
    
    w = png.Writer(width=args.image_size, height=args.image_size, greyscale=False, alpha=True, bitdepth=8)
    with open(args.outfile, 'wb') as pngfile:
        w.write(pngfile, flattened)
    exit(0)
