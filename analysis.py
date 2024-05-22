#!/usr/bin/env python3
# Licensed under the MIT license.

from collections import defaultdict
import argparse
import csv
import itertools
import math

def calc_intra(luvs):
    out = []
    if len(luvs) <= 1:
        return out
    for i in range(len(luvs)):
        for j in range(i + 1, len(luvs)):
            out.append(math.dist(luvs[i], luvs[j]))
    return out

def calc_inter(luvs1, luvs2):
    out = []
    for prod in itertools.product(luvs1, luvs2):
        out.append(math.dist(prod[0], prod[1]))
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = 'analysis',
        description = 'Analyze intra- and inter-group color space distance')
    parser.add_argument('infile', help='input CSV file with data points')
    parser.add_argument('-v', '--verbose', default=0, action='count', help='print verbose runtime information')
    args = parser.parse_args()

    data = defaultdict(lambda: defaultdict(list))
    categories = ["Case", "Font", "Size", "Number", "Spacing"]
    with open(args.infile, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        if args.verbose >= 2:
            print("Field names: %s" % reader.fieldnames)

        for row in reader:
            L = float(row['L'])
            u = float(row['u'])
            v = float(row['v'])
            if args.verbose >= 2:
                print("Adding %d %d %d" % (L, u, v), end='')
            k1 = row['Vowel']
            for category in categories:
                k2 = "%s:%s" % (category, row[category])
                if args.verbose >= 2:
                    print(" to %s %s, " % (k1, k2), end='')
                data[k1][k2].append((L, u, v))
            if args.verbose >= 2:
                print()
    if args.verbose >= 3:
        print(data)

    intra_list = defaultdict(lambda: defaultdict(list))
    inter_list = defaultdict(lambda: defaultdict(list))
    for vowel in data.keys():
        for category in categories:
            values = set()
            for k2 in data[vowel].keys():
                if not k2.startswith("%s:" % category):
                    continue
                value = k2.split(':')[1]
                if args.verbose >= 2:
                    print("Calculating intra for %s %s:%s" % (vowel, category, value))
                intra_list[vowel][category] += calc_intra(data[vowel][k2])
                for ex in values:
                    if args.verbose >= 2:
                        print("Calculating inter for %s %s:(%s and %s)" % (vowel, category, value, ex))
                    inter_list[vowel][category] += calc_inter(data[vowel][k2], data[vowel]["%s:%s" % (category, ex)])
                values.add(value)

    if args.verbose >= 3:
        print("intra list: ", intra_list)
        print("inter list: ", inter_list)

    for vowel in data.keys():
        print("Vowel %s," % vowel)
        for category in categories:
            print("intra %s," % category, end='')
            print(','.join(map(str, intra_list[vowel][category])))
            print("inter %s," % category, end='')
            print(','.join(map(str, inter_list[vowel][category])))
        print(",")
