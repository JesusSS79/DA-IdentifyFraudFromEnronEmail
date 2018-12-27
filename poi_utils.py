#!/usr/bin/python

import matplotlib.pyplot as plt

def showOnPlot(features, labels, x, y, features_list):
    i = 0
    points = 0
    xva_poi = []
    yva_poi = []
    xva_npoi = []
    yva_npoi = []
    for feature in features:
        poi = labels[i]
        i += 1
        xv = feature[x]
        yv = feature[y]

        if (yv > 6000000):
            print xv, yv
        if (yv < 0):
            print xv, yv

        if yv != 'NaN':
            points += 1
            if poi == 1:
                xva_poi.append(xv)
                yva_poi.append(yv)
            else:
                xva_npoi.append(xv)
                yva_npoi.append(yv)

        # plt.plot(ages, reg.predict(ages), color="blue")
    print "Points on Plot:", points
    plt.scatter(xva_poi, yva_poi, color="r", label="POI")
    plt.scatter(xva_npoi, yva_npoi, color="b", label="NO POI")
    plt.legend()
    plt.xlabel(features_list[x+1])
    plt.ylabel(features_list[y+1])
    plt.show()


def showNaNs(features, features_list):
    number_values = len(features[0])
    nans = []

    for i in range(number_values):
        nans.append(0)

    for person in features:
        i = 0
        for value in person:
            if value == 0.0:
                nans[i] += 1
            i += 1

    print "NaN Percent"
    for i in range(len(nans)):
        print round(nans[i] / float(len(features)) * 100, 1), features_list[i]


def include_perc_poi_messages(dataset, prop_name, num_prop, den_prop):
    for person in dataset.values():
        num = 0
        den = 0
        for prop in num_prop:
            value = person[prop]
            if value != 'NaN':
                num += value

        for prop in den_prop:
            value = person[prop]
            if value != 'NaN':
                den += value

        if (den == 0):
            perc = 0
        else:
            perc = num/float(den)

        person[prop_name] = perc
