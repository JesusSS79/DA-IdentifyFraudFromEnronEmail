#!/usr/bin/python

import matplotlib.pyplot as plt

def show_on_plot(features, labels, x, y, features_list):
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

        if yv != 'NaN':
            points += 1
            if poi == 1:
                xva_poi.append(xv)
                yva_poi.append(yv)
            else:
                xva_npoi.append(xv)
                yva_npoi.append(yv)

        # plt.plot(ages, reg.predict(ages), color="blue")
    #print "Points on Plot:", points
    plt.scatter(xva_poi, yva_poi, color="r", label="POI")
    plt.scatter(xva_npoi, yva_npoi, color="b", label="NO POI")
    plt.legend()
    plt.xlabel(features_list[x+1])
    plt.ylabel(features_list[y+1])
    plt.show()


def show_nans(features, features_list):
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


def show_poi_balance(labels):
    count = 0
    for label in labels:
        if label == 1:
            count += 1
    print "POI Balance:", round(100*count/float(len(labels)), 2), "(", count, "/", len(labels), ")"


def show_person_names(data):
    for person in data.keys():
        print person


def show_person_properties(dataset, prop_name, threshold, conditional):
    for key, values in dataset.items():
        if (values[prop_name] != 'NaN'):
            if ((conditional == 'upper') and (float(values[prop_name]) >= threshold)) or \
               ((conditional == 'lower') and (float(values[prop_name]) <= threshold)):
                print key, values


def include_div_poi(dataset, features_list, prop_name, num_prop, den_prop):
    features_list.append(prop_name)
    for person in dataset.values():
        num = 0
        den = 0
        valid = True
        for prop in num_prop:
            value = person[prop]
            if value != 'NaN':
                num += value
            else:
                valid = False

        for prop in den_prop:
            value = person[prop]
            if value != 'NaN':
                den += value
            else:
                valid = False

        if (den == 0) or valid is False:
            perc = 'NaN'
        else:
            perc = round(num/float(den), 3)

        person[prop_name] = perc


def include_add_poi(dataset, features_list, prop_name, prop_list):
    features_list.append(prop_name)
    for person in dataset.values():
        valid = True
        for prop in prop_list:
            if person[prop] == 'NaN':
                valid = False
        if valid:
            person[prop_name] = sum([person[prop] for prop in prop_list])
        else:
            person[prop_name] = 'NaN'
