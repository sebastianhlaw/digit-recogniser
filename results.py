# Author:   Sebastian Law
# Date:     22-Nov-2016
# Revised:  16-Dec-2016

import os
import csv
# import datetime


def dump(labels, name, folder="results", description=None):
    # timestamp = '{:%Y-%m-%d %H%M%S}'.format(datetime.datetime.now())
    file_name = os.path.join(".", folder, str(name)+".csv")
    with open(file_name, "w", newline='') as file:
        output = csv.writer(file)
        output.writerow(["ImageId", "Label"])
        for i, y in enumerate(labels):
            output.writerow([i+1, y])
    file_name = os.path.join(".", folder, str(name) + ".txt")
    if description:
        with open(file_name, "w") as file:
            file.write(description)
