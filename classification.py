###########################
# Author: Te-Yuan Liu
###########################

###########################
# Import Packages
###########################
import json
import numpy as np
###########################
# Define Functions
###########################
def generate_X_y(filename):
    X_list, y_list = [], []
    counter = 0
    with open(filename+'.txt') as data:
        for line in data:
            line = json.loads(line)
            if counter > 1000:
                break
            if counter == 0:
                print(line['title'])
                print(line['tweet']['user']['location'])
            if line['tweet']['user']['location'] == 'Washington':
                X_list.append(line['title'])
                y_list.append(0.)
            elif line['tweet']['user']['location'] == 'Massachusetts':
                X_list.append(line['title'])
                y_list.append(1.)
            counter += 1
    return X_list, y_list


###########################
# Main
###########################
def main():
    print('starting part 2...')
    X_list, y_list = generate_X_y('tweets_#superbowl')
    print(len(X_list), len(y_list))
    print(y_list)
if __name__ == "__main__":
    main()




