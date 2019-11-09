#!/usr/bin/python3

# Copyright (C) 2019 Dennis Salzmann
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import argparse
import json
import multiprocessing
import os
import pickle
import sys
import time
from datetime import datetime

from bs4 import BeautifulSoup
from scipy import sparse
from sklearn import svm
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import confusion_matrix

from static_analysis import start_static_analysis


class FileObject:

    def __init__(self, path, classification, string):
        self.path = path
        self.classification = classification
        self.parent_html = string

    def __repr__(self):
        return self.path + ' : ' + self.classification


class Model:

    def __init__(self):
        # the actual analysis data
        self.X = []
        # the real classification of the data
        self.Y = []
        # the file where the data comes from
        self.files = []
        # the parent if there is one (used for html analysis)
        self.parents = []

    def append_x(self, string):
        self.X.append(string)

    def append_y(self, string):
        self.Y.append(string)

    def append_file(self, string):
        self.files.append(string)

    def append_parent(self, string):
        self.parents.append(string)

    def my_print(self):
        for x, y in zip(self.X, self.Y):
            print(str(x) + " : " + str(y))
        print(str(len(self.X)) + " : " + str(len(self.Y)))


class Result:

    def __init__(self):
        # the files (paths) which were analysed
        self.files = []
        # the predictions of the svm
        self.predictions = []
        # the real classification
        self.classification = []

    def append_file(self, string):
        self.files.append(string)

    def append_prediction(self, string):
        self.predictions.append(string)

    def append_classification(self, string):
        self.classification.append(string)


# this function takes a path, a verbose flag and the, to the file, corresponding classification
# this function returns a list of file objects
def parse_html(path, verbose, classification):
    # read the html file
    with open(path) as f:
        content = f.read()
    # parse the html with beautifulsoup
    soup = BeautifulSoup(content, "html.parser")
    # find all the scripts
    scripts = soup.find_all('script')
    # create a path to save the extracted scripts to
    scripts_path = os.path.join(sys.path[0], 'tmp/')
    # if there are any scripts
    if len(scripts) > 0:
        # if the folder to save the scripts to does not yet exist
        if not os.path.exists(scripts_path):
            if verbose:
                print(str(datetime.now()) + ": Created directory for extracted script files.")

            # create the folder
            os.makedirs(scripts_path)
        if verbose:
            print(str(datetime.now()) + ": Found " + str(len(scripts)) + " script(s) in html file: " + path)

    # create the list to hold the file objects
    files = []
    # iterate over the scripts
    for counter, script in enumerate(scripts):
        # create the full path to save the script to
        s_path = os.path.join(scripts_path, (str(counter) + '_' + os.path.basename(path) + '.js'))
        # open/create the file (overwrites if already exists)
        with open(s_path, 'w') as script_file:
            if verbose:
                print(str(datetime.now()) + ": Saved script in file: " + script_file.name)

            # write the script into the file
            script_file.write(script.text)
            # append the file object to the files list
            files.append(FileObject(s_path, classification, path))
    return files


# this function takes a path, the corresponding classification, a verbose flag and a "p" flag
# this function returns a list of file objects
def parse_input(path, classification, verbose, p):
    # create a list to hold the file objects
    files = []
    # create a list to hold the file paths for the loop
    paths = [path]
    # as long as there are still paths to parse
    while len(paths) > 0:
        # iterate over all paths (copying the paths to not kill the iteration when deleting from the list)
        for path in paths[:]:
            # if the path actually exists
            if os.path.exists(path):
                # if the path is a file
                if os.path.isfile(path):
                    # if the filename ends with html
                    if path.lower().endswith('.html'):
                        if verbose:
                            print(str(datetime.now()) + ": Extracting scripts from file: " + path)
                        # call the parse html function and add the returned file objects to the files list
                        files += parse_html(path, verbose, classification)

                    # if the filename ends with js or bin
                    elif path.lower().endswith('.js') or path.lower().endswith('.bin'):
                        if verbose:
                            print(str(datetime.now()) + ": Found file: " + path + " with classification: "
                                  + classification)
                        if p:
                            # get the path of the file to find the folder name to use as parent
                            parent = os.path.abspath(os.path.join(path, os.pardir))
                            # create the file object and append it to the files list
                            files.append(FileObject(path, classification, parent))
                        else:
                            # create the file object and append it to the files list
                            files.append(FileObject(path, classification, ""))

                    # remove the handled path from the paths list
                    paths.remove(path)

                # if the path is a directory
                elif os.path.isdir(path):
                    if verbose:
                        print(str(datetime.now()) + ": Found directory: " + path)

                    # iterate over all childs of the directory
                    for file in os.listdir(path):
                        # append the path of the child to the paths to keep iterating over them
                        paths.append(os.path.join(path, file))

                    # remove this path from the paths
                    paths.remove(path)

                # if the path is neither directory nor file
                else:
                    if verbose:
                        print(str(datetime.now()) + ": Path (" + path + ") is not a file nor a directory, ignoring.")
                    paths.remove(path)

            # if the path does not exist
            else:
                if verbose:
                    print(str(datetime.now()) + ": Path (" + path + ") does not exist, ignoring.")
                paths.remove(path)

    # return the list of file objects
    return files


def main():
    # SECTION - ARGUMENT PARSING
    parser = argparse.ArgumentParser()
    parser.add_argument("-ai", "--analyse_input", nargs='*',
                        help="specifies the path(s) from which to load the input which is to be analyzed")
    parser.add_argument("-ti", "--train_input", nargs='*',
                        help="specifies the path(s) from which to load the input which gets used to train the model")
    parser.add_argument("-ac", "--analyse_classification", nargs='*',
                        help="specifies the classification per given analyzed input (same order)",
                        choices=["malicious", "benign"])
    parser.add_argument("-tc", "--train_classification", nargs='*',
                        help="specifies the classification per given trained input (same order)",
                        choices=["malicious", "benign"])
    parser.add_argument("--save_model",
                        help="indicates the path where the model should be saved (model will not be saved if empty)")
    parser.add_argument("--load_model",
                        help="specifies the path from which to load a saved model. If used together with train-input"
                             " they will be combined")
    parser.add_argument("-n", "--n_gram_size", type=int, help="indicates the size of the n-grams", default=3)
    parser.add_argument("-p", "--p", action='store_true',
                        help="indicates that html files should be looked at instead of the extracted scripts")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--ram_opt", action="store_true")
    parser.add_argument("--ram_window", type=int, default=1000)
    args = parser.parse_args()

    # SECTION - SANITY CHECKS on the arguments - i.e. is it even possible to run

    if not args.train_input and not args.load_model:
        print("There was nothing given to train the SVM.")
        sys.exit(1)

    if args.train_input and args.load_model:
        print("Loading a trained model and giving train input are mutually exclusive.")
        sys.exit(1)

    if args.train_input and args.train_classification:
        if len(args.train_input) != len(args.train_classification):
            print("There was a mismatch in the amount of training inputs (" + str(len(args.train_input)) +
                  ") and classifications (" + str(len(args.train_classification)) + ").")
            sys.exit(1)

    if args.analyse_classification:
        if len(args.analyse_input) != len(args.analyse_classification):
            print("There was a mismatch in the amount of analyze inputs (" + str(len(args.train_input)) +
                  ") and classifications (" + str(len(args.train_classification)) + ").")
            sys.exit(1)

    # SECTION - INPUT HANDLING
    if args.verbose:
        print(str(datetime.now()) + ": Starting to parse the inputs.")

    # create lists to hold the file paths which were given as arguments
    files_to_analyse = []
    files_to_train = []
    # if there were arguments for analysis
    if args.analyse_input:
        if args.verbose:
            print(str(datetime.now()) + ": Starting to parse the inputs to be analyzed.")

        # iterate over the given paths and their classifications
        for path, classification in zip(args.analyse_input, args.analyse_classification):
            # call the parse input function
            files = parse_input(path, classification, args.verbose, args.p)
            # append the returned file objects of parse input to the analysis files
            files_to_analyse += files

        if args.verbose:
            print(str(datetime.now()) + ": Finished parsing the inputs to be analyzed.")
        if args.debug:
            print(files_to_analyse)

    # if there were arguments for training
    if args.train_input:
        if args.verbose:
            print(str(datetime.now()) + ": Starting to parse the inputs for training.")

        # iterate over the given paths and their classifications
        for path, classification in zip(args.train_input, args.train_classification):
            # call the parse input function
            files = parse_input(path, classification, args.verbose, args.p)
            # append the returned file objects of parse input to the analysis files
            files_to_train += files

        if args.verbose:
            print(str(datetime.now()) + ": Finished parsing the inputs for training.")
        if args.debug:
            print(files_to_train)

    if args.verbose:
        print(str(datetime.now()) + ": Finished parsing the inputs.")

    # SECTION - Analysis processes
    # create a manager object which handles our multiprocess variables to keep them thread-safe
    manager = multiprocessing.Manager()
    # create a list for the analysis data which the manager will handle - i.e. the processes will write to
    analyse_data = manager.list()
    # create a list for the training data which the manager will handle - i.e. the processes will write to
    train_data = manager.list()
    # create a list to hold our processes
    processes = []
    # create a string for verbose/debug output
    string = ""


    # variable for the verbose/debug output of the processes
    string = "static"

    # if we have files for analysis
    if len(files_to_analyse) != 0:
        # create the process objects with the corresponding function target and args
        static_analyze_process = multiprocessing.Process(target=start_static_analysis,
                                                         args=(files_to_analyse, analyse_data,
                                                               args.debug, args.verbose, 'analysing'))
        processes.append(static_analyze_process)

    # if we have files for training
    if len(files_to_train) != 0:
         # create the process objects with the corresponding function target and args
        static_train_process = multiprocessing.Process(target=start_static_analysis,
                                                       args=(files_to_train, train_data,
                                                             args.debug, args.verbose, 'training'))
        processes.append(static_train_process)

    # handle the created processes
    if args.verbose:
        print(str(datetime.now()) + ": Starting processes for the " + string + " analysis.")

    # iterate over all the processes which were created and start them
    for process in processes:
        process.start()
    # as long as there are processes still running
    while len(multiprocessing.active_children()) > 0:
        if args.debug:
            print(str(datetime.now()) + ": Waiting for processes to finish.")
        # if there is only one process left check if it is the manager process which handles the multiprocess variables
        if len(multiprocessing.active_children()) == 1 and \
                (multiprocessing.active_children())[0].name == "SyncManager-1":
            # leave the while loop - this process will terminate as soon as we do not need the multiprocess variables
            #  any more
            break
        # add a sleep to not kill the cpu
        time.sleep(0.1)
    if args.verbose:
        print(str(datetime.now()) + ": Finished the " + string + " analysis.")

    # SECTION - PREPARE DATA FOR THE SVM
    # create the vectorizer object
    hash_vectorizer = HashingVectorizer(ngram_range=(args.n_gram_size, args.n_gram_size), token_pattern='\S+',
                                        norm='l1', alternate_sign=False, binary=True)
    if args.verbose:
        print(str(datetime.now()) + ": Starting the preparation of the data.")

    # SUBSECTION - Create/Load trainings data
    # if we are supposed to load the data for training
    if args.load_model:
        # open the passed file
        with open(args.load_model, 'br') as f:
            # load the data - train_matrix is a csr matrix and y the classifications
            train_matrix, train_y = pickle.load(f)
        if args.verbose:
            print(str(datetime.now()) + ": Loaded the specified model.")
    # if we are not supposed to load the data for training
    else:
        # create temporary variables to save the trainings data - x is a list of the outputs which get used to train
        # and y are the classifications
        train_x = []
        train_y = []
        # iterate over the data of the analysis and fill the temporary variables
        for item in train_data:
            train_x.append((json.loads(item))['output'])
            train_y.append((json.loads(item))['classification'])
        # create the csr matrix
        train_matrix = hash_vectorizer.transform(train_x)
        # if we are supposed to save the trainings data
        if args.save_model:
            # open the file
            with open(args.save_model, 'bw') as f:
                # save the data to the file
                pickle.dump((train_matrix, train_y), f)
            if args.verbose:
                print(str(datetime.now()) + ": Saved the model to the specified location.")

    # SUBSECTION - Create the analysis data
    # create a custom object to hold the necessary data
    analysed_model = Model()
    # iterate over the data of the analysis and fill the object
    for item in analyse_data:
        # this is the actual output of the analysis
        analysed_model.append_x((json.loads(item))['output'])
        # this is the corresponding classification of the classification
        analysed_model.append_y((json.loads(item))['classification'])
        # this is the path of the file which got analysed
        analysed_model.append_file((json.loads(item))['file'])
        # this is the path of the html from which the script got extracted - if there is one
        analysed_model.append_parent((json.loads(item))['parent_html'])
    if args.verbose:
        print(str(datetime.now()) + ": Starting the vectorizer.")

    # create a variable to hold the csr matrix of the data for analysis
    analyse_matrix = None
    # check if there is actual data in the model
    len_model_x = len(analysed_model.X)
    print(len_model_x, len(analysed_model.Y))  # [38250, 38249]
    if len_model_x != 0:
        # if we should conserve ram
        if args.ram_opt:
            # our sliding pointer
            current = 0
            # create the variable for the matrix
            analyse_matrix = None
            # do as long as our pointer + window is lower than the length of our data
            while current + args.ram_window <= len_model_x:
                if args.verbose:
                    print(str(datetime.now()) + ": Working on slice: " + str(current))

                # create a matrix from the data which our sliding window points to
                tmp_matrix = hash_vectorizer.transform(analysed_model.X[current:(current + args.ram_window)])
                # if we have not initialized the matrix yet
                if analyse_matrix is None:
                    # just write the tmp matrix to the variable
                    analyse_matrix = tmp_matrix

                # if the matrix is already initialized
                else:
                    if args.verbose:
                        print(str(datetime.now()) + ": Appending slice: " + str(current))

                    # append the tmp matrix to the total matrix
                    analyse_matrix = sparse.vstack((analyse_matrix, tmp_matrix), format='csr')

                # increase the pointer
                current += args.ram_window

            # check if there is still data left to transform
            if current < len_model_x:
                # transform the last of the data and append it to the total matrix
                tmp_matrix = hash_vectorizer.transform(analysed_model.X[current:len_model_x])
                analyse_matrix = sparse.vstack((analyse_matrix, tmp_matrix), format='csr')
        else:
            # create the matrix
            analyse_matrix = hash_vectorizer.transform(analysed_model.X)
    if args.verbose:
        print(str(datetime.now()) + ": Finished the preparation of the data.")

    # SECTION - SVM
    # if there is data to analyse
    if analyse_matrix is not None:
        # create the svm
        linear_svc = svm.LinearSVC()
        # train the svm
        linear_svc.fit(train_matrix, train_y)
        # analyse the analysis data
        prediction = linear_svc.predict(analyse_matrix)

        # if the p argument was given
        if args.p:
            # create a dict to hold the results
            results = {}
            # iterate over the analysis data and the corresponding predictions
            for file, parent, predict, classification in \
                    zip(analysed_model.files, analysed_model.parents, prediction, analysed_model.Y):
                # if there is no parent to the file - i.e. this is not a script which was extracted from html
                if parent == '':
                    # append the prediction and real classification to the results dict
                    results[file] = (predict, classification)
                # if there is a parent - i.e. this file was extracted from html
                else:
                    # if the prediction says it is malicious
                    if predict == 'malicious':
                        # one malicious prediction is enough to classify the whole parent as malicious
                        results[parent] = (predict, classification)
                        continue
                    # the prediction is benign
                    else:
                        # check if this parent is already in the results dict
                        if parent in results:
                            # there is already an entry, nothing to do because we only `potentially` change the entry
                            #  if the prediction is malicious
                            continue
                        # otherwise add it
                        else:
                            # predict is always benign here and there is no entry of the page yet
                            results[parent] = (predict, classification)

            # create the metrics variables
            true_negative, false_positive, false_negative, true_positive = 0, 0, 0, 0
            # iterate over the results and calculate the metrics
            for key, result in results.items():
                prediction = result[0]
                classification = result[1]
                if prediction == 'benign':
                    if prediction == classification:
                        true_negative += 1
                    else:
                        false_negative += 1
                else:
                    if prediction == classification:
                        true_positive += 1
                    else:
                        false_positive += 1
                # print the actual predictions
                print(key + ' : ' + prediction)

        # if the p argument was not given
        else:
            # output
            for file, predict in zip(analysed_model.files, prediction):
                #if predict == "malicious":  # HNS
                print(file + ' : ' + predict)
            # metrics
            c_matrix = confusion_matrix(analysed_model.Y, prediction, labels=['benign', 'malicious'])
            true_negative, false_positive, false_negative, true_positive = c_matrix.ravel()

        # print the metrics
        total = true_positive + true_negative + false_negative + false_positive
        print("Amount of True Negatives (Reality: benign - Prediction: benign): " + str(true_negative))
        print("Amount of False Positives (Reality: benign - Prediction: malicious): " + str(false_positive))
        print("Amount of False Negatives (Reality: malicious - Prediction: benign): " + str(false_negative))
        print("Amount of True Positives (Reality: malicious - Prediction: malicious): " + str(true_positive))
        print("Detection Accuracy: " + str((true_negative + true_positive) / total))


if __name__ == '__main__':
    main()
    sys.exit(0)
