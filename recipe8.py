# For Python 2 / 3 compatability
from __future__ import print_function

# Toy dataset.
# Format: each row is an example.
# The last column is the label.
# The first two columns are features.
training_data = [
    ["Green", 3, "Apple"],
    ["Yellow", 3, "Apple"],
    ["Red", 1, "Grape"],
    ["Red", 1, "Grape"],
    ["Yellow", 3, "Lemon"],
    ["Red", 3, "Apple"],
    ["Red", 3, "Apple"],
]

# print the tree
header = ["color", "diameter", "label"]


# print(header)


def unique_vals(rows, col):
    # Find the unique values for a column in a dataset.
    return set([row[col] for row in rows])


# unique_vals(training_data, 0)
# returns column 0 of training data

def class_counts(rows):
    # Counts the number of each type of example in a dataset.
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


# class_counts(training_data)
# returns how many times a label is present

def is_numeric(value):
    """Test if a value is numeric. (E.g Distinguish color names from diameter)"""
    return isinstance(value, int) or isinstance(value, float)


# Demo
# is_numeric(7)

# Represents how a question is determined
class Question:
    """A Question is used to partition a dataset.

    This class just records a 'column number' (e.g., 0 for Color) and a
    'column value' (e.g., Green). The 'match' method is used to compare
    the feature value in an example to the feature value stored in the
    question. See the demo below.
    """

    def __init__(self, column, value):
        # stores for the threshold used to partition the data
        self.column = column  # string of type of feature
        self.value = value  # value of the given node in the table

    def match(self, example):
        # Compare the feature value in an example to the
        # feature value in this question.
        val = example[self.column]
        # print(val) prints the feature at self.column and the given index of training_data
        if is_numeric(val):  # returns boolean values
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))


# Prints example questions
q = Question(0, 'Green')  # Is color == Green?
print(q)
print(Question(1, 3))  # Is diameter >= 3?

# An example from the training set to see if it matches the question
example = training_data[0]
print(q.match(example))  # this will be true, since the first example is Green.


def partition(rows, question):
    """Partitions a dataset.

    For each row in the dataset, check if it matches the question. If
    so, add it to 'true rows', otherwise, add it to 'false rows'.
    """
    true_rows, false_rows = [], []
    for row in rows:  # If the question is true for the given data, place it in true, otherwise it's false
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


# Partition the training data based on whether rows are Red.
true_rows, false_rows = partition(training_data, Question(0, 'Red'))


# print(true_rows) Contains rows with Red

def gini(rows):
    """Calculate the Gini Impurity for a list of rows.

    There are a few different ways to do this, this one was
    the most concise. See:
    https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    """
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl ** 2
    return impurity


# First, we'll look at a dataset with no mixing.
no_mixing = [['Apple'],
             ['Apple']]
# this will return 0
print(gini(no_mixing))

some_mixing = [['Apple'],
               ['Orange']]
# this will return 0.5 - meaning, there's a 50% chance of misclassifying
# a random example we draw from the dataset.
print(gini(some_mixing))

lots_of_mixing = [['Apple'],
                  ['Orange'],
                  ['Grape'],
                  ['Grapefruit'],
                  ['Blueberry']]
# This will return 0.8
print(gini(lots_of_mixing))


def info_gain(left, right, current_uncertainty):
    """Information Gain.

    The uncertainty of the starting node, minus the weighted impurity of
    two child nodes.
    """
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)


# Calculate the uncertainy of our training data.
current_uncertainty = gini(training_data)
print(current_uncertainty)


def find_best_split(rows):
    """Find the best question to ask by iterating over every feature / value
    and calculating the information gain."""
    best_gain = 0  # keep track of the best information gain
    best_question = None  # keep train of the feature / value that produced it
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1  # number of columns

    for col in range(n_features):  # for each feature

        values = set([row[col] for row in rows])  # unique values in the column

        for val in values:  # for each value

            question = Question(col, val)

            # try splitting the dataset
            true_rows, false_rows = partition(rows, question)

            # Skip this split if it doesn't divide the
            # dataset.
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate the information gain from this split
            gain = info_gain(true_rows, false_rows, current_uncertainty)

            # You actually can use '>' instead of '>=' here
            # but I wanted the tree to look a certain way for our
            # toy dataset.
            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question


# Find the best question to ask first for our toy dataset.
best_gain, best_question = find_best_split(training_data)
print(best_question)


class Leaf:
    """A Leaf node classifies data.

    This holds a dictionary of class (e.g., "Apple") -> number of times
    it appears in the rows from the training data that reach this leaf.
    """

    def __init__(self, rows):
        self.predictions = class_counts(rows)


class Decision_Node:
    """A Decision Node asks a question.

    This holds a reference to the question, and to the two child nodes.
    """

    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


def build_tree(rows):
    """Builds the tree.

    Rules of recursion: 1) Believe that it works. 2) Start by checking
    for the base case (no further information gain). 3) Prepare for
    giant stack traces.
    """

    # Try partitioing the dataset on each of the unique attribute,
    # calculate the information gain,
    # and return the question that produces the highest gain.
    gain, question = find_best_split(rows)

    # Base case: no further info gain
    # Since we can ask no further questions,
    # we'll return a leaf.
    if gain == 0:
        return Leaf(rows)

    # If we reach here, we have found a useful feature / value
    # to partition on.
    true_rows, false_rows = partition(rows, question)

    # Recursively build the true branch.
    true_branch = build_tree(true_rows)

    # Recursively build the false branch.
    false_branch = build_tree(false_rows)

    # Return a Question node.
    # This records the best feature / value to ask at this point,
    # as well as the branches to follow
    # dependingo on the answer.
    return Decision_Node(question, true_branch, false_branch)


# Recieves the entire training set as input
my_tree = build_tree(training_data)
print(my_tree)


def classify(row, node):
    """See the 'rules of recursion' above."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node.predictions

    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


# training data is an apple with confidence 1.
print(classify(training_data[0], my_tree))


def print_leaf(counts):
    """A nicer way to print the predictions at a leaf."""
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs


# Printing that a bit nicer
print(print_leaf(classify(training_data[0], my_tree)))

# Evaluate
testing_data = [
    ['Green', 3, 'Apple'],
    ['Yellow', 4, 'Apple'],
    ['Red', 2, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 3, 'Lemon'],
]

for row in testing_data:
    print("Actual: %s. Predicted: %s" %
          (row[-1], print_leaf(classify(row, my_tree))))
