import numpy as np
import matplotlib.pyplot as plt

greyhounds = 500 # sample population
labs = 500

greyHeight = 28 + 4 * np.random.randn(greyhounds) # change to make it realistic for actual data, but random ints added to increase variation
labHeight = 24 + 4 * np.random.randn(labs) # random adds height to entire population

plt.hist([greyHeight, labHeight], stacked = True, color = ['r', 'b']) # Histogram to represent data
plt.show()