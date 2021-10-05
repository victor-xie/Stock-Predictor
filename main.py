"""
Stock market prediction using Markov chains.

For each function, replace the return statement with your code.  Add
whatever helper functions you deem necessary.
"""

import comp140_module3 as stocks

#Require defaultdict
from collections import defaultdict

#Creating a test dataset
import random
test_list = [random.randrange(0,4) for i in range(10)]
#print(test_list)

### Model

def markov_chain(data, order):
    """
    Create a Markov chain with the given order from the given data.

    inputs:
        - data: a list of ints or floats representing previously collected data
        - order: an integer repesenting the desired order of the markov chain

    returns: a dictionary that represents the Markov chain
    """
    

    
    #Empty dictionary that will eventually hold the completed Markov chain
    markov = defaultdict(dict)
    
    #Iterate through the dataset, checking groups of n numbers
    #For every instance of the group appearing, add one to the markov dictionary
    
    dataset_size = len(data)
    #This counter marks the leftmost index being checked in a group of n elements
    #It must never go beyond dataset_size - order, because that would cause an IndexError
    left_bound = 0
    while left_bound <= dataset_size - order - 1:
        #Creates the key for the dictionary
        group = []
        for ele_num in range(order+1):
            group.append(data[left_bound + ele_num])
        
        next_ele = group[-1]
        group = group[:-1]
        group = tuple(group)
   
        
        #Increments the tally for the specific group of elements
        if next_ele in markov[group]:
            markov[group][next_ele] += 1
        else:
            markov[group][next_ele] = 1
        
        #Move the left_bound one right to start counting the next group
        left_bound += 1
        

    
    for group_value in markov.values():
        total_frequency = sum(group_value.values())
        for key in group_value:
            group_value[key] = group_value[key] / total_frequency


    #cleans up the dictionary as the defaultdict properties are no longer needed
    markov = dict(markov)
    
    return markov

print(markov_chain([1,1,3,1,1], 3))

### Predict

#Helper function that does weighted random choice
def weighted_random_choice(probs):
    """
    Performs weighted random choice based on a probability distribution
    provided by a dictionary.
    
    inputs:
        -probs: A dictionary with keys and their corresponding weighted
                probabilities
    returns: An integer, one of the keys chosen through weighted random
             choice
    """
    ans = 0
    rand = random.random()
    total = 0
    for key, val in probs.items():
        total += val
        if rand <= total:
            ans = key
            break
    return ans

def predict(model, last, num):
    """
    Predict the next num values given the model and the last values.

    inputs:
        - model: a dictionary representing a Markov chain
        - last: a list (with length of the order of the Markov chain)
                representing the previous states
        - num: an integer representing the number of desired future states

    returns: a list of integers that are the next num states
    """
    #Creates a list to store the predicted states
    next_states = []
    #Create a counter that holds the most recent group of states
    recent_states = last.copy()
    #Create a counter that holds how many future states you have yet
    #to predict
    yet_to_predict = num

    while yet_to_predict > 0:
        #Will hold the predicted states
        next_state = 0
        #Will hold the probability distribution
        prob_dist = {}
        
        #Use recent_states to access the 
        #corresponding probability distribution in model
        if tuple(recent_states) in model.keys():
            prob_dist = model[tuple(recent_states)]

             #Use probabilities to randomly predict a future state
            next_state = weighted_random_choice(prob_dist)
            #Change the group of states counter to hold this new future state
            #by appending the newly-predicted state and removing the first element
            #of the list
            recent_states.append(next_state)
            recent_states.pop(0)
        else:
            next_state = random.randrange(4)
            next_states.append(next_state)

       
        
        #Store the predicted state
        next_states.append(next_state)
        
        #Decrement the yet_to_predict counter
        yet_to_predict -= 1
        
    return next_states


### Error

def mse(result, expected):
    """
    Calculate the mean squared error between two data sets.

    The length of the inputs, result and expected, must be the same.

    inputs:
        - result: a list of integers or floats representing the actual output
        - expected: a list of integers or floats representing the predicted output

    returns: a float that is the mean squared error between the two data sets
    """
    size = len(expected)
    
    #Stores total squared error
    total_se = 0
    
    #Calculates total squared error
    for idx in range(size):
        total_se += (result[idx] - expected[idx]) ** 2
    
    #Divides by the number of elements in each data set to get MSE
    result = total_se / size
    return result


### Experiment

def run_experiment(train, order, test, future, actual, trials):
    """
    Run an experiment to predict the future of the test
    data given the training data.

    inputs:
        - train: a list of integers representing past stock price data
        - order: an integer representing the order of the markov chain
                 that will be used
        - test: a list of integers of length "order" representing past
                stock price data (different time period than "train")
        - future: an integer representing the number of future days to
                  predict
        - actual: a list representing the actual results for the next
                  "future" days
        - trials: an integer representing the number of trials to run

    returns: a float that is the mean squared error over the number of trials
    """
    #Store the average MSE across trials
    avg_mse = 0
    
    iterations = trials
    while iterations > 0:
        #Creates a Markov chain of order order based on the training data
        model = markov_chain(train, order)

        #Uses the model to predict future days based on the testing data
        predictions = predict(model, test, future)

        #Calculates meam squared error for this trial
        trial_mse = mse(predictions, actual)
        
        avg_mse += trial_mse
        iterations -= 1
    
    avg_mse = avg_mse / trials
    
    return avg_mse


### Application

def run():
    """
    Run application.

    You do not need to modify any code in this function.  You should
    feel free to look it over and understand it, though.
    """
    # Get the supported stock symbols
    symbols = stocks.get_supported_symbols()

    # Get stock data and process it

    # Training data
    changes = {}
    bins = {}
    for symbol in symbols:
        prices = stocks.get_historical_prices(symbol)
        changes[symbol] = stocks.compute_daily_change(prices)
        bins[symbol] = stocks.bin_daily_changes(changes[symbol])

    # Test data
    testchanges = {}
    testbins = {}
    for symbol in symbols:
        testprices = stocks.get_test_prices(symbol)
        testchanges[symbol] = stocks.compute_daily_change(testprices)
        testbins[symbol] = stocks.bin_daily_changes(testchanges[symbol])

    # Display data
    #   Comment these 2 lines out if you don't want to see the plots
    stocks.plot_daily_change(changes)
    stocks.plot_bin_histogram(bins)

    # Run experiments
    orders = [1, 3, 5, 7, 9]
    ntrials = 500
    days = 5

    for symbol in symbols:
        print(symbol)
        print("====")
        print("Actual:", testbins[symbol][-days:])
        for order in orders:
            error = run_experiment(bins[symbol], order,
                                   testbins[symbol][-order-days:-days], days,
                                   testbins[symbol][-days:], ntrials)
            print("Order", order, ":", error)
        print()

# You might want to comment out the call to run while you are
# developing your code.  Uncomment it when you are ready to run your
# code on the provided data.

run()
