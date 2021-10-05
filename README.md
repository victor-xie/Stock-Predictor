# Stock-Predictor
Algorithm that predicts future shifts in stock prices based on historical data by leveraging Markov chains.

### What are Markov Chains?
Markov chains are a stochastic mathematical model of states that randomly transitions between states. In other words, imagine a system of states labelled "A", "B", "C", etc. and arrows between those states. Each arrow would be labelled with the transition probability of moving from one state to the next. 

Higher-order Markov chains store memory of the last n states that have been traversed, which affects the prediction of the next state to come.

### How are Markov chains applied in this project?
By grouping the variations in stock prices into four "bins", 0, 1, 2, 3, which represent a greater than 1 percent decrease, less than 1 percent decrease, less than 1 percent increase, and greater than 1 percent increase, respectively, in a stock's price, we can apply higher-order Markov chains to predict future price fluctuations based on historical stock price data. 

### What functions are included in this project?
A function to generate the Markov chain model, a function to predict future stock price data based on said model, a function to generate mean-squared error, and a function to run trials that test the efficacy of the model based on minimizing mean-squared error.

### How can this project be tested?
Copy/paste the code into a compiler of your choice and add in test cases as you please. Note that while the Markov chain generator can intake any states, the other functions only provide correct results when restricted to the four bins (0, 1, 2, 3).

