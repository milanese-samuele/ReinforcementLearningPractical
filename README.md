# ReinforcementLearningPractical
Our code is divided into three main part. 
The first one is the **main.py** where all the control calls are made. There you can adjust the hyperparameters of the different algorithms, the parameters for the experiment and the type of the reward function (Gaussian or Bernoulli). In addition, the plots are also made in this class.

The second part is the **bandit.py** which is the class for the rewards. Here the parameters of the Gaussian and Bernoulli reward function can be set (mu, sigma, etc.).

The last part is the different types of algorithms. As one can see there are four different files for them.

* Greedy: This file contains the greedy, epsilon-greedy and the optimistic initial value classes
* UCB: This file contains the class for the Upper Confidence Bound algorithm
* Softmax: This file contains the class for the SoftMax algorithm
* Action_Preferences: This file contains the class for the Action Preferences algorithm

Note that the running time of the algorithm can be around 60 seconds (also differs on the parameters, 60 second is with default) 

