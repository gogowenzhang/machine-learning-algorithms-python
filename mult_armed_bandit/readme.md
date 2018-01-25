# A simulation of multi-armed bandit problem. 

# Use this code you can see how different strategies
performs in Multi-arm bandit problem. 

```
from bandits import Bandits
from banditstrategy import BanditStrategy

bandits = Bandits([0.05, 0.03, 0.06])
strat = BanditStrategy(bandits, 'random_choice')
strat.sample_bandits(1000)
print("Number of trials: ", strat.trials)
print("Number of wins: ", strat.wins)
print("Conversion rates: ", strat.wins / strat.trials)
print("A total of %d wins of %d trials." % \
    (strat.wins.sum(), strat.trials.sum()))
```
# Available strategy:  epsilon-greedy, softmax, ucb1 and bayesian bandits
