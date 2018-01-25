# A simulation of multi-armed bandit problem. 

While A/B testing with frequentist and Bayesian methods can be incredibly useful for determining the effectiveness of various changes to your products, better algorithms exist for making educated decision on-the-fly. Two such algorithms that typically out-perform A/B tests are extensions of the Multi-armed bandit problem which uses an epsilon-greedy strategy. Using a combination of exploration and exploitation, this strategy updates the model with each successive test, leading to higher overall click-through rate. An improvement on this algorithm uses an epsilon-first strategy called UCB1. Both can be used in lieu of traditional A/B testing to optimize products and click-through rates.

### Use this code you can see how different strategies
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
Available strategy:  epsilon-greedy, softmax, ucb1 and bayesian bandits
