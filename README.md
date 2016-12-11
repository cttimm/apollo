# Dota 2 Neural Network
### Term Project - CSCI 3450 - cttimm4427@ung.edu
#### Anaylsis of Dota 2 match statistics using a back-propagating neural network
---
Acquiring matches from valve's server requires installing dota2api:
```
pip install dota2api
```

Additionally, a valve api key is required. [https://developer.valvesoftware.com/wiki/Steam_Web_API](https://developer.valvesoftware.com/wiki/Steam_Web_API)

---

[Presentation](https://prezi.com/s9_rpnywxfb1/present/?auth_key=4v968uh&follow=o8tgwlhu_7dy&kw=present-s9_rpnywxfb1&rc=ref-158339460)
#### Sample outputs:
[Pudge](https://github.com/cttimm/d2nn/blob/master/src/sample_pudge)

[Phantom Assassin](https://github.com/cttimm/d2nn/blob/master/src/sample_pa)

[Alchemist](https://github.com/cttimm/d2nn/blob/master/src/sample_alch)

The interface uses a two layer neural network initialized by calling the method NN:
```python
class NN:
    def __init__(self, n_input = 7, n_layer1 = 7,  n_layer2 = 7, n_output = 1):
        self.n_input = n_input + 1
        self.n_layer1 = n_layer1
        self.n_layer2 = n_layer2
        self.n_output = n_output
        # Activations for nodes
        self.a_input = [1.0] * self.n_input
        self.a_layer1 = [1.0] * self.n_layer1
        self.a_layer2 = [1.0] * self.n_layer2
        self.a_output = [1.0] * self.n_output
        # Initialize weight matrix
        self.input_weights = fill_matrix(self.n_input, self.n_layer1)
        self.hidden_weights = fill_matrix(self.n_layer1, self.n_layer2)
        self.output_weights = fill_matrix(self.n_layer2, self.n_output)
```


The rate of change is an attempt to quantify the activation weights of the nodes by measuring the output after a marginal change to each individual statistic. 
```Python

    def relative_weights(self, h = .25):
        # Iterate through stats in averages
        base = self.averages()
        temp = [0.0] * 7
        results = [0.0] * 7
        for i in range(len(base)):
            temp = base[:]
            temp[i] = base[i] + (base[i] * h)
            #  (f(x+h) - f(x)) / h 
            results[i] = (self.predict(temp) - self.predict(base)) / (base[i] * h)
        print(results)
```

The output shows the iterations in brackets, the error remaining, the prediction for the match, and finally the results of the weight calculations using the rate of change:
```
[4000]	Error 0.75638
[4200]	Error 0.90693
[4400]	Error 0.67979
[4600]	Error 0.57559
[4800]	Error 0.49088
Prediction for match: https://www.dotabuff.com/matches/2391623428
0.99995074594702
Prediction for match: https://www.dotabuff.com/matches/2488562753
-0.012561278813607782
Rate of change per statistic:
[4.509975583053128, -0.2146852718512037, 0.008550522019067992, -0.018841539009557356, 0.07093585133377683, -0.00747917806470218, -7.143313232355583e-05]

```
In this case there are two matches used as examples. [Match 1](https://www.dotabuff.com/matches/2391623428), [Match 2](https://www.dotabuff.com/matches/2488562753). The player in question is named 'nuII' and the hero being played is alchemist. The weight calculations are in the format ```[GPM, XPM, Kills, Deaths, Assists, Damage, Healing]```

##### References & Libraries
* [dotabuff](https://www.dotabuff.com/) - In depth match analysis and statistics
* [dota2api](https://dota2api.readthedocs.io/en/latest/) - A python library for handling requests from Valve's Web API
* [bpnn](https://gist.github.com/yusugomori/2501438) - Back-propagate neural network example in python
* [Steam WebAPI](https://developer.valvesoftware.com/wiki/Steam_Web_API) - Providing methods to get the data for this project


