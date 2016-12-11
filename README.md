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

##### References & Libraries
* [dotabuff](https://www.dotabuff.com/) - In depth match analysis and statistics
* [dota2api](https://dota2api.readthedocs.io/en/latest/) - A python library for handling requests from Valve's Web API
* [bpnn](https://gist.github.com/yusugomori/2501438) - Back-propagate neural network example in python
* [Steam WebAPI](https://developer.valvesoftware.com/wiki/Steam_Web_API) - Providing methods to get the data for this project


