# Dota 2 Neural Network
### Term Project - CSCI 3450 - cttimm4427@ung.edu
#### Anaylsis of Dota 2 match statistics using a back-propagating neural network
---

[Presentation](https://prezi.com/s9_rpnywxfb1/present/?auth_key=4v968uh&follow=o8tgwlhu_7dy&kw=present-s9_rpnywxfb1&rc=ref-158339460)
#### Sample outputs:
[Pudge](https://github.com/cttimm/d2nn/blob/master/src/sample_pudge)

[Phantom Assassin](https://github.com/cttimm/d2nn/blob/master/src/sample_pa)

[Alchemist](https://github.com/cttimm/d2nn/blob/master/src/sample_alch)

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


