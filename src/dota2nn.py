from bpnn import NN
import json

class dota2nn(NN):
    def __init__(self):
        with open("sampledata.json","r") as file:
            self.data = json.load(file)
            NN.__init__(self)
        file.close()

    def loadHero(self, heroid):
        """ Loads the existing dataset from a json file, will be indexed with results for individual heros """
        print("Using a training set of %d samples" % len(self.data[heroid]))
        patterns = self.data[heroid]
        NN.train(self, patterns)

    def getPredict(self, values):
        """ Takes input list and returns float indicating valuation of win """
        return NN.test(self, [[values]])

    def getWeights(self):
        """ Prints weights for inputs """
        weights = NN.weights(self)
        return weights

if __name__ == "__main__":
    apollo = dota2nn()
    apollo.loadHero(74)
    print(apollo.getPredict([.1,.1,.0,.1,.0,.2,.0,.02]))
    print(apollo.getWeights())