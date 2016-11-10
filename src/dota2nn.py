from bpnn import NN
import json

class dota2nn(NN):
    def loadHero(self, heroid):
        """ Loads the existing dataset from a json file, will be indexed with results for individual heros """
        with open("samples.json", "r") as file:
            NN.train(self, json.load(file))
        file.close()

    def getPredict(self, values):
        """ Takes input list and returns float indicating valuation of win """
        return NN.test(self, [[values]])

    def getWeights(self):
        """ Prints weights for inputs """
        weights = NN.weights()
        print(weights)

if __name__ == "__main__":
    apollo = apollo()
    apollo.loadHero(1)
    test_values = [.4,.4,.01,.05,.1,.2]
    print("Test values:\n[gpm, xpm, k, d, a, damage]:\n" + str(test_values))
    print("Predicted win: %.2f" % apollo.getPredict(test_values))
    test_values = [.5,.5,.01,.05,.1,.2]
    print("Test values:\n[gpm, xpm, k, d, a, damage]:\n" + str(test_values))
    print("Predicted win: %.2f" % apollo.getPredict(test_values))
    test_values = [.5,.5,.1,.05,.15,.2]
    print("Test values:\n[gpm, xpm, k, d, a, damage]:\n" + str(test_values))
    print("Predicted win: %.2f" % apollo.getPredict(test_values))