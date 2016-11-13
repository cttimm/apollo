from bpnn import NN
import json

class dota2nn(NN):
    def __init__(self):
        with open("data/sampledata.json","r") as file:
            self.data = json.load(file)
        file.close()
        
        NN.__init__(self)

    def loadHero(self, heroid):
        """ Loads the existing dataset from a json file, will be indexed with results for individual heros """

        patterns = self.data[heroid][:200]
        NN.train(self, patterns)

    def getPredict(self, values):
        """ Takes input list and returns float indicating valuation of win """
        return NN.test(self, [[values]])

    def getScaledWeights(self):
        """ Prints weights for inputs """
        input_weights = NN.weights(self)[0]
        output_weights = NN.weights(self)[1]
        scaled_weights = [0.0]*8
        for i in range(len(output_weights):
            for j in range(len(input_weights)):
                scaled_weights[j] += (output_weights[i] * input_weights[j][i])
        print(scaled_weights)
                

if __name__ == "__main__":
    test = dota2nn()
    test.loadHero(1)
    test.getScaledWeights()