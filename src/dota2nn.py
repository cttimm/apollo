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
        inputs = self.weights()[0]
        hidden = self.weights()[1]
        output = self.weights()[2]
        scaled = [0.0]*8
    
        for k in range(len(inputs)-1):
            sum_hidden = 0.0
            sum_output = 0.0
            for i in range(len(hidden)):
                for j in range(len(output)):
                    sum_hidden = sum_hidden + hidden[i][j]
                    sum_output = sum_output + output[j][0]
                scaled[k] = -1 * inputs[k][i] * sum_hidden * sum_output
        print(scaled)

if __name__ == "__main__":
    test = dota2nn()
    test.loadHero(11)
    test.test([[[.3,.3,.1,.9,.5,.1,.1,.009]]])
    test.getScaledWeights()