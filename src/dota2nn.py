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
        print("n = %d" % len(patterns))
        NN.train(self, patterns)

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
                scaled[k] = inputs[k][i] * (sum_hidden + sum_output)
        print(scaled)

    def predict(self, values):
        results = self.test(values)
        if(results > .5):
            print("Victory, %.3f" % results)
        else:
            print("Defeat, %.3f" % results)


if __name__ == "__main__":
    test = dota2nn()
    print("Training network for hero Mirana")
    test.loadHero(9)
    print("Testing input matchid: 2771458676 [L]")
    test.predict([[[.342,.447,.06,.04,.13,.171,.0,.00033]]])
    print("Testing input matchid: 2769705846 [W]")
    test.predict([[[.3,.399,.01,.04,.2,.119,.0,.00638]]])
    print("Testing input matchid: 2754209826 [W]")
    test.predict([[[.552,.628,.09,.06,.13,.281,.0,.025]]])