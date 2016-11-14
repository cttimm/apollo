from bpnn import NN
import json

class d2nn(NN):
    def __init__(self):
        with open("data/sampledata.json","r") as file:
            self.data = json.load(file)
        file.close()
        NN.__init__(self)
        self.heroid = 1

    def load(self):
        """ Loads the existing dataset from a json file, will be indexed with results for individual heros """
        patterns = self.data[self.heroid][:200]
        print("n = %d" % len(patterns))
        NN.train(self, patterns)

    def scalers(self):
        """ Prints combined weights for inputs """
        inputs = self.weights()[0]
        hidden = self.weights()[1]
        output = self.weights()[2]
        scaled = [0.0]*8
    
        for i in range(len(inputs)-1):
            sum_hidden = 0.0
            sum_output = 0.0
            for j in range(len(hidden)):
                for k in range(len(output)):
                    sum_hidden = sum_hidden + hidden[j][k]
                    sum_output = sum_output + output[k][0]
                scaled[i] = inputs[i][j] * (sum_hidden + sum_output)
        print(scaled)

    def predict(self, values):
        results = self.test(values)
        if(results > .5):
            print("Victory, %.3f" % results)
        else:
            print("Defeat, %.3f" % results)



if __name__ == "__main__":
    test = d2nn()
    test.heroid = 74
    test.load()
    test.scalers()