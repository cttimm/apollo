from bpnn import NN
import json

class Initialize(NN):
    def __init__(self):
        # Open sample data, see gather.py
        try:
            with open("data/sampledata.json","r") as file:
                self.data = json.load(file)
            file.close()
        except FileNotFoundError:
            print("data/sampledata.json not found, see gather.py")
        # Open hero data, indexed by heroid -1, see gather.py
        try:
            with open("data/heroes.json", "r") as file:
                self.heroes = json.load(file)
            file.close()
        except: FileNotFoundError:
            print("data/heroes.json not found, see gather.py")

        # Inheritance
        NN.__init__(self)
        
        # Default hero is anti mage
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
        """ X * (sum(hidden_layer) * sum(hidden_layer2)) """
        for i in range(len(inputs)-1):
            sum_hidden = 0.0
            sum_output = 0.0
            for j in range(len(hidden)):
                for k in range(len(output)):
                    sum_hidden = sum_hidden + hidden[j][k]
                    sum_output = sum_output + output[k][0]
                scaled[i] = inputs[i][j] * (sum_hidden + sum_output)
        return scaled

    def predict(self, values):
        """ Makes prediction based on match values """
        results = self.test(values)
        if(results > .5):
            print("Victory, %.3f" % results)
        else:
            print("Defeat, %.3f" % results)

    def pretty(self, values):
        """ Outputs values of game statistics """
        print("[GPM]:\t\t %.4f" % (values[0] * 10000))
        print("[XPM]:\t\t %.4f" % (values[1] * 10000))
        print("[Kills]:\t %.4f" % (values[2] * 100))
        print("[Deaths]:\t %.4f" % (values[3] * 100))
        print("[Assists]:\t %.4f" % (values[4] * 100))
        print("[Hero Damage]:\t %.4f" % (values[5] * 10000))
        print("[Healing]:\t %.4f" % (values[6] * 10000))
        print("[Tower Damage]:\t %.4f" % (values[7] * 10000))

    def set_hero(self, hero_name):
        """ Sets the current heroid by name """
        heroname = hero_name.strip().lower().replace(" ","").replace("-","")
        try:
            self.heroid = self.heroes.index(heroname) + 1
            print("Updated heroid to %d" % self.heroid)
        except ValueError:
            print("Hero not found")
        


if __name__ == "__main__":
    d2nn = Initialize()
    d2nn.set_hero("spirit breaker")