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
            print("[WARNING] data/sampledata.json not found, see gather.py")
        # Open hero data, indexed by heroid -1, see gather.py
        try:
            with open("data/heroes.json", "r") as file:
                self.heroes = json.load(file)
            file.close()
        except FileNotFoundError:
            print("[WARNING] data/heroes.json not found, see gather.py")

        # Inheritance
        NN.__init__(self)
        self.heroid = 1

    def load(self, N=.0025, M=.0025, iterations=5000):
        ''' Trains the neural network with match statistics for the current heroid '''
        patterns = self.data[self.heroid][:200]
        print("n = %d" % len(patterns))
        NN.train(self, patterns, iterations, N, M)

    def predict(self, values):
        ''' Returns prediction from the neural network; used to simplify the method call, removing the [[]] '''
        results = self.test([[values]])
        return results

    def set_hero(self, hero_name):
        ''' Sets the current heroid by hero name input '''
        heroname = hero_name.strip().lower().replace(" ","").replace("-","")
        try:
            self.heroid = self.heroes.index(heroname) + 1
            print("Updated heroid to %d" % self.heroid)
        except ValueError:
            print("Hero not found")

    def relative_weights(self):
        ''' Returns a list of weight estimates '''
        avg = [.4,.4,.05,.02,.1,.1,.01,.01]
        results = []
        for i in range(len(avg)):
            
            # Needs to be cleaned up
            results.append([0.0]*2)
            diff = avg[:]
            diff[i] -= (avg[i] * .5)
            results[i][0] = (self.predict(diff) - self.predict(avg))/(avg[i] * .5)
            diff = avg[:]
            diff[i] += (avg[i] * .5)
            results[i][1] = (self.predict(diff) - self.predict(avg))/(avg[i] * .5)
        return results


    def averages(self):
        ''' Returns the average values for all data '''
        for i in range(len(self.data)):
            pass

    def lastmatch(self, playerid):
        ''' Returns the prediction of a players last match '''
        pass

            

if __name__ == "__main__":
    d2nn = Initialize()
    d2nn.set_hero("invoker")
    d2nn.averages()
    print(d2nn.relative_weights())
