# Interface for two layer bpnn that adds functionality to pull information from Valve's WebAPI on match data
# Charles Timmerman - Term Project CSCI 3450 -
# cttimm4427@ung.edu

from bpnn import NN
import players
import json


class start(NN):
    def __init__(self):
        NN.__init__(self)
        self.heroid = 1

        # Open sample data, see gather.py
        try:
            with open("data/sampledata.json","r") as file:
                self.data = json.load(file)
            with open("data/heroes.json", "r") as file:
                self.heroes = json.load(file)
            file.close()
        except FileNotFoundError as e:
            print("[WARNING] %s not found." % e.filename)


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

    def relative_weights(self, h = .25):
        # Iterate through stats in averages
        #  (f(x+h) - f(x)) / h 
        base = self.averages()
        temp = [0.0] * 8
        results = [0.0] * 8
        for i in range(len(base)):
            temp = base[:]
            temp[i] = base[i] + (base[i] * h)   
            results[i] = (self.predict(temp) - self.predict(base)) / (base[i] * h)
        print(results)
        

    def averages(self):
        ''' Returns the average values for all data '''
        avg = [0.0] * 8
        n = 0
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                n = n + 1
                for k in range(len(self.data[i][j][0])):
                    avg[k] = avg[k] + self.data[i][j][0][k]
            
        for i in range(len(avg)):
            avg[i] = avg[i] / n
        return avg
        

    def lastmatch(self, playerid):
        ''' Returns the prediction of a players last match '''
        pass

            

if __name__ == "__main__":
    
    d2nn = start()
    d2nn.set_hero('pudge')
    d2nn.load()
    d2nn.relative_weights(h = .5)
