import apollo
import json
apollo = apollo.initialize()

# Proof of concept
with open("results.json", "r") as file:
    pattern = json.load(file)
# "Teach" the bpnn some examples
# [gpm, xpm, kills],[1 = win, 0 = loss]

apollo.train(pattern)
print("Results")
#.96 certainty of win
apollo.test([[[.340 ,.340,.00]]])

#.08 certainty of win
apollo.test([[[.3,.300,.00]]])

# Get weights
apollo.weights()