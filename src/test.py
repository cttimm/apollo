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

#.08 certainty of win
apollo.test([[[.3,.3,.00]]])

apollo.test([[[.31,.31,.00]]])

apollo.test([[[.315 ,.315,.00]]])

apollo.test([[[.315 ,.315,.02]]])

apollo.test([[[.315 ,.315,.04]]])

apollo.test([[[.32,.32,.00]]])

apollo.test([[[.33 ,.33,.00]]])

apollo.test([[[.34 ,.34,.00]]])
# Get weights
apollo.weights()