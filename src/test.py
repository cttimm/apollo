import apollo

apollo = apollo.initialize()

# Proof of concept

# "Teach" the bpnn some examples
# [gpm, xpm, kills],[1 = win, 0 = loss]
pattern = [
    [[.7,.700,.2],[1]],
    [[.300,.300,.01],[0]],
    [[.540,.540,.05],[1]],
    [[.600,.560,.5],[1]]
]

apollo.train(pattern)

#.96 certainty of win
apollo.test([[[.500,.500,.11]]])

#.08 certainty of win
apollo.test([[[.300,.300,.05]]])

# Get weights
apollo.weights()