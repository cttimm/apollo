import dota2api
import json

key = "44D2B9F7C72B1931CC601FF4086C9014"

def getMatches(heroID, n = 1000):
    """ Get Match Details of the correct format, i. e.
        [[stat, stat, stat, stat, stat],[stat, stat, stat, stat, stat], ...]
    """
    module = dota2api.Initialise(key)

    match_dict = json.loads(module.get_match_history(hero_id = 74, game_mode = 22).json)

    match_list = list(match_dict.keys())

    print(match_list)