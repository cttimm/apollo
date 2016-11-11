import dota2api
import json

key = "44D2B9F7C72B1931CC601FF4086C9014"
module = dota2api.Initialise(key)
def getMatches(heroID = 74, n = 100):
    """ Returns list of match ids consisting of a specific hero """

    # Get a list of matches from the history dictionary
    match_list = []
    match_dict = json.loads(module.get_match_history(hero_id = heroID, matches_requested = 100).json)
    for i in range(len(match_dict["matches"])):
        id = int(match_dict["matches"][i]["match_id"])
        if int(match_dict["matches"][i]["lobby_type"]) not in [0,7]:
            continue
        if id not in match_list:
            match_list.append(id)
    return match_list

def getDetails(matchID):
    
        # Open JSON, append match details dict
        match_dict = json.loads(module.get_match_details(matchID).json)
        details_list = []
        players = match_dict["players"]
        for n in range(len(players)):
            if players[n]["hero_id"] == 74:
                details_list = [
                    players[n].get("gold_per_min")/1000,
                    players[n].get("xp_per_min")/1000,
                    players[n].get("kills")/100,
                    players[n].get("deaths")/100,
                    players[n].get("assists")/100,
                    players[n].get("hero_damage")/10000
                ]
                if (n < 5) == match_dict["radiant_win"]:
                    win = 1
                else:
                    win = 0
        return [details_list, [win]]

if __name__ == "__main__":
    game_stats = []
    match_list = getMatches()
    for match in match_list:
       game_stats.append(getDetails(match))
    with open('samples.json', 'w+') as file:
        json.dump(game_stats, file)
    file.close()
        
