# Gathers match samples from Valve's web API
# Charles Timmerman - Term Project - CSCI 3450
# cttimm4427@ung.edu

import dota2api
import json
        
def samples(max = 10000):
    """ Returns dictionary of match data up to max, default 1000 """
    list_results = []
    with open("data/key.json", "r") as file: # Requires web api key to be saved in a file named "key" with quotes around the key 
        key = json.load(file)
    i = 0
    while(len(list_results) != max):
        try:
            results = dota2api.Initialise(key).get_match_history_by_seq_num(start_at_match_seq_num=(2416543502-(100*i)))
            for j in range(len(results["matches"])):
                if(results["matches"][j]["duration"] > 900 and results["matches"][j]["duration"] < 3600 and results["matches"][j]["game_mode"] == 22):
                    if(len(list_results) == max):
                        print("Match threshold acquired, saving file...")
                        break
                    else:
                        list_results.append(results["matches"][j])
            i += 1
            print("Analyzed %d matches; using %d." % (i*100, len(list_results)))
        except:
            pass

    with open("data/matchdata.json", "w") as file:
        json.dump(list_results, file)
    file.close()

def parse():
    """ Parses matches to the correct format for the bpnn, outputs a list """
    """ Requires samples to have already been grabbed """
    with open("data/matchdata.json", "r") as file:
        workingset = json.load(file)
    file.close()

    if (len(workingset) == 10000):
        save_list = []
        for i in range(114):
            save_list.append([])
        for i in range(1000):
            for j in range(10):
                current = workingset[i]["players"][j]
                heroid = current["hero_id"]
                detailslist = [
                    current.get("gold_per_min")/1000,
                    current.get("xp_per_min")/1000,
                    current.get("kills")/100,
                    current.get("deaths")/100,
                    current.get("assists")/100,
                    current.get("hero_damage")/100000,
                    current.get("hero_healing")/10000
                ]
                if (current["player_slot"] < 5) == workingset[i]["radiant_win"]:
                    save_list[heroid].append([detailslist,[1]])
                else:
                    save_list[heroid].append([detailslist,[0]])
        with open("data/sampledata.json", "w") as file:
            json.dump(save_list, file)
    else:
        print("Error with matchdata, incorrect length")

def heroes():
    """ parses ..ref/hereoes.json and indexes heroes by heroid -1 """
    
    heroes = [""] * 115
    with open("../ref/heroes.json","r") as file:
        heroes_raw = json.load(file)["heroes"]
    file.close()
    for hero in heroes_raw:
        heroes[hero["id"]-1] = hero["localized_name"].strip().lower().replace(" ","").replace("-","")
    with open("data/heroes.json","w") as file:
        json.dump(heroes, file)
    file.close()

if __name__ == "__main__":
    
    parse()