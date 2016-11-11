
import dota2api
import json


        
def gatherSamples():
    list_results = []
    key = "44D2B9F7C72B1931CC601FF4086C9014"
    i = 0
    while(len(list_results) != 1000):
        try:
            results = dota2api.Initialise(key).get_match_history_by_seq_num(start_at_match_seq_num=(2416540502-(100*i)))
            for j in range(len(results["matches"])):
                if(results["matches"][j]["duration"] > 900 and results["matches"][j]["game_mode"] == 22):
                    if(len(list_results) == 1000):
                        print("Match threshold acquired, saving file...")
                        break
                    else:
                        list_results.append(results["matches"][j])
            i += 1
            print("Analyzed %d matches; using %d." % (i*100, len(list_results)))
        except:
            pass

    with open("matchdata.json", "w") as file:
        json.dump(list_results, file)
    file.close()

def parseMatches():
    with open("matchdata.json", "r") as file:
        workingset = json.load(file)
    file.close()

    if (len(workingset) == 1000):
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
                    current.get("hero_healing")/100000,
                    current.get("tower_damage")/100000
                ]
                if (current["player_slot"] < 5) == workingset[i]["radiant_win"]:
                    save_list[heroid].append([detailslist,[1]])
                else:
                    save_list[heroid].append([detailslist,[0]])
        with open("sampledata.json", "w") as file:
            json.dump(save_list, file)
    else:
        print("Error with matchdata, incorrect length")

if __name__ == "__main__":
    parseMatches()