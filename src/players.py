# Profides function for saving player ids, giving them an alias, and fetching statistics from recent matches
# Charles Timmerman - Term Project CSCI 3450
# cttimm4427@ung.edu

import json
import dota2api

class start():
    def __init__(self):
        self.players = {}
        self.key = None

        # Attempt to load existing player list from file
        try:
            with open("data/players.json","r") as file:
                self.players = json.load(file)
            with open("data/key.json","r") as file:
                self.key = json.load(file)

        except (FileNotFoundError, json.decoder.JSONDecodeError) as e:
            if e.filename == "data/players.json":
                with open("data/players.json","w") as file:
                    json.dump(self.players, file)
            else:
                print(e.strerror)
        file.close()

    def last(self, player):
        ''' Returns a list of the stats from the players most recent match [[stats], [result], [heroid]]'''
        pass

    def set(self, playerid, alias):
        ''' Sets the alias (name) for the provided playerid; also used to add players '''
        self.players[int(playerid)] = alias

    def delete(self, playerid):
        ''' Removes player from the dict with the current playerid '''
        try:
            self.players.__delitem__(str(playerid))
        except KeyError as e:
            print("Player not found.")

    def show(self):
        ''' Lists current alias '''
        print(self.players)
    
    def save(self):
        ''' Save changes to alias list '''
        with open("data/players.json","w") as file:
            json.dump(self.players, file)
        