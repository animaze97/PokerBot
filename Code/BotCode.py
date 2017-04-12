from random import randint
import numpy as np
import main

# cards array
# suit 1-club , 2-spade , 3-heart, 4-diamond
# Ranks 1-13
cards = [x+1 for x in range(52)]

# utility functions

def get_card_info(card):
    """
    #1-13 --->club
    #14-26-->spade
    #27-39-->heart
    #40-52-->diamond
    """
    result = ((card-1)/13 + 1,  card - ((card-1)/13)*13)
    return result

def deal_hands():
    hands = []
    while len(hands)!=5:
        num = randint(1,52)
        if cards[num-1] != -1:
            hands.append(num)
            cards[num-1] = -1

    return hands

def vectorized_result_x(suit, rank):
    e = np.zeros(17)
    e[suit - 1] = 1
    e[rank + 3] = 1
    return e


def findX(x):
    res= []
    for i in range(0, 5):
        res.extend(vectorized_result_x(x[2 * i], x[2 * i + 1]))
    return res

# Start Point

noOFPlayers = 2
#noOFPlayers = raw_input("Enter no. of players : ")
noOFPlayers = int(noOFPlayers)

players = []
for x in range(noOFPlayers):
    players.append(deal_hands())


def classify_bot_hand(bot_hand):
    card = ()
    inp_arr = []
    for x in bot_hand:
        card = get_card_info(x)
        inp_arr.append(card[0])
        inp_arr.append(card[1])

    test_inputs = np.reshape(findX(inp_arr), (85, 1))
    classified_hand = main.get_classified_hand(test_inputs)
    print classified_hand

classify_bot_hand(players[0])