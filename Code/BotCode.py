from random import randint
import numpy as np
import network2
import itertools
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


def get_classified_hand(test_inputs):
    net = network2.load("../TrainedModel/network.txt")
    return net.feedforward(test_inputs)

# Start Point

noOFPlayers = 2
#noOFPlayers = raw_input("Enter no. of players : ")
noOFPlayers = int(noOFPlayers)

players = []
for x in range(noOFPlayers):
    players.append(deal_hands())



def betting(current_bet, current_hand, risk_factor=1):
    """
    :param current_bet:
    :param current_hand:
    :return: 0-fold, 1-call, 2-raise
    """
    max_bets = [i * 100 * risk_factor for i in range(1, 10)]
    if current_bet > max_bets[current_hand]:
        return 0
    else:
        if 1.5*current_bet > max_bets[current_hand]:
            return 1
        else:
            return 2

def get_same_rank_cards(cards):
    card_groups = []
    flag = 0
    for card in cards:
        if len(card_groups) == 0:
            card_groups.append([card])
            continue
        for card_group in card_groups:
            if get_card_info(card_group[0])[1] == get_card_info(card)[1]:
                card_group.append(card)
                flag = 1
                continue
        if flag == 0:
            card_groups.append([card])
    return card_groups

def get_suite_with_max_cards(cards, is_royal=0):
    suite = np.zeros((4))
    for card in cards:
        if get_card_info(card)[1] in {1, 10, 11, 12, 13} or not is_royal:
            suite[get_card_info(card)[0] - 1] += 1
    suite = np.argmax(suite) + 1
    discard = []
    for card in cards:
        if get_card_info(card)[0] != suite:
            discard.append(card)
    return discard


def classify_bot_hand(bot_hand):
    card = ()
    inp_arr = []
    for x in bot_hand:
        card = get_card_info(x)
        inp_arr.append(card[0])
        inp_arr.append(card[1])

    test_inputs = np.reshape(findX(inp_arr), (85, 1))
    classified_hand = get_classified_hand(test_inputs)
    classified_hand = [activation for a in classified_hand for activation in a]
    # print classified_hand
    return np.argmax(classified_hand)

def discard_cards(cards, classified_hand):
    if classified_hand == 2 or classified_hand==3 or classified_hand == 6:
        groups= get_same_rank_cards(cards)
        group = list(itertools.chain(*groups))
        return list(set(cards) - set(group))
    else:
        if classified_hand == 5 or classified_hand == 9:
            return get_suite_with_max_cards(cards, is_royal=(classified_hand == 9))
        else:
            if classified_hand == 7:
                groups = get_same_rank_cards(cards)
                groups = sorted(groups, key=len, reverse=True)
                return list(set(cards) - set(groups[0]))
            else:
                if classified_hand == 8:
                    discard = discard_cards(cards, 5)
                    if len(discard) == 0:
                        discard = discard_cards(cards, 4)
                    return discard
                else:
                    if classified_hand == 4: #TODO Case 0, 1, 4 and else conditions
                        pass

# classified_hand = classify_bot_hand(players[0])
print get_suite_with_max_cards([1, 14, 2, 23, 25], 1)