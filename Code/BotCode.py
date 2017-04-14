from random import randint, shuffle
import numpy as np
import network2
import itertools
# Basic Info
# cards array
# suit 1-club , 2-spade , 3-heart, 4-diamond
# Ranks 1-13

# Utility Functions
def get_card_info(card):
    """
    #1-13 --->club
    #14-26-->spade
    #27-39-->heart
    #40-52-->diamond
    """
    result = ((card-1)/13 + 1,  card - ((card-1)/13)*13)
    return result

def create_card(suite, rank):
    return (suite-1)*13 + rank

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

def get_same_rank_cards(cards):
    card_groups = []
    flag = 0
    for card in cards:
        flag = 0
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
    relevant_groups = []
    for group in card_groups:
        if len(group) >= 2:
            relevant_groups.append(group)
    return relevant_groups

def get_suite_with_max_cards(cards, is_royal = 0):
    suite = np.zeros((4))
    if is_royal:
        for card in cards:
            if get_card_info(card)[1] in {1, 10, 11, 12, 13}:
                suite[get_card_info(card)[0] - 1] += 1
    else :
        for card in cards:
            suite[get_card_info(card)[0] - 1] += 1
    return np.argmax(suite)+1

def other_suit_cards(suite,cards, is_royal=0):
    discard = []
    if is_royal:
        for card in cards:
            if get_card_info(card)[0] != suite or get_card_info(card)[1] not in {1,10,11,12,13}:
                discard.append(card)
    else:
        for card in cards:
            if get_card_info(card)[0] != suite:
                discard.append(card)
    return discard

def display_hand(player_index):
    global players_names, players_cards
    print "\nCards for player",players_names[player_index]," : "
    for card in players_cards[player_index]:
        print "S-",get_card_info(card)[0]," R-",get_card_info(card)[1]

def print_current_game_status():
    global current_pot, max_bet, num_players, players_names, player_chips
    print "Current Pot: ", current_pot
    print "Maximum Bet: ", max_bet


# Major Functions
def process_response_round(player_index, player_response):
    global current_pot, max_bet ,players_names, players_cards, player_chips, game_end , num_players
    player_response = str(player_response)
    if player_response == 'F':
        print "\n",players_names[player_index], "has folded."
        del players_cards[player_index]
        del player_chips[player_index]
        del players_names[player_index]
        num_players -= 1

        if num_players == 1:
            print "\n",players_names[0], " has won Rs.", current_pot, " !"
            game_end = 1
            return players_names[0]

    else:
        if player_response == 'C':
            current_pot += max_bet
            player_chips[player_index] -= max_bet
            print "\n",players_names[player_index], "has called."
            print_current_game_status()
        else:
            if player_response == 'R':
                current_pot += (max_bet*1.5)
                player_chips[player_index] -= (1.5*max_bet)
                max_bet *= 1.5
                print "\n",players_names[player_index], " has raised to ", max_bet
                print_current_game_status()
    return -1

def showdown():
    global players_names, current_pot, players_cards, num_players
    hands = []
    for x in range(num_players):
        display_hand(x)
        print "Current Hand is Identified as:", identify_current_hand(players_cards[x])
        hands.append(identify_current_hand(players_cards[x]))

    winners = [players_names[a] for a in range(len(hands)) if hands[a] == max(hands)]
    for winner in winners:
        print winner, " has won Rs.", current_pot/len(winners), " !"
    return winners

def process_winner():
    global players_names, current_pot
    if len(players_names) > 1:
        return showdown()
    else:
        if len(players_names) == 1:
            display_hand(0)
            print players_names[0], " has won Rs.", current_pot, " !"
            return players_names[0]
    return -1

def replace_cards(player_index, cards_to_discard):
    global players_names, players_cards
    players_cards[player_index] = list(set(players_cards[player_index]) - set(cards_to_discard))
    new_cards = deal_hands(len(cards_to_discard))
    players_cards[player_index].extend(new_cards)
    print players_names[player_index], " removed ", len(cards_to_discard), " cards."

def deal_hands(num_cards):
    global cards
    hands = []
    while len(hands)!=num_cards:
        num = randint(1,52)
        if cards[num-1] != -1:
            hands.append(num)
            cards[num-1] = -1
    return hands

def betting(current_bet, current_hand, risk_factor=1):
    """
    :param current_bet:
    :param current_hand:
    :return: F-fold, C-call, R-raise
    """
    max_possible_bets = [i * 500 * risk_factor for i in range(1, 10)]
    if current_bet > max_possible_bets[current_hand]:
        return "F"
    else:
        if 1.5*current_bet > max_possible_bets[current_hand]:
            return "C"
        else:
            return "R"

def classify_bot_hand(bot_hand):
    card = ()
    inp_arr = []
    for x in bot_hand:
        card = get_card_info(x)
        inp_arr.append(card[0])
        inp_arr.append(card[1])

    test_inputs = np.reshape(findX(inp_arr), (85, 1))
    net = network2.load("../TrainedModel/network.txt")
    classified_hand = net.feedforward(test_inputs)
    classified_hand = [activation for a in classified_hand for activation in a]
    # print classified_hand
    return max(identify_current_hand(bot_hand), np.argmax(classified_hand))

def discard_cards(cards, classified_hand):
    if classified_hand == 2 or classified_hand==3 or classified_hand == 6 or classified_hand == 0 or classified_hand == 1:
        groups= get_same_rank_cards(cards)
        group = list(itertools.chain(*groups))
        return list(set(cards) - set(group))
    else:
        if classified_hand == 5 or classified_hand == 9:
            return other_suit_cards(get_suite_with_max_cards(cards, is_royal=(classified_hand == 9)),cards=cards, is_royal=(classified_hand==9))
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
                    if classified_hand == 4:

                        discard = []

                        #Keep one copy from repeated in pairs
                        same_rank_cards = get_same_rank_cards(cards)
                        for rank_cards in same_rank_cards:
                            del rank_cards[0]
                        #Remove same Cards from the main card set after grouping
                        group = list(itertools.chain(*same_rank_cards))
                        cards = list(set(cards)-set(group))

                        discard.extend(group)

                        # diff = np.zeros(5)
                        temp_cards = cards
                        # print sorted(cards, cmp=highRank)
                        cards = sorted(cards, cmp=highRank)
                        max_diff = 2

                        while max_diff > 1 and len(cards) != 1:
                            diff = np.zeros(len(cards)-1)
                            for i in range(len(cards)-1):
                                diff[i] = get_card_info(cards[i+1])[1] - get_card_info(cards[i])[1]
                            # diff[len(cards)-1] = get_card_info(cards[-1])[1] - get_card_info(cards[0])[1]
                            max_diff = np.argmax(diff)
                            if diff[max_diff] > 1:
                                discard.append(cards[max_diff+1])
                                del cards[max_diff+1]
                            #changing Max Diff to monitor the maxDiffference at that index
                            max_diff = diff[max_diff]
                        cards = temp_cards
                        return discard

def highRank(a, b):
    if get_card_info(a)[1] > get_card_info(b)[1]:
        return 1
    return -1
    # return get_card_info(a)[1] > get_card_info(b)[1]

def identify_current_hand(cards):
    groups = get_same_rank_cards(cards)
    groups = sorted(groups, key=len, reverse=True)
    possible_hand = 0
    if len(groups) == 0:
        possible_hand = 0
    else:
        if len(groups) == 1:
            if len(groups[0]) == 2: possible_hand = max(possible_hand,1)
            else :
                if len(groups[0]) == 3: possible_hand = max(possible_hand,3)
                else :
                    if len(groups[0]) == 4: possible_hand = max(possible_hand,7)
        else:
            if len(groups) == 2:
               if len(groups[0]) == 3:possible_hand = max(possible_hand,6)
               else: possible_hand = max(possible_hand,2)

    if len(other_suit_cards(get_suite_with_max_cards(cards, is_royal=1),cards, is_royal=1)) == 0:
        possible_hand = max(possible_hand,9)
    flag = 0
    if len(other_suit_cards(get_suite_with_max_cards(cards),cards)) == 0:
            card_ranks = []
            for card in cards :
                card_ranks.append(get_card_info(card)[1])
            card_ranks.sort()
            for i in range(4):
                if card_ranks[i+1] - card_ranks[i] != 1 :
                    possible_hand = max(possible_hand,5)
                    flag = 1
            if flag == 0:
                possible_hand = max(possible_hand, 8)

    flag = 0
    card_ranks = []
    for card in cards:
        card_ranks.append(get_card_info(card)[1])
    card_ranks.sort()
    for i in range(4):
        if card_ranks[i + 1] - card_ranks[i] != 1:
            flag = 1
            break
    if flag == 0 : possible_hand = max(possible_hand, 4)

    return possible_hand

#Global GAME VARIABLES
cards = [x+1 for x in range(52)]
num_players = 2
#player_status = [] # active/folded
players_names = []
players_cards = []
player_chips = []
max_bet = 1
current_pot = 0
game_end = 0
winner = -1


def start_game(): #TODO initialize global variables, Check if winner is set then return winner
    # Start Point
    global winner, cards, num_players, player, players_names, players_cards, player_chips, max_bet, current_pot, game_end

    # Initialize global variables
    winner = -1
    cards = [c for c in range(1, 53)]
    max_bet = 1
    current_pot = 0
    game_end = 0
    del players_cards[:]
    del player_chips[:]
    del players_names[:]
    del player_status[:]
    num_players = 2

    print "WELCOME TO 5 CARD DRAW POKER!!!\n"
    print "Cards are represented as :\nSuit:\n1-club \n2-spade \n3-heart\n4-diamond\nRank: 1-13\n"

    num_players = raw_input("Enter no. of players : ")
    num_players = int(num_players)

    for x in range(num_players-1):
        name = raw_input("Enter player name : ")
        players_names.append(name)
        players_cards.append(deal_hands(5))
        player_chips.append(20000)
        display_hand(x)
        print "\n"

    players_names.append("PokeUs(Bot)")
    players_cards.append(deal_hands(5))
    player_chips.append(20000)

    # Initial pot
    initialBet = 100
    # print players_names[0]," posts small blind of Rs ",str(initialBet)
    # current_pot += initialBet
    # player_chips[0] -= initialBet
    # max_bet = max(max_bet, initialBet)

    print "The Game starts with an ante of Rs.",str(initialBet)

    for i in range(0, num_players):
        current_pot += (initialBet)
        player_chips[i] -= initialBet
        max_bet = max(max_bet, initialBet)
    print "\n"
    for x in range(num_players):
        print "Player: ", players_names[x]
        print "Chips remaining: ", player_chips[x], "\n"

    # Round 1 Betting
    for x in range(num_players - 1):
        print players_names[x],". It's your turn.\n"
        response = 'A'
        while response not in ['C','F','R']:
            response = raw_input("Respond with (C)all/(R)aise/(F)old: ")
        winner = process_response_round(x,response)
        if winner != -1:
            break
    if winner == -1:
        bot_response = betting(max_bet, identify_current_hand(players_cards[-1]))
        winner = process_response_round(-1,bot_response)



    # Round 2 Discard Card
    if game_end == 0:
        for x in range(num_players - 1):
            print "\n",players_names[x], ": Enter no. of cards to discard.(0-5): "
            num_cards_discard = raw_input()
            num_cards_discard = int(num_cards_discard)
            cards_to_discard = []
            for i in range(num_cards_discard):
                suit = int(raw_input("Suit: "))
                rank = int(raw_input("Rank: "))
                cards_to_discard.append(create_card(suit, rank))
            replace_cards(x,cards_to_discard)
            display_hand(x)
        # Card Discarded By Bot
        replace_cards(-1, discard_cards(players_cards[-1], classify_bot_hand(players_cards[-1])))



    #Round 3: Betting
    if game_end == 0:
        for x in range(num_players - 1):
            print players_names[x],". It's your turn.\n"
            response = "A"
            while response not in ["C","F","R"]:
                response = raw_input("Respond with (C)all/(R)aise/(F)old: ")
            winner = process_response_round(x,response)
            if winner != -1:
                break
        if winner == -1:
            bot_response = betting(max_bet, identify_current_hand(players_cards[-1]))
            winner = process_response_round(-1,bot_response)
        if winner == -1:
            print "Showdown"
            process_winner()


def random_bot_game():
    """
        Player 0 is Random(Bot)
        Player 1 is PokeUs(Bot)
    """
    global winner, cards, num_players, player, players_names, players_cards, player_chips, max_bet, current_pot, game_end

    # Initialize global variables
    winner = -1
    cards = [c for c in range(1, 53)]
    max_bet = 1
    current_pot = 0
    game_end = 0
    del players_cards[:]
    del player_chips[:]
    del players_names[:]
    num_players = 2

    players_names.append("Random(Bot)")
    players_cards.append(deal_hands(5))
    player_chips.append(20000)
    players_names.append("PokeUs(Bot)")
    players_cards.append(deal_hands(5))
    player_chips.append(20000)

    #Initial Bet
    initialBet = 100

    current_pot += initialBet
    player_chips[0] -= initialBet
    max_bet = max(max_bet, initialBet)

    current_pot += initialBet
    player_chips[1] -= initialBet
    max_bet = max(max_bet,  initialBet)


    #Round 1: Betting
    possible_valid_responses = ["F", "C", "R"]
    random_bot_response = possible_valid_responses[randint(0,2)]
    winner = process_response_round(0, random_bot_response)

    if winner == -1:
        bot_response = betting(max_bet, identify_current_hand(players_cards[-1]))
        winner = process_response_round(1, bot_response)

    #Round 2: Discard Cards
    if game_end == 0 :
        num_discard = randint(0,5)
        shuffle(players_cards[0])
        replace_cards(0, [players_cards[0][c] for c in range(num_discard)])

        replace_cards(-1, discard_cards(players_cards[-1], classify_bot_hand(players_cards[-1])))


    #Round 3: Betting
    if game_end == 0 and winner == -1:
        possible_valid_responses = ["F", "C", "R"]
        random_bot_response = possible_valid_responses[randint(0, 2)]
        winner = process_response_round(0, random_bot_response)

        if winner == -1:
            bot_response = betting(max_bet, identify_current_hand(players_cards[-1]))
            winner =process_response_round(1,bot_response)

            if winner == -1:
                print "Showdown"
                winner = process_winner()

    return winner


win = []
win.append(0)
win.append(0)
money = []
money.append(0)
money.append(0)
for g in range(0, 1000):
    print "GAME: ", g+1
    money[0] -= 20000
    money[1] -= 20000
    win_value = random_bot_game()
    money[0]+=player_chips[0]
    money[1]+=player_chips[1] #FIXME don't delete player when fold
    if win_value == "PokeUs(Bot)":
        win[1] += 1
        money[1]+=current_pot
    else:
        if len(win_value) == 2:
            win[0]+=0.5
            win[1]+=0.5
            money[0]+=(current_pot/2)
            money[1]+=(current_pot/2)
        else:
            win[0]+=1
            money[0] += current_pot

print "Random(Bot) won: ", win[0], " / ", win[0]+win[1], " : ", (float(win[0])*100)/(win[0]+win[1]), "% games"
print "Random(Bot) won: Rs.", money[0]
print "PokeUs(Bot) won: ", win[1], " / ", win[0]+win[1], " : ", (float(win[1])*100)/(win[0]+win[1]), "% games"
print "PokeUs(Bot) won: Rs.", money[1]
