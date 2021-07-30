"""
The reward function an agent optimizes to win at Hearts.
"""

import numpy as np

from hearts_gym.utils.typing import Reward
from .hearts_env import HeartsEnv


class RewardFunction:
    """
    The reward function an agent optimizes to win at Hearts.

    Calling this returns the reward.
    """

    def __init__(self, env: HeartsEnv):
        self.env = env
        self.game = env.game

    def __call__(self, *args, **kwargs) -> Reward:
        return self.compute_reward(*args, **kwargs)

    def compute_reward(
        self, player_index: int, prev_active_player_index: int, trick_is_over: bool,
    ) -> Reward:
        """Return the reward for the player with the given index.

        It is important to keep in mind that most of the time, the
        arguments are unrelated to the player getting their reward. This
        is because agents receive their reward only when it is their
        next turn, not right after their turn. Due to this peculiarity,
        it is encouraged to use `self.game.prev_played_cards`,
        `self.game.prev_was_illegals`, and others.

        Args:
            player_index (int): Index of the player to return the reward
                for. This is most of the time _not_ the player that took
                the action (which is given by `prev_active_player_index`).
            prev_active_player_index (int): Index of the previously
                active player that took the action. In other words, the
                active player index before the action was taken.
            trick_is_over (bool): Whether the action ended the trick.

        Returns:
            Reward: Reward for the player with the given index.
        """

        # index = self.env.get_offset_indices(
        #     np.array([player_index]),
        #     self.game.leading_player_index,
        #     self.game.num_players,
        # )[0]

        # print("Player_idx", player_index)
        # print("Player_rel_idx", index)
        # print("Curr_Played_card", self.game.table_cards[index])
        # if len(self.game.prev_table_cards) > 3:
        #     print("Prev_Played_card", self.game.prev_table_cards[player_index])
        # print("Prev. Hands", self.game.prev_hands[player_index])
        # print("Curr. Table", self.game.table_cards)
        # print("Prev. Table", self.game.prev_table_cards)
        # exit(1)
        # if len(self.game.hands[player_index]) < 10:
        #     exit(1)

        # Danimir's version
        # reward = 0
        # if self.game.prev_was_first_trick:
        #                 return self.game.prev_table_cards[player_index].rank / 11

        #             else:
        #                 hand = self.game.prev_hands[player_index]
        #                 suits = ['club', 'diamond', 'heart', 'spade']
        #                 handsuit = {'club' : [], 'diamond' : [], 'heart' : [], 'spade' : []}
        #                 for card in hand:
        #                     suit = card.suit
        #                     rank = card.rank
        #                     handsuit[suits[suit]].append(rank)

        #                 for suit in suits:
        #                     if len(handsuit[suit]) == 0:
        #                         reward +=0.1*len(hand)
        #                     else:
        #                         P = values_me = np.asarray(handsuit[suit]) + 0.00001
        #                         Q = values_other = np.linspace(0,12,len(values_me)) + 0.00001
        #                         divergence = np.sum(P*np.log(P/Q))
        #                         reward -=divergence

        #                 if self.game.prev_trick_winner_index == player_index:
        #                     assert self.game.prev_trick_penalty is not None
        #                     return -self.game.prev_trick_penalty
        #                 else:
        #                     reward += 1

        #                 return reward

        # Illegal action
        if self.game.prev_was_illegals[player_index]:
            # reward = -self.game.max_penalty * self.game.max_num_cards_on_hand
            return -self.game.max_penalty / self.game.max_penalty

        card = self.game.prev_played_cards[player_index]

        # The agent did not take a turn until now; no information to provide.
        if card is None:
            return 0

        # Shot to the moon
        if trick_is_over and self.game.has_shot_the_moon(player_index):
            # reward = self.game.max_penalty * self.game.max_num_cards_on_hand
            return self.game.max_penalty / self.game.max_penalty

        # ################ Prev Current table ###################
        # if len(self.game.prev_table_cards) > player_index:
        #     played_card = self.game.prev_table_cards[player_index]
        #     # If leading suit is heart
        #     if self.game.prev_leading_suit == 2:
        #         table_ranks = [
        #             card.rank for card in self.game.prev_table_cards if card.suit == 2
        #         ]
        #         max_heart_rank = max(table_ranks)

        #         if played_card.rank > max_heart_rank:
        #             return (-played_card.rank / 11) / self.game.max_penalty
        #         else:
        #             return (played_card.rank / 11) / self.game.max_penalty

        #     # If leading suit is spade
        #     if self.game.prev_leading_suit == 3:
        #         table_ranks = [
        #             card.rank for card in self.game.prev_table_cards if card.suit == 3
        #         ]
        #         if 10 in table_ranks:
        #             if played_card.rank > 10:
        #                 return -self.game.max_penalty / self.game.max_penalty
        #             else:
        #                 return (played_card.rank / 11) / self.game.max_penalty

        ################ Previous table ##################
        # First trick: highest card
        if self.game.prev_was_first_trick:
            return (
                1 + self.game.prev_table_cards[player_index].rank / 11
            ) / self.game.max_penalty

        # If lead of trick: lowest card
        if len(self.game.prev_table_cards) > 0:
            if self.game.prev_table_cards[0] in self.game.prev_hands[player_index]:
                return (
                    1 + 1 - self.game.prev_table_cards[player_index].rank / 11
                ) / self.game.max_penalty
            # If not lead
            else:
                prev_hand_suits = [
                    card.suit for card in self.game.prev_hands[player_index]
                ]
                # If no suit: highest hearth card (pref: Spade Queen --> heart --> other)
                if self.game.leading_suit not in prev_hand_suits:
                    played_card = self.game.prev_table_cards[player_index]
                    # Spade Queen
                    if played_card.suit == 3 and played_card.rank == 10:
                        return self.game.max_penalty / self.game.max_penalty
                    # Hearth + highest card
                    else:
                        multiplier = 1
                        if played_card.suit == 2:
                            multiplier = 2
                        return (
                            1
                            + self.game.prev_table_cards[player_index].rank
                            / 11
                            * multiplier
                        ) / self.game.max_penalty

        # penalty = self.game.penalties[player_index]

        # if self.game.is_done():
        #     return -penalty

        if self.game.prev_trick_winner_index == player_index:
            assert self.game.prev_trick_penalty is not None
            return -self.game.prev_trick_penalty / self.game.max_penalty

        # Normalize between -1 and 1
        # reward = reward / self.game.max_penalty
        return 1 / self.game.max_penalty
        # return -penalty
