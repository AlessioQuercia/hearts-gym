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
        # print("Player_idx", player_index)
        # if len(self.game.prev_table_cards) > 3:
        #     print("Played_card", self.game.prev_table_cards[player_index])
        # print("Prev. Hands", self.game.prev_hands[player_index])
        # print("Prev. Table", self.game.prev_table_cards)
        # # exit(1)
        # if len(self.game.hands[player_index]) < 10:
        #     exit(1)

        reward = 1

        # Illegal action
        if self.game.prev_was_illegals[player_index]:
            # reward = -self.game.max_penalty * self.game.max_num_cards_on_hand
            reward = -self.game.max_penalty

        card = self.game.prev_played_cards[player_index]

        # The agent did not take a turn until now; no information to provide.
        if card is None:
            reward = 0

        # Shot to the moon
        if trick_is_over and self.game.has_shot_the_moon(player_index):
            # reward = self.game.max_penalty * self.game.max_num_cards_on_hand
            reward = self.game.max_penalty

        ################ Current table ###################
        # If leading suit is heart
        if len(self.game.table_cards) > 0:
            if self.game.leading_suit == 2:
                table_ranks = [
                    card.rank for card in self.game.table_cards if card.suit == 2
                ]
                max_heart_rank = max(table_ranks)
                played_card = self.game.table_cards[player_index]
                if played_card.rank > max_heart_rank:
                    reward = -played_card.rank / 11
                else:
                    reward = played_card.rank / 11

        # If leading suit is spade
        if len(self.game.table_cards) > 0:
            if self.game.leading_suit == 3:
                table_ranks = [
                    card.rank for card in self.game.table_cards if card.suit == 3
                ]
                if 10 in table_ranks:
                    played_card = self.game.table_cards[player_index]
                    if played_card.rank > 10:
                        reward = -self.game.max_penalty
                    else:
                        reward = played_card.rank / 11

        ################ Previous table ##################
        # First trick: highest card
        if self.game.prev_was_first_trick:
            reward = 1 + self.game.prev_table_cards[player_index].rank / 11

        # If lead of trick: lowest card
        if len(self.game.prev_table_cards) > 0:
            if self.game.prev_table_cards[0] in self.game.prev_hands[player_index]:
                reward = 1 + 1 - self.game.prev_table_cards[player_index].rank / 11
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
                        reward = self.game.max_penalty
                    # Hearth + highest card
                    else:
                        multiplier = 1
                        if played_card.suit == 2:
                            multiplier = 2
                        reward = (
                            1
                            + self.game.prev_table_cards[player_index].rank
                            / 11
                            * multiplier
                        )

        # penalty = self.game.penalties[player_index]

        # if self.game.is_done():
        #     return -penalty

        if self.game.prev_trick_winner_index == player_index:
            assert self.game.prev_trick_penalty is not None
            reward = -self.game.prev_trick_penalty

        # Normalize between -1 and 1
        reward = reward / self.game.max_penalty
        return reward
        # return -penalty
