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

        reward = 0

        # Illegal action
        if self.game.prev_was_illegals[player_index]:
            # reward = -self.game.max_penalty * self.game.max_num_cards_on_hand
            return -self.game.max_penalty

        card = self.game.prev_played_cards[player_index]

        # The agent did not take a turn until now; no information to provide.
        if card is None:
            return 0

        # Shot to the moon
        if trick_is_over and self.game.has_shot_the_moon(player_index):
            # reward = self.game.max_penalty * self.game.max_num_cards_on_hand
            return self.game.max_penalty

        ################ Previous table ##################
        # First trick: highest card
        if self.game.prev_was_first_trick:
            return 1 + self.game.prev_table_cards[player_index].rank / 11
        # else:
        #     hand = self.game.prev_hands[player_index]
        #     suits = ["club", "diamond", "heart", "spade"]
        #     handsuit = {"club": [], "diamond": [], "heart": [], "spade": []}
        #     for card in hand:
        #         suit = card.suit
        #         rank = card.rank
        #         handsuit[suits[suit]].append(rank)

        #     for suit in suits:
        #         if len(handsuit[suit]) == 0:
        #             reward_dan += 0.1 * len(hand)
        #         else:
        #             P = values_me = np.asarray(handsuit[suit]) + 0.00001
        #             Q = values_other = np.linspace(0, 12, len(values_me)) + 0.00001
        #             divergence = np.sum(P * np.log(P / Q))
        #             reward_dan -= divergence

        if len(self.game.prev_table_cards) > 0:
            # If lead of trick: lowest card
            if self.game.prev_table_cards[0] in self.game.prev_hands[player_index]:
                return 1 + 1 - self.game.prev_table_cards[player_index].rank / 11
            # If not lead of the trick:
            else:
                prev_hand_suits = [
                    card.suit for card in self.game.prev_hands[player_index]
                ]
                played_card = self.game.prev_table_cards[player_index]
                # If got no leading suit cards: Queen of Spades --> highest (hearth) card)
                if self.game.leading_suit not in prev_hand_suits:
                    # Queen of Spades
                    if played_card.suit == 3 and played_card.rank == 10:
                        reward += self.game.max_penalty
                    # Highest (hearth) card
                    else:
                        multiplier = 1
                        if played_card.suit == 2:
                            multiplier = 5
                        reward += (
                            1
                            + self.game.prev_table_cards[player_index].rank
                            / 11
                            * multiplier
                        )
                # If got leading suit cards: play the highest card, but smaller than the leading one on the table
                else:
                    table_ranks = [
                        card.rank
                        for card in self.game.prev_table_cards
                        if card.suit == self.game.prev_leading_suit
                    ]
                    max_rank = max(table_ranks)
                    # If played card is greater than max leading_card --> penalty
                    if played_card.rank > max_rank:
                        return -self.game.max_penalty
                    # If played card is smaller than max leading_card --> reward
                    else:
                        return 1 + played_card.rank / 11

        if self.game.prev_trick_winner_index == player_index:
            assert self.game.prev_trick_penalty is not None
            return -self.game.prev_trick_penalty
        else:
            reward += 1

        # Normalize between -1 and 1
        # reward = reward / self.game.max_penalty
        return reward
