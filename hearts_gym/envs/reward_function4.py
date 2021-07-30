"""
The reward function an agent optimizes to win at Hearts.
"""

import numpy as np

from hearts_gym.utils.typing import Reward
from .hearts_env import HeartsEnv


def normalize(value, shift=26, scale=52):
    return (value + shift) / scale


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
        global_max_rank = 12
        prev_played_card = self.game.prev_played_cards[player_index]

        # The agent did not take a turn until now; no information to provide.
        if prev_played_card is None or len(self.game.prev_table_cards) == 0:
            return normalize(0)

        prev_leading_card = self.game.prev_table_cards[0]

        prev_hand_suits = np.array(
            [card.suit for card in self.game.prev_hands[player_index]]
        )
        prev_table_ranks = [
            card.rank
            for card in self.game.prev_table_cards
            if card.suit == self.game.prev_leading_suit
        ]
        prev_max_rank = max(prev_table_ranks)

        # Illegal action
        if self.game.prev_was_illegals[player_index]:
            return normalize(-self.game.max_penalty)

        # Shot to the moon
        if trick_is_over and self.game.has_shot_the_moon(player_index):
            return normalize(self.game.max_penalty)

        ################ Previous table ##################
        # First trick: highest card
        if self.game.prev_was_first_trick:
            return normalize(prev_played_card.rank)

        # If the agent is the leader of the trick: max reward for the lowest (non-hearth) card.
        # If the agent played hearth even if it could play a different suit, penalize it.
        # if prev_leading_card == prev_played_card:
        #     multiplier = 1
        #     if (
        #         len(prev_hand_suits[prev_hand_suits != 2]) > 0
        #         and prev_played_card.suit == 2
        #     ):
        #         multiplier = -global_max_rank
        #     value = (1 - prev_played_card.rank / global_max_rank) * multiplier
        #     return normalize(value)
        if prev_leading_card == prev_played_card:
            if (
                len(prev_hand_suits[prev_hand_suits != 2]) > 0
                and prev_played_card.suit == 2
            ):
                return normalize(-prev_played_card.rank)
            else:
                return normalize(prev_played_card.rank)
        # If the agent is not the leader of the trick: max reward for the highest (hearth) card (or QoS)
        else:
            # If the agent has no leading suit cards:
            if self.game.leading_suit not in prev_hand_suits:
                multiplier = 1
                # If the agent has the QoS, play it:
                if prev_played_card.suit == 3 and prev_played_card.rank == 10:
                    return normalize(self.game.max_penalty)
                # Otherwise, play the highest (hearth) card: if the agent had a hearth and did not play it, penalize it.
                # Max reward for the highest (health) card
                elif 2 in prev_hand_suits and prev_played_card.suit != 2:
                    value = (
                        1 - prev_played_card.rank / global_max_rank
                    ) * -global_max_rank
                    return normalize(value)
                value = (prev_played_card.rank) * multiplier
                return normalize(value)
            # If the agent has no leading suit cards: play the highest card, but smaller than the leading one on the table
            else:
                # If played card is greater than max leading_card --> penalty
                if prev_played_card.rank > prev_max_rank:
                    return normalize(-prev_played_card.rank)
                # If played card is smaller than max leading_card --> reward
                else:
                    return normalize(prev_played_card.rank)

        # # Previous trick winner
        if self.game.prev_trick_winner_index == player_index:
            assert self.game.prev_trick_penalty is not None
            return normalize(-self.game.max_penalty)

        return 0
