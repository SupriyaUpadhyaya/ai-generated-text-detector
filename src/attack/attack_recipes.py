from textattack import Attack
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.pre_transformation import MaxModificationRate
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.goal_functions import InputReduction, UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR
from textattack.transformations import WordDeletion, WordSwapWordNet

from textattack.attack_recipes import AttackRecipe


class InputReductionFeng2018(AttackRecipe):
    """Feng, Wallace, Grissom, Iyyer, Rodriguez, Boyd-Graber. (2018).

    Pathologies of Neural Models Make Interpretations Difficult.

    https://arxiv.org/abs/1804.07781
    """

    @staticmethod
    def build(model_wrapper):
        # At each step, we remove the word with the lowest importance value until
        # the model changes its prediction.
        transformation = WordDeletion()

        constraints = [RepeatModification(), StopwordModification(), MaxModificationRate(max_rate=0.05, min_threshold=1)]
        #
        # Goal is untargeted classification
        #
        goal_function = InputReduction(model_wrapper, maximizable=True)
        #
        # "For each word in an input sentence, we measure its importance by the
        # change in the confidence of the original prediction when we remove
        # that word from the sentence."
        #
        # "Instead of looking at the words with high importance values—what
        # interpretation methods commonly do—we take a complementary approach
        # and study how the model behaves when the supposedly unimportant words are
        # removed."
        #
        search_method = GreedyWordSwapWIR(wir_method="delete")

        return Attack(goal_function, constraints, transformation, search_method)

class PWWSRen2019_threshold(AttackRecipe):
    """Add threshold
    """

    @staticmethod
    def build(model_wrapper, target_max_score=None):
        transformation = WordSwapWordNet()
        constraints = [RepeatModification(), StopwordModification(), MaxModificationRate(max_rate=0.05, min_threshold=5)]
        goal_function = UntargetedClassification(model_wrapper, target_max_score=target_max_score)
        # search over words based on a combination of their saliency score, and how efficient the WordSwap transform is
        search_method = GreedyWordSwapWIR("weighted-saliency")
        return Attack(goal_function, constraints, transformation, search_method)

"""
Pruthi2019: Combating with Robust Word Recognition
=================================================================

"""

from textattack import Attack
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.constraints.pre_transformation import (
    MinWordLength,
    RepeatModification,
    StopwordModification,
)
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedySearch
from textattack.transformations import (
    CompositeTransformation,
    WordSwapNeighboringCharacterSwap,
    WordSwapQWERTY,
    WordSwapRandomCharacterDeletion,
    WordSwapRandomCharacterInsertion,
)


class Pruthi2019_threshold(AttackRecipe):
    """An implementation of the attack used in "Combating Adversarial
    Misspellings with Robust Word Recognition", Pruthi et al., 2019.

    This attack focuses on a small number of character-level changes that simulate common typos. It combines:
        - Swapping neighboring characters
        - Deleting characters
        - Inserting characters
        - Swapping characters for adjacent keys on a QWERTY keyboard.

    https://arxiv.org/abs/1905.11268

    :param model: Model to attack.
    :param max_num_word_swaps: Maximum number of modifications to allow.
    """

    @staticmethod
    def build(model_wrapper, max_num_word_swaps=1):
        # a combination of 4 different character-based transforms
        # ignore the first and last letter of each word, as in the paper
        transformation = CompositeTransformation(
            [
                WordSwapNeighboringCharacterSwap(
                    random_one=False, skip_first_char=True, skip_last_char=True
                ),
                WordSwapRandomCharacterDeletion(
                    random_one=False, skip_first_char=True, skip_last_char=True
                ),
                WordSwapRandomCharacterInsertion(
                    random_one=False, skip_first_char=True, skip_last_char=True
                ),
                WordSwapQWERTY(
                    random_one=False, skip_first_char=True, skip_last_char=True
                ),
            ]
        )
        # only edit words of length >= 4, edit max_num_word_swaps words.
        # note that we also are not editing the same word twice, so
        # max_num_word_swaps is really the max number of character
        # changes that can be made. The paper looks at 1 and 2 char attacks.
        constraints = [
            MinWordLength(min_length=10),
            StopwordModification(),
            MaxWordsPerturbed(max_num_words=max_num_word_swaps),
            RepeatModification(),
        ]
        # untargeted attack
        goal_function = UntargetedClassification(model_wrapper)
        search_method = GreedySearch()
        return Attack(goal_function, constraints, transformation, search_method)