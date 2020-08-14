import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def get_expert_model():
    holes = ctrl.Antecedent(np.arange(60), 'holes')
    holes['no'] = fuzz.membership.trimf(holes.universe, [0, 0, 0])
    holes['mid'] = fuzz.membership.trimf(holes.universe, [0, 1, 3])
    holes['full'] = fuzz.membership.trapmf(holes.universe, [2, 4, 60, 60])

    piles = ctrl.Antecedent(np.arange(-4, 5), 'piles')
    piles['nice'] = fuzz.membership.trapmf(piles.universe, [-4, -4, 0, 1])
    piles['not_that_bad'] = fuzz.membership.trimf(piles.universe, [0, 1, 2])
    piles['wtf'] = fuzz.membership.trapmf(piles.universe, [1, 2, 5, 5])

    wells = ctrl.Antecedent(np.arange(20), 'wells')
    wells['meh'] = fuzz.membership.trimf(U_wells, [0, 0, 2])
    wells['wuuu'] = fuzz.membership.trapmf(U_wells, [1, 3, 20, 20])

    mood = ctrl.Consequent(np.arange(11), 'mood')
    mood[':)'] = fuzz.membership.trimf(U_mood, [0, 2, 3])
    mood[':|'] = fuzz.membership.trapmf(U_mood, [2, 4, 5, 7])
    mood[':('] = fuzz.membership.trapmf(U_mood, [6, 8, 10, 10])

    rules = [
        ctrl.Rule(holes['no']   & piles['nice']         & wells['meh'], mood[':|']),
        ctrl.Rule(holes['no']   & piles['nice']         & wells['wuuu'], mood[':)']),
        # ctrl.Rule(holes['no'] & piles['not_that_bad'] & wells['meh'], mood[':']),
        ctrl.Rule(holes['no']   & piles['not_that_bad'] & wells['wuuu'], mood[':)']),
        # ctrl.Rule(holes['no'] & piles['wtf']          & wells['meh'], mood[':']),
        ctrl.Rule(holes['no']   & piles['wtf']          & wells['wuuu'], mood[':)']),
        ctrl.Rule(holes['mid']  & piles['nice']         & wells['meh'], mood[':)']),
        ctrl.Rule(holes['mid']  & piles['nice']         & wells['wuuu'], mood[':)']),
        ctrl.Rule(holes['mid']  & piles['not_that_bad'] & wells['meh'], mood[':)']),
        ctrl.Rule(holes['mid']  & piles['not_that_bad'] & wells['wuuu'], mood[':|']),
        ctrl.Rule(holes['mid']  & piles['wtf']          & wells['meh'], mood[':(']),
        ctrl.Rule(holes['mid']  & piles['wtf']          & wells['wuuu'], mood[':|']),
        ctrl.Rule(holes['full'] & piles['nice']         & wells['meh'], mood[':)']),
        ctrl.Rule(holes['full'] & piles['nice']         & wells['wuuu'], mood[':|']),
        ctrl.Rule(holes['full'] & piles['not_that_bad'] & wells['meh'], mood[':(']),
        ctrl.Rule(holes['full'] & piles['not_that_bad'] & wells['wuuu'], mood[':|']),
        ctrl.Rule(holes['full'] & piles['wtf']          & wells['meh'], mood[':(']),
        ctrl.Rule(holes['full'] & piles['wtf']          & wells['wuuu'], mood[':(']),
    ]

    tetris_brain = ctrl.ControlSystemSimulation(
        ctrl.ControlSystem(rules)
    )

    return tetris_brain


def make_inference(holes, piles, wells, brain):
    brain.input['holes'] = holes
    brain.input['piles'] = piles
    brain.input['wells'] = wells

    brain.compute()

    return brain.output['mood']
