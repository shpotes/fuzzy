import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from src.utils import *

def get_fuzzy_variables():
    """
    Fuzzy variables getter
    """
    holes = ctrl.Antecedent(np.arange(-2, 6, 0.1), 'holes')
    holes_no = fuzz.membership.dsigmf(holes.universe, 0, 2, 0.5, 4)
    holes_no = np.where(holes_no >= 0, holes_no, 0)
    holes_no = holes_no * 1 / holes_no.max()

    holes['no'] = holes_no
    holes_mid = fuzz.membership.dsigmf(holes.universe, 1, 2, 2, 4)
    holes_mid = np.where(holes_mid >= 0, holes_mid, 0)
    holes_mid = holes_mid * 1 / holes_mid.max()

    holes['mid'] = holes_mid
    holes['many'] = fuzz.membership.sigmf(holes.universe, 2, 2)

    piles = ctrl.Antecedent(np.arange(-4, 5, 0.1), 'piles')
    piles['nice'] = fuzz.membership.sigmf(piles.universe, 0.5, -4)
    piles['not_that_bad'] = fuzz.membership.gaussmf(piles.universe, 1, 0.5)
    piles['wtf'] = fuzz.membership.sigmf(piles.universe, 1.5, 3)

    wells = ctrl.Antecedent(np.arange(-4, 4, 0.1), 'wells')
    wells['removed'] = fuzz.membership.sigmf(wells.universe, 0, -3)
    wells['accumulate_more'] = fuzz.membership.sigmf(wells.universe, 1, 3)

    mood = ctrl.Consequent(np.arange(15), 'mood')
    mood[':)'] = fuzz.membership.gaussmf(mood.universe, 2, 1)
    mood[':|'] = fuzz.membership.gbellmf(mood.universe, 2, 3, 5)
    mood[':('] = fuzz.membership.gaussmf(mood.universe, 8, 1)
    mood[':(('] = fuzz.membership.sigmf(mood.universe, 12, 2)

    return (holes, piles, wells), mood


def get_expert_model(t_norm=t_min, s_norm=s_max, deffuz='centroid', with_variables=False):
    """
    Expert model getter
    """
    features, mood = get_fuzzy_variables()
    holes, piles, wells = features
    rules = [
        ctrl.Rule(
            ~holes['no'] & ~holes['mid'] & ~holes['many'], mood[':(('],
            and_func=t_drastic
        ),
        ctrl.Rule(
            holes['no'] & (piles['not_that_bad'] | wells['accumulate_more']), mood[':)'],
            and_func=t_norm, or_func=s_norm
        ),
        ctrl.Rule(
            piles['nice'] & wells['removed'], mood[':)'],
            and_func=t_norm, or_func=s_norm
        ),
        ctrl.Rule(
            holes['mid'] & wells['accumulate_more'], mood[':|'],
            and_func=t_norm, or_func=s_norm
        ),
        ctrl.Rule(
            (~holes['no'] | ~holes['mid']) & wells['accumulate_more'], mood[':('],
            and_func=t_norm, or_func=s_norm
        ),
        ctrl.Rule(
            piles['nice'] & wells['accumulate_more'], mood[':)'],
            and_func=t_norm, or_func=s_norm
        ),
        ctrl.Rule(
            piles['not_that_bad'] & wells['accumulate_more'], mood[':|'],
            and_func=t_norm, or_func=s_norm
        ),
        ctrl.Rule(
            piles['wtf'] | ~(holes['no'] | holes['mid']), mood[':('],
            and_func=t_norm, or_func=s_norm
        ),
        ctrl.Rule(
            (holes['mid'] | piles['not_that_bad']), mood[':|'],
            and_func=t_norm, or_func=s_norm
        ),
        ctrl.Rule(
            wells['accumulate_more'] & piles['wtf'] & holes['many'], mood[':('],
            and_func=t_norm, or_func=s_norm
        )
    ]

    tetris_brain = ctrl.ControlSystemSimulation(
        ctrl.ControlSystem(rules)
    )

    if with_variables:
        return tetris_brain, (holes, piles, wells), mood

    return tetris_brain


def make_inference(holes, piles, wells, expert):
    """
    Run inferences on the given expert using the given parameters
    """
    expert.input['holes'] = holes
    expert.input['piles'] = piles
    expert.input['wells'] = wells

    expert.compute()

    return expert.output['mood']
