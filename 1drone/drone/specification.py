from stl import Atom, Reach, Implies, Eventually, Until, Globally, And


def build_strel_specification(grid_side: int = 10):
    low_batt = Atom(var_index=4, threshold=0.3, lte=True, labels=[1])
    high_batt = Atom(var_index=4, threshold=0.3, lte=False, labels=[1])
    full_batt = Atom(var_index=4, threshold=0.9, lte=False, labels=[1])
    alive = Atom(var_index=4, threshold=0.05, lte=False, labels=[1])

    target_y_threshold = 0.75 * float(grid_side)
    at_target = Atom(var_index=1, threshold=target_y_threshold, lte=False, labels=[2])
    free_base = Atom(var_index=4, threshold=0.1, lte=True, labels=[0])

    go_to_target = Reach(
        left_child=alive,
        right_child=at_target,
        d1=0.0001,
        d2=0.51,
        left_label=[1],
        right_label=[2],
        distance_function="Euclid",
    )
    phi_2 = Implies(
        high_batt,
        Eventually(go_to_target, left_time_bound=0, right_time_bound=20),
    )

    reach_free_base = Reach(
        left_child=alive,
        right_child=free_base,
        d1=0.0001,
        d2=0.91,
        left_label=[1],
        right_label=[0],
        distance_function="Euclid",
    )
    stay_on_base = Eventually(
        Until(reach_free_base, full_batt, unbound=True),
        left_time_bound=0,
        right_time_bound=12,
    )
    phi_3 = Implies(low_batt, stay_on_base)

    return Globally(And(phi_2, And(phi_3, alive)), unbound=True)
