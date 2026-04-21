EEG_TRIGGER_MAP = {}

target_positions = {"Target-1-": 0, "Target-2-": 100, "Target-3-": 200}
distractor_positions = {"Singleton-1-": 10, "Singleton-2-": 20, "Singleton-3-": 30}
transition_probabilities = {"HP-Distractor-Loc-1-0.8": 1, "HP-Distractor-Loc-1-0.6": 2,
                            "HP-Distractor-Loc-3-0.6": 3, "HP-Distractor-Loc-3-0.8": 4}

for t_pos, t_val in target_positions.items():
    for d_pos, d_val in distractor_positions.items():
        if t_pos.split('-')[1] != d_pos.split('-')[1]:
            for prob_id, prob_val in transition_probabilities.items():
                trigger_val = t_val + d_val + prob_val
                combination_key = f"{t_pos}{d_pos}{prob_id}"
                EEG_TRIGGER_MAP[combination_key] = trigger_val
