EEG_TRIGGER_MAP = {
    "test_trigger": 1,
    "camera_calibration_onset": 2,
    "camera_calibration_offset": 3,
    "block_onset": 4,
    "block_offset": 5,
    "experiment_onset": 6,
    "experiment_offset": 7,
    "trial_onset": 8,
    "trial_offset": 9
}

target_positions = {"Target-1-": 0, "Target-2-": 100, "Target-3-": 200}
distractor_positions = {"Singleton-1-": 10, "Singleton-2-": 20, "Singleton-3-": 30}
transition_probabilities = {"HP-Distractor-Loc-1-0.8": 1, "HP-Distractor-Loc-1-0.6": 2,
                            "HP-Distractor-Loc-3-0.6": 3, "HP-Distractor-Loc-3-0.8": 4}

for t_pos, t_val in target_positions.items():
    for d_pos, d_val in distractor_positions.items():
        if t_pos != d_pos:
            for prob_id, prob_val in transition_probabilities.items():
                trigger_val = t_val + d_val + prob_val
                combination_key = f"{t_pos}{d_pos}{prob_id}"
                EEG_TRIGGER_MAP[combination_key] = trigger_val
