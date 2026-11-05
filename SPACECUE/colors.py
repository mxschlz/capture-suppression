"""
SPACECUE Standard Color Scheme and Palette Definitions.

Conventions:
- Target-related activity & behavior: Green
    * Younger Adults: Light Green (#4CAF50)
    * Older Adults:   Dark Green  (#1B5E20)
    * General / GA:   Forest Green (#2E7D32)
- Distractor-related activity & behavior: Red
    * Younger Adults: Light Red   (#EF5350)
    * Older Adults:   Dark Red    (#B71C1C)
    * General / GA:   Crimson Red (#C62828)
- Control-related activity & behavior: Grey
    * Younger Adults: Light Grey  (#9E9E9E)
    * Older Adults:   Dark Grey   (#424242)
    * General / GA:   Medium Grey (#757575)
- Difference waves (e.g. N2ac, Pd, Contra-Ipsi): Black (#000000)
"""

# Base Hex Codes
COLOR_TARGET = "#2E7D32"
COLOR_TARGET_YOUNG = "#4CAF50"
COLOR_TARGET_OLD = "#1B5E20"

COLOR_DISTRACTOR = "#C62828"
COLOR_DISTRACTOR_YOUNG = "#EF5350"
COLOR_DISTRACTOR_OLD = "#B71C1C"

COLOR_CONTROL = "#757575"
COLOR_CONTROL_YOUNG = "#9E9E9E"
COLOR_CONTROL_OLD = "#424242"

COLOR_DIFF = "#000000"
COLOR_DIFFERENCE = "#000000"

# Mapping Dictionaries
COHORT_COLORS = {
    'Young': {
        'target': COLOR_TARGET_YOUNG,
        'distractor': COLOR_DISTRACTOR_YOUNG,
        'control': COLOR_CONTROL_YOUNG,
        'diff': COLOR_DIFF,
    },
    'Old': {
        'target': COLOR_TARGET_OLD,
        'distractor': COLOR_DISTRACTOR_OLD,
        'control': COLOR_CONTROL_OLD,
        'diff': COLOR_DIFF,
    },
    'General': {
        'target': COLOR_TARGET,
        'distractor': COLOR_DISTRACTOR,
        'control': COLOR_CONTROL,
        'diff': COLOR_DIFF,
    }
}

STIMULUS_COLORS = {
    'Target': {
        'Young': COLOR_TARGET_YOUNG,
        'Old': COLOR_TARGET_OLD,
        'General': COLOR_TARGET,
    },
    'Distractor': {
        'Young': COLOR_DISTRACTOR_YOUNG,
        'Old': COLOR_DISTRACTOR_OLD,
        'General': COLOR_DISTRACTOR,
    },
    'Control': {
        'Young': COLOR_CONTROL_YOUNG,
        'Old': COLOR_CONTROL_OLD,
        'General': COLOR_CONTROL,
    },
    'Difference': {
        'Young': COLOR_DIFF,
        'Old': COLOR_DIFF,
        'General': COLOR_DIFF,
    }
}

# Behavioral Metric Mapping
BEHAVIORAL_COLORS = {
    # Towardness metrics
    'target_towardness': COLOR_TARGET,
    'distractor_towardness': COLOR_DISTRACTOR,
    'control_towardness': COLOR_CONTROL,
    
    # Towardness by cohort
    'target_towardness_young': COLOR_TARGET_YOUNG,
    'target_towardness_old': COLOR_TARGET_OLD,
    'distractor_towardness_young': COLOR_DISTRACTOR_YOUNG,
    'distractor_towardness_old': COLOR_DISTRACTOR_OLD,
    'control_towardness_young': COLOR_CONTROL_YOUNG,
    'control_towardness_old': COLOR_CONTROL_OLD,
    
    # Biases
    'target_towardness_bias': COLOR_TARGET,
    'distractor_towardness_bias': COLOR_DISTRACTOR,
    'control_towardness_bias': COLOR_CONTROL,
}

def get_color(stimulus: str, cohort: str = None) -> str:
    """
    Get standardized color for stimulus and cohort.
    
    Parameters
    ----------
    stimulus : str
        'target', 'distractor', 'control', or 'diff'/'difference'.
    cohort : str, optional
        'Young', 'Old', or None (returns general stimulus color).
        
    Returns
    -------
    str : Hex color code.
    """
    stim = stimulus.capitalize()
    if stim in ['Diff', 'Difference']:
        return COLOR_DIFF
    
    if cohort is None:
        c_key = 'General'
    else:
        c_key = 'Young' if cohort.lower().startswith('y') else 'Old'
        
    return STIMULUS_COLORS.get(stim, {}).get(c_key, COLOR_CONTROL)
