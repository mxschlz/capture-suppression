import seaborn as sns
# Uni Lübeck colors palette consists of 8 different colors with 4 gradients each. The naming of the dictionary below
# follows the eight colors (1 - 8, red - blue), plus one of the four gradients (1 - 4, light - dark). Color values
# range from 11 to 84. So, the color 11 would equate to color red light, and 84 would yield a dark blue color. 42 would
# give a green color with mid gradient.
COLORS_RGB = {
    11: [232, 291, 173],  # broken
    12: [228, 32, 50],
    13: [181, 22, 33],
    14: [127, 21, 24],
    21: [240, 205, 178],
    22: [236, 116, 4],
    23: [203, 81, 25],
    24: [129, 53, 19],
    31: [236, 218, 136],
    32: [250, 187, 0],
    33: [191, 134, 20],
    34: [117, 83, 17],
    41: [219, 215, 187],
    42: [149, 188, 14],
    43: [126, 133, 37],
    44: [50, 88, 75],
    51: [205, 219, 216],
    52: [59, 178, 160],
    53: [36, 143, 133],
    54: [0, 90, 91],
    61: [198, 220, 226],
    62: [0, 174, 199],
    63: [0, 145, 168],
    64: [0, 97, 122],
    65: [0, 75, 90],
    71: [194, 217, 230],
    72: [60, 169, 213],
    73: [0, 131, 173],
    74: [0, 88, 119],
    81: [195, 217, 237],
    82: [110, 165, 206],
    83: [0, 106, 163],
    84: [0, 70, 114],
    91: [214, 223, 228],
    92: [191, 203, 213],
    93: [140, 158, 171],
    94: [88, 105, 121],
    101: [228, 228, 228],
    102: [208, 208, 208],
    103: [157, 157, 157],
    104: [101, 101, 101],
    111: [237, 226, 228],
    112: [218, 214, 203],
    113: [163, 160, 145],
    114: [117, 115, 105]
}

COLORS_HEX = {
    11: "#E8E1AD",
    12: "#E42032",
    13: "#B51621",
    14: "#7F1518",
    21: "#F0CDA2",
    22: "#EC7404",
    23: "#CB5119",
    24: "#813513",
    31: "#ECDAA8",
    32: "#FABB00",
    33: "#BF8614",
    34: "#755311",
    41: "#DBD7BB",
    42: "#95BC0E",
    43: "#7E8525",
    44: "#32584B",
    51: "#CDD9D8",
    52: "#3BB2A0",
    53: "#248F85",
    54: "#005A5B",
    61: "#C6DCE6",
    62: "#00AEC7",
    63: "#0091A8",
    64: "#00617A",
    65: "#004B5A",
    71: "#C2D9E6",
    72: "#3CA9D5",
    73: "#0083AD",
    74: "#005877",
    81: "#C3D9ED",
    82: "#6EA5CE",
    83: "#006AA3",
    84: "#004672",
    91: "#D6DFE4",
    92: "#BFCBD5",
    93: "#8C9EAB",
    94: "#586979",
    101: "#E4E4E4",
    102: "#D0D0D0",
    103: "#9D9D9D",
    104: "#656565",
    111: "#EDE2E4",
    112: "#DAD6CB",
    113: "#A3A091",
    114: "#757369"
}

values_scaled = [(r / 255, g / 255, b / 255) for r, g, b in COLORS_RGB.values()]
COLORS_RBG_SCALED = {k: v for k, v in zip(list(COLORS_RGB.keys()), values_scaled)}

palette = sns.color_palette(COLORS_HEX.values())
palette = {k: v for k, v in zip(list(COLORS_HEX.keys()), palette)}


def get_subpalette(subcolor_keys, palette=palette):
    subpalette = {}
    for key in subcolor_keys:
        subpalette[key] = palette[key]
    return subpalette


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib
    plt.ion()

    matplotlib.use("Qt5Agg")

    cmap = LinearSegmentedColormap.from_list("my_cmap", values_scaled)

    plt.imshow([[x for x in range(len(COLORS_HEX.keys()))]], cmap=cmap)

    plt.savefig("colors.svg")
