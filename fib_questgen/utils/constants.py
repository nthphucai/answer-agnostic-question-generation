TIME_PREPOSITION = ["before", "for", "during", "by", "after", "in", "on", "at"]
CARDINAL_PREPOSITION = ["at", "in", "on", "by"]

LOCATION_PREPOSITION = ["at", "in", "on", "from"]

POSITION_PREPOSITION = [
    "above",
    "among",
    "behind",
    "below",
    "beneath",
    "beside",
    "between",
    "by",
    "in",
    "inside",
    "on",
    "outside",
    "underneath",
]

DIRECTION_PREPOSITION = [
    "along",
    "away from",
    "down",
    "from",
    "into",
    "onto",
    "out of",
    "to",
    "toward",
    "up",
]

ALL_PREPOSITION = (
    TIME_PREPOSITION
    + LOCATION_PREPOSITION
    + DIRECTION_PREPOSITION
    + POSITION_PREPOSITION
)

PREP_MAP = {
    "GPE": LOCATION_PREPOSITION,
    "CARDINAL": CARDINAL_PREPOSITION,
    "DATE": TIME_PREPOSITION,
    "TIME": TIME_PREPOSITION,
    "ALL": ALL_PREPOSITION,
}
