def normalize_range_min1_plus1(observations, min_, max_):
    normalized = normalize_range_0_1(observations, min_, max_)
    normalized = normalized * 2 - 1
    return normalized


def normalize_range_0_1(observations, min_, max_):
    normalized = (observations - min_) / (max_ - min_)
    return normalized
