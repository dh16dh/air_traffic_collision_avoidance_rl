from typing import Union

Number = Union[int, float]


def nmi_to_km(nmi: Number) -> Number:
    return nmi * 1.852


def km_to_nmi(km: Number) -> Number:
    return km / 1.852


def ft_to_m(ft: Number) -> Number:
    return ft * 0.3048


def ft_to_km(ft: Number) -> Number:
    return ft_to_m(ft) / 1e3


def kmh_to_kms(kmh: Number) -> Number:
    return kmh / 3.6e3
