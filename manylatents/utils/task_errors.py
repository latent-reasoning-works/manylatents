"""Typed failures for finite task instruments; no downstream dependency."""


class ContractViolation(ValueError):
    """The supplied inputs violate the declared instrument contract."""


class NumericalFailure(FloatingPointError):
    """A mathematically valid operation cannot be represented safely."""
