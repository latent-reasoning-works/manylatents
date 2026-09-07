"""Explicit failures for computations whose measurement is unavailable."""


class MeasurementUnavailable(ValueError):
    """The requested measurement cannot be computed from the supplied evidence.

    Raised instead of returning a numeric sentinel, substituting a parameter, or
    reporting a statistic conditioned on silently discarded observations.
    Callers may catch this to display an unavailable state and its reason; they
    must not turn it into a numeric score or a correction target.

    A partially measurable parameter sweep retains instances as result values
    under the requested entry's key. A wholly unavailable sweep still raises.
    """

    def to_dict(self) -> dict[str, str]:
        """Preserve the unavailable state and reason in saved/logged results."""
        return {"status": "unavailable", "reason": str(self)}
