"""Dependency-free proportional provenance ledger for membrane analysis."""


class ProportionalSourceLedger:
    """Track how much of a membrane state descends from each input time step.

    Leakage scales every source equally.  After a soft reset, ``redistribute``
    assigns the observed residual membrane to the pre-reset sources in the same
    proportions.  This is an attribution convention, not an additional neuron
    operation; source values always sum to the observed membrane state.
    """

    def __init__(self, decay):
        self.decay = float(decay)
        self.sources = []

    def charge(self, current_input):
        self.sources = [value * self.decay for value in self.sources]
        self.sources.append(float(current_input))

    def redistribute(self, target_total, eps=1e-12):
        source_total = self.total
        target_total = float(target_total)
        if abs(source_total) <= eps:
            if abs(target_total) <= eps:
                self.sources = [0.0 for _ in self.sources]
                return
            raise ValueError(
                f'cannot proportionally assign nonzero membrane {target_total} '
                'from zero source total'
            )
        scale = target_total / source_total
        self.sources = [value * scale for value in self.sources]

    @property
    def current(self):
        return self.sources[-1]

    @property
    def history(self):
        return sum(self.sources[:-1])

    @property
    def total(self):
        return sum(self.sources)

    def padded(self, length):
        return self.sources + [0.0] * (int(length) - len(self.sources))
