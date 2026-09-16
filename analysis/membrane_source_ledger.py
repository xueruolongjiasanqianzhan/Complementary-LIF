"""Dependency-free source ledgers for membrane attribution analysis."""


class SoftResetLedger:
    """Exact source ledger for ``v <- decay * v + x - spike * threshold``."""

    def __init__(self, decay, threshold):
        self.decay = float(decay)
        self.threshold = float(threshold)
        self.input_terms = []
        self.reset_terms = []

    def charge(self, current_input):
        self.input_terms = [value * self.decay for value in self.input_terms]
        self.reset_terms = [value * self.decay for value in self.reset_terms]
        self.input_terms.append(float(current_input))

    def reset(self, spike):
        if float(spike) != 0.0:
            self.reset_terms.append(-float(spike) * self.threshold)

    @property
    def current_input(self):
        return self.input_terms[-1]

    @property
    def past_input(self):
        return sum(self.input_terms[:-1])

    @property
    def reset_loss(self):
        return sum(self.reset_terms)

    @property
    def pre_reset_total(self):
        return sum(self.input_terms) + sum(self.reset_terms)

    @property
    def post_reset_total(self):
        return self.pre_reset_total


class NoResetLedger:
    """Input-source ledger for the LS state, which has no reset term."""

    def __init__(self, decay):
        self.decay = float(decay)
        self.input_terms = []

    def charge(self, current_input):
        self.input_terms = [value * self.decay for value in self.input_terms]
        self.input_terms.append(float(current_input))

    @property
    def current_input(self):
        return self.input_terms[-1]

    @property
    def past_input(self):
        return sum(self.input_terms[:-1])

    @property
    def total(self):
        return sum(self.input_terms)
