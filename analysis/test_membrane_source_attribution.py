import unittest

from analysis.membrane_source_ledger import NoResetLedger, SoftResetLedger


class SourceLedgerTest(unittest.TestCase):
    def test_soft_reset_source_decomposition(self):
        ledger = SoftResetLedger(decay=0.5, threshold=1.0)
        ledger.charge(0.6)
        self.assertAlmostEqual(ledger.pre_reset_total, 0.6)
        ledger.charge(0.6)
        ledger.reset(1.0)
        self.assertAlmostEqual(ledger.current_input, 0.6)
        self.assertAlmostEqual(ledger.past_input, 0.3)
        self.assertAlmostEqual(ledger.reset_loss, -1.0)
        self.assertAlmostEqual(ledger.post_reset_total, -0.1)
        ledger.charge(0.0)
        self.assertAlmostEqual(ledger.past_input, 0.45)
        self.assertAlmostEqual(ledger.reset_loss, -0.5)

    def test_no_reset_ledger_preserves_past_input(self):
        ledger = NoResetLedger(decay=0.5)
        ledger.charge(0.6)
        ledger.charge(0.6)
        ledger.charge(0.0)
        self.assertAlmostEqual(ledger.current_input, 0.0)
        self.assertAlmostEqual(ledger.past_input, 0.45)
        self.assertAlmostEqual(ledger.total, 0.45)


if __name__ == '__main__':
    unittest.main()
