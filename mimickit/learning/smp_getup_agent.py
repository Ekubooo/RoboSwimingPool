"""SMP with train-mode restoration before the caller resets after evaluation."""
from learning.smp_agent import SMPAgent


class SMPGetupAgent(SMPAgent):
    def test_model(self, num_episodes):
        previous = self._mode
        try:
            return super().test_model(num_episodes)
        finally:
            # BaseAgent.train_model resets immediately after test_model returns.
            # Ensure that reset uses the 90/10 training distribution, not starts.
            self.set_mode(previous)

    def _output_train_model(self, iter, out_model_file, int_out_dir):
        super()._output_train_model(iter, out_model_file, int_out_dir)
        self._env.clear_train_diagnostics()
