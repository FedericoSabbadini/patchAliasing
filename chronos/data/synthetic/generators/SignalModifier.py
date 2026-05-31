import numpy as np

class SignalModifier:
    def __init__(self, fs: float = 1.0, P: int = 16):
        """
        Initializes the modifier with sampling and structural metadata.
        
        :param fs: Sampling frequency (Hz), required for 'hz' mode.
        :param P: Patch size, required for 'cpp' mode.
        """
        self.fs = fs
        self.P = P

    def addComponent(self, 
                     signal: np.ndarray, 
                     amplitude: float, 
                     frequency: float, 
                     phase: float = 0.0, 
                     mode: str = "hz", 
                     normalize: bool = False) -> np.ndarray:
        """
        Injects a sinusoidal component into an existing time-series signal.
        
        :param signal: The base NumPy array signal.
        :param amplitude: Peak amplitude of the injected wave.
        :param frequency: Frequency value (either Hz or Cycles Per Patch).
        :param phase: Phase shift in radians.
        :param mode: Injected frequency context, either 'hz' or 'cpp'.
        :param normalize: If True, scales the base signal to std=1 before injection.
        :return: A new NumPy array containing the modified signal.
        """
        # Work on a copy to prevent unintended in-place mutation of the original signal
        modified_signal = signal.copy()
        
        if normalize:
            std = modified_signal.std()
            modified_signal = modified_signal / std if std > 1e-8 else modified_signal

        n = np.arange(len(modified_signal))
        
        if mode == "hz":
            t = n / self.fs
            comp = amplitude * np.sin(2 * np.pi * frequency * t + phase)
        elif mode == "cpp":
            period_samples = self.P / frequency
            comp = amplitude * np.sin(2 * np.pi * n / period_samples + phase)
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose either 'hz' or 'cpp'.")
            
        return modified_signal + comp