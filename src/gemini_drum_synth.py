from dataclasses import dataclass
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import butter, lfilter, iirpeak, sosfilt
# from scipy.signal import sosfreqz


# Sampling parameters

@dataclass
class DrumLayer:
    """
    Generate a drum sound using basic synthesis.

    Parameters:
    - sample_rate: sample rate in number of individual samples
    - output_file_bitdepth: bit depth of file to be generated
    - src_type: type of audio signal to be generated as a basis for the drum layer sound
    - mod_type: type of modulation being applied to signal if not specified in src_type
    - env_type: type of envelope to apply to signal: linear, exponential 1, exponential 2, exponential 3, punchy, double peak 
    - level: audio level 0-1 of output signal 
    - frequency: Fundamental frequency of the oscillator or in the case of noise the cutoff filter applied to noise (Hz)
    - detune: offers a detuning function that adds an oscillator of a fix difference frequency specified by this value
    - attack: time in seconds for signal to rise to peak
    - decay: time in seconds for signal to drop to persistent zero after peak 
    - mod_amount: amount or amplitude of modulation to effect oscillation signal
    - mod_rate: frequency of modulated signal

    """
    sample_rate: int = 88200  # Sample rate (Hz)
    output_file_bitdepth: int = 16
    src_type: str = 'square'
    mod_type: str = 'exp'
    env_type: str = 'linear'
    level: float = .5
    frequency: int = 440
    detune: int = 10
    attack: float = .01       # Duration of the synthesized sound (seconds)
    decay: float = 2.0
    mod_amount: float = 1.0
    mod_rate: int = 220
    velocity: float = .5
    dynamic_filter: float = 0

    wave_guide_mix: float = 0
    bit_crushing: float = 0
    analog_drive: float = 0
    clean_gain: float = 0
    delay: float = 0
    natural_reverb: float = 0
    filter_cutoff: float = 0
    filter_type: str = 'high'
    filter_resonance: float = 0

    def __post_init__(self):
        self.duration = self.attack + self.decay
        self.num_samples = int(self.duration * self.sample_rate)
        self.oscillator_functions = {'sine':self.gen_sine_wave,'saw':self.gen_saw_wave,'rev_saw':self.gen_rev_saw_wave,
                                'square':self.gen_square_wave,'tri':self.gen_tri_wave,'white_noise':self.gen_white_noise,
                                'blue_noise':self.gen_blue_noise,'pink_noise':self.gen_pink_noise,'brown_noise':self.gen_brown_noise
                                }
        # if self.b is None:
        #     self.b = 'Bravo'
        # if self.c is None:
        #     self.c = 'Charlie'

    def apply_spectra(self):
        """Run code to generate source audio that will be used as basis for drum sounds"""        
        self.source = self.oscillator_functions[self.source_type]()

    def gen_sine_wave(self, pitch_override=0):
        """Generate Sine Wave"""
        if not pitch_override:
            pitch_override = self.frequency
        t = np.linspace(0, self.duration, self.num_samples, endpoint=False)
        sine_wave = np.sin(2 * np.pi * pitch_override * t)
        sine_wave /= (np.max(np.abs(sine_wave))) 
        return sine_wave
    
    def gen_saw_wave(self, pitch_override=0):
        """Generate Saw Wave"""
        if not pitch_override:
            pitch_override = self.frequency
        t = np.linspace(0, self.duration, self.num_samples, endpoint=False)
        sawtooth_wave = 2 * (pitch_override * t - np.floor(self.frequency * t + 0.5))
        sawtooth_wave /= np.max(np.abs(sawtooth_wave))
        return sawtooth_wave
    
    def gen_rev_saw_wave(self, pitch_override=0):
        """Generate reverse saw wave"""

        rev_saw_wave = self.gen_saw_wave(pitch_override) * -1 

    def gen_square_wave(self):
        """Generate Sine Wave"""
        t = np.linspace(0, self.duration, self.num_samples, endpoint=False)
        square_wave = np.sign(np.sin(2 * np.pi * self.frequency * t))
        return square_wave
    
    def gen_tri_wave(self):
        """
        Generate a triangle wave audio signal.

        Parameters:
            freq (float): Frequency of the triangle wave in Hz.
            duration (float): Duration of the waveform in seconds.
            sample_rate (int): Sampling rate of the waveform (samples per second).

        Returns:
            ndarray: Generated triangle wave audio signal.
        """
        t = np.linspace(0, self.duration, self.num_samples, endpoint=False)  # Time array

        # Calculate angular frequency in radians
        angular_freq = 2 * np.pi * self.frequency

        # Generate triangle wave using modulo operation
        triangle_wave = 2 * np.abs((t * angular_freq / (2*np.pi) % 1) - 0.5) - 1

        return triangle_wave

    def gen_white_noise(self):
        """
        Generate white noise audio signal.

        Parameters:
            duration (float): Duration of the noise signal in seconds.
            sample_rate (int): Sampling rate of the noise signal (samples per second).

        Returns:
            ndarray: Generated white noise audio signal.
        """
        return np.random.normal(scale=1, size=self.num_samples)

    def gen_pink_noise(self):
        """
        Generate pink noise audio signal (1/f noise).

        Parameters:
            duration (float): Duration of the noise signal in seconds.
            sample_rate (int): Sampling rate of the noise signal (samples per second).

        Returns:
            ndarray: Generated pink noise audio signal.
        """
        pink_noise = np.random.randn(self.num_samples)

        # Apply pink noise filter (1/f)
        b = np.array([0.02109238, -0.02109238, 0.00682693, -0.00717957])
        a = np.array([1, -2.14952515, 1.46453786, -0.38151554])

        # Filter the noise signal
        pink_noise = np.convolve(pink_noise, b, mode='same')
        pink_noise = np.convolve(pink_noise, a, mode='same')

        return pink_noise

    def gen_brown_noise(self):
        """
        Generate brown noise audio signal.

        Parameters:
            duration (float): Duration of the noise signal in seconds.
            sample_rate (int): Sampling rate of the noise signal (samples per second).

        Returns:
            ndarray: Generated brown noise audio signal.
        """

        brown_noise = np.random.randn(self.num_samples).cumsum()

        # Normalize brown noise to stay within -1 to 1
        brown_noise /= np.max(np.abs(brown_noise))

        return brown_noise

    def gen_blue_noise(self):
        """
        Generate blue noise (also known as azure noise) audio signal.

        Parameters:
            duration (float): Duration of the noise signal in seconds.
            sample_rate (int): Sampling rate of the noise signal (samples per second).

        Returns:
            ndarray: Generated blue noise audio signal.
        """
        # Generate white noise (random Gaussian noise)
        white_noise = self.gen_white_noise()
        # white_noise = np.random.randn(self.num_samples)

        # Design blue noise filter coefficients (3 dB per octave increase)
        b = np.array([1, -2, 1])  # Numerator coefficients
        a = np.array([1, -1.999])  # Denominator coefficients

        # Apply blue noise filter to white noise
        blue_noise = lfilter(b, a, white_noise)

        # Normalize blue noise to stay within -1 to 1
        blue_noise /= np.max(np.abs(blue_noise))

        return blue_noise

    # def gen_simulated_drum_head_sound(Lx, Ly, Nx, Ny, T, dt, c, damping_coeff):
    #     """
    #     Simulate a 2D waveguide model for a drum head with damping using finite difference method.

    #     Parameters:
    #         Lx (float): Length of the drum head in the x-direction (meters).
    #         Ly (float): Length of the drum head in the y-direction (meters).
    #         Nx (int): Number of grid points in the x-direction.
    #         Ny (int): Number of grid points in the y-direction.
    #         T (float): Total simulation time (seconds).
    #         dt (float): Time step size (seconds).
    #         c (float): Wave speed (meters/second).
    #         damping_coeff (float): Damping coefficient (0.0 for no damping, higher values for stronger damping).

    #     Returns:
    #         ndarray: Displacement field of the drum head over time (shape: (num_steps, Ny, Nx)).
    #     """
    #     # Initialize grid
    #     x = np.linspace(0, Lx, Nx)
    #     y = np.linspace(0, Ly, Ny)
    #     X, Y = np.meshgrid(x, y)

    #     # Initialize displacement field
    #     u = np.zeros((Ny, Nx))   # Current displacement field
    #     u_prev = np.zeros_like(u)  # Previous displacement field

    #     # Function to apply boundary conditions (fixed edges)
    #     def apply_boundary_conditions(u):
    #         u[:, 0] = 0.0   # Left boundary (fixed)
    #         u[:, -1] = 0.0  # Right boundary (fixed)
    #         u[0, :] = 0.0   # Bottom boundary (fixed)
    #         u[-1, :] = 0.0  # Top boundary (fixed)

    #     # Simulation loop
    #     num_steps = int(T / dt)
    #     displacement_history = []

    #     for step in range(num_steps):
    #         # Apply boundary conditions
    #         apply_boundary_conditions(u)

    #         # Update displacement field using finite difference method (wave equation with damping)
    #         u_next = 2 * u - u_prev + (c**2 * dt**2) * (
    #             np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
    #             np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 4 * u
    #         ) - 2 * damping_coeff * dt * (u - u_prev)

    #         # Store current displacement field
    #         displacement_history.append(u.copy())

    #         # Update displacement fields for next time step
    #         u_prev = u
    #         u = u_next

    #     # Convert displacement history to numpy array
    #     displacement_history = np.array(displacement_history)

    #     return displacement_history

    #     # Example usage:
    #     if __name__ == '__main__':
    #         # Simulation parameters
    #         Lx = 1.0       # Length of the drum head in the x-direction (meters)
    #         Ly = 1.0       # Length of the drum head in the y-direction (meters)
    #         Nx = 50        # Number of grid points in the x-direction
    #         Ny = 50        # Number of grid points in the y-direction
    #         T = 2.0        # Total simulation time (seconds)
    #         dt = 0.0001    # Time step size (seconds)
    #         c = 100.0      # Wave speed (meters/second)
    #         damping_coeff = 0.05  # Damping coefficient (adjust to control decay, higher values for stronger damping)

    #         # Simulate drum head with damping
    #         displacement_history = simulate_drum_head(Lx, Ly, Nx, Ny, T, dt, c, damping_coeff)

    # def apply_dynamic_bandpass_filter(self):
    #     """Apply a dynamic bandpass filter with variable cutoff frequency and resonance (Q-factor)."""
    #     # Calculate center frequency modulation based on velocity and intensity
    #     max_frequency_change = 2000  # Maximum frequency change in Hz
        
    #     # Calculate center frequency modulation based on velocity (0 to 1) and intensity (-1 to 1)
    #     center_frequency_modulation = max_frequency_change * self.velocity * self.intensity
        
    #     # Define center frequency for the bandpass filter
    #     center_frequency = 3000 + center_frequency_modulation  # Base center frequency of 3000 Hz
        
    #     # Calculate bandwidth modulation based on resonance
    #     max_bandwidth = 2000  # Maximum bandwidth change in Hz
        
    #     # Calculate bandwidth based on resonance (higher Q-factor for narrower bandwidth)
    #     bandwidth = max_bandwidth * self.filter_resonanceresonance
        
    #     # Calculate lower and upper cutoff frequencies for the bandpass filter
    #     lower_cutoff = center_frequency - (bandwidth / 2)
    #     upper_cutoff = center_frequency + (bandwidth / 2)
        
    #     # Design a bandpass Butterworth filter with specified cutoff frequencies and order
    #     order = 4  # Filter order
    #     b_bandpass, a_bandpass = butter(order, [lower_cutoff, upper_cutoff], btype='bandpass', fs=self.sample_rate, output='ba')
        
    #     # Apply the bandpass filter to the audio signal
    #     filtered_signal = lfilter(b_bandpass, a_bandpass, self.source)
        
    #     return filtered_signal

    # def apply_dynamic_lowpass_filter(audio_signal, sample_rate, velocity, intensity, resonance):
    #     """Apply a dynamic low-pass filter with variable cutoff frequency and resonance (Q-factor)."""
    #     # Calculate cutoff frequency modulation based on velocity and intensity
    #     max_frequency_change = 2000  # Maximum frequency change in Hz
        
    #     # Calculate cutoff frequency modulation based on velocity (0 to 1) and intensity (-1 to 1)
    #     cutoff_modulation = max_frequency_change * velocity * intensity
        
    #     # Define cutoff frequency for the low-pass filter
    #     cutoff_frequency = 5000 + cutoff_modulation  # Base cutoff frequency of 5000 Hz
        
    #     # Calculate Q-factor (resonance) for the low-pass filter
    #     Q_factor = resonance  # Adjust resonance (higher Q_factor for more resonance)
        
    #     # Design a low-pass Butterworth filter with specified cutoff and Q-factor
    #     b_low, a_low = butter(4, cutoff_frequency, btype='low', fs=sample_rate, output='ba')
        
    #     # Apply the low-pass filter to the audio signal
    #     filtered_signal = lfilter(b_low, a_low, audio_signal)
        
    #     return filtered_signal

    # def apply_low_pass_filter(signal, cutoff_frequency, sample_rate):
    #     """Apply a low-pass Butterworth filter to the signal."""
    #     nyquist_freq = 0.5 * sample_rate
    #     normal_cutoff = cutoff_frequency / nyquist_freq
    #     b, a = butter(4, normal_cutoff, btype='low', analog=False)
    #     filtered_signal = lfilter(b, a, signal)
    #     filtered_signal /= np.max(np.abs(filtered_signal))
    #     return filtered_signal

    # def apply_high_pass_filter(self):
    #     """Apply a high-pass Butterworth filter to the signal."""
    #     nyquist_freq = 0.5 * self.sample_rate
    #     normal_cutoff = self.filter_cutoff / nyquist_freq
    #     b, a = butter(4, normal_cutoff, btype='high', analog=False)
    #     filtered_signal = lfilter(b, a, self.source)
    #     filtered_signal /= np.max(np.abs(filtered_signal))
    #     return filtered_signal

    # def gen_log_decay(self):
    #     """Generate a logarithmic decay envelope."""
    #     t = np.linspace(0, 1, self.duration)
    #     envelope = np.exp(-5 * (1 - t) / self.attack) * np.exp(-5 * t / self.decay)
    #     self.envelope

    # def gen_linear_decay(self):
    #     """Generate a linear decay envelope."""
    #     t = np.linspace(0, 1, self.duration)
    #     envelope = 1 - t * (1 - np.exp(-5 / self.decay))
    #     envelope[t < self.attack] *= t[t < self.attack] / self.attack
    #     return envelope

    # def gen_double_peak(self):
    #     """Generate a double peak envelope."""
    #     t = np.linspace(0, 1, self.duration)
    #     envelope = np.exp(-5 * (t - 0.5)**2 / (0.1 * self.attack**2))
    #     envelope *= np.exp(-5 * t / self.decay)
    #     return envelope

    # def modulate_amplitude(self):
    #     """Generate an amplitude-modulated signal (AM) using a modulation envelope."""
    #     t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), endpoint=False)
        
    #     # Carrier signal (sine wave)
    #     carrier_signal = np.sin(2 * np.pi * self.source * t)
        
    #     # Apply amplitude modulation (AM) with the modulation envelope
    #     am_signal = carrier_signal * self.modulation_env
        
    #     return am_signal

    # def apply_envelope(wave, attack, decay):
    #     # Apply a simple exponential decay envelope
    #     # attack_samples = int(sample_rate * attack_time)
    #     # attack_ramp = np.linspace(0, 1, attack_samples)
    #     # wave[:attack_samples] *= attack_ramp
    #     # decay = np.exp(-t / decay_time)
    #     # wave *= decay

    # # Amplitude modulation (AM) parameters

    #     modulation_type = 'log_decay'  # Choose modulation type: 'log_decay', 'linear_decay', 'double_peak'
    #     attack_time = 0.1       # Attack time (seconds)
    #     decay_time = 0.5        # Decay time (seconds)

    #     # Generate modulation envelope based on selected type
    #     length = int(sample_rate * duration)
    #     if modulation_type == 'log_decay':
    #         modulation_env = gen_log_decay(length, attack_time, decay_time)
    #     elif modulation_type == 'linear_decay':
    #         modulation_env = gen_linear_decay(length, attack_time, decay_time)
    #     elif modulation_type == 'double_peak':
    #         modulation_env = generate_double_peak(length, attack_time, decay_time)
    #     else:
    #         raise ValueError("Invalid modulation type. Choose from 'log_decay', 'linear_decay', or 'double_peak'.")

    #     # Generate amplitude-modulated (AM) signal using the selected modulation envelope
    #     am_signal = amplitude_modulation(carrier_freq, modulation_env)

    #     # Normalize the AM signal to the range [-1, 1]
    #     am_signal /= np.max(np.abs(am_signal))


    # def generate_log_decay_mod(length, attack_time, decay_time):
    #     """Generate a logarithmic decay envelope."""
    #     t = np.linspace(0, 1, length)
    #     envelope = np.exp(-5 * (1 - t) / attack_time) * np.exp(-5 * t / decay_time)
    #     return envelope

    # def generate_sine_wave(length, frequency):
    #     """Generate a sine wave."""
    #     t = np.linspace(0, duration, length, endpoint=False)
    #     waveform = np.sin(2 * np.pi * frequency * t)
    #     return waveform

    # def generate_noise(length):
    #     """Generate white noise."""
    #     noise = np.random.uniform(low=-1.0, high=1.0, size=length)
    #     return noise

    # def frequency_modulation(carrier_freq, modulation_type, mod_freq, mod_amount):
    #     """Generate a frequency-modulated signal (FM)."""
    #     t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        
    #     # Modulation waveform generation based on modulation type
    #     if modulation_type == 'log_decay':
    #         modulation_waveform = generate_log_decay(len(t), attack_time=0.1, decay_time=0.5)
    #     elif modulation_type == 'sine':
    #         modulation_waveform = generate_sine_wave(len(t), frequency=mod_freq)
    #     elif modulation_type == 'noise':
    #         modulation_waveform = generate_noise(len(t))
    #     else:
    #         raise ValueError("Invalid modulation type. Choose from 'log_decay', 'sine', or 'noise'.")
        
    #     # Carrier signal (sine wave)
    #     carrier_signal = np.sin(2 * np.pi * carrier_freq * t)
        
    #     # Frequency modulation (FM) by scaling the carrier frequency with the modulation waveform
    #     fm_signal = np.sin(2 * np.pi * (carrier_freq + mod_amount * modulation_waveform) * t)
        
    #     return fm_signal
    
    # def apply_bit_crushing(audio_signal, bit_depth):
    #     """Apply bit crushing effect to the audio signal."""
    #     # Calculate the number of quantization levels based on bit depth
    #     levels = 2 ** bit_depth
        
    #     # Quantize the audio signal
    #     quantized_signal = np.floor(audio_signal * levels) / levels
        
    #     return quantized_signal

    # def apply_sample_reduction(audio_signal, reduction_factor):
    #     """Apply sample reduction effect to the audio signal."""
    #     # Reduce the sample rate by a specified reduction factor
    #     reduced_signal = audio_signal[::reduction_factor]
        
    #     return reduced_signal

    # def apply_spring_reverb(audio_signal, decay_factor, delay_length):
    #     """Apply spring reverb effect to the audio signal."""
    #     # Create a feedback delay line with a low-pass filter
    #     feedback_delay = np.zeros(delay_length)
    #     output_signal = np.zeros_like(audio_signal)
        
    #     for i in range(len(audio_signal)):
    #         # Calculate output sample (comb filter with feedback)
    #         output_sample = audio_signal[i] + decay_factor * feedback_delay[0]
            
    #         # Update delay line (feedback delay)
    #         feedback_delay = np.roll(feedback_delay, 1)
    #         feedback_delay[0] = output_sample
            
    #         # Store output sample
    #         output_signal[i] = output_sample
        
    #     return output_signal

    # def apply_natural_reverb(audio_signal, decay_factor, delay_length):
    #     """Apply natural reverb effect to the audio signal (simple comb filter)."""
    #     # Create a feedback delay line (comb filter)
    #     feedback_delay = np.zeros(delay_length)
    #     output_signal = np.zeros_like(audio_signal)
        
    #     for i in range(len(audio_signal)):
    #         # Calculate output sample (comb filter with feedback)
    #         output_sample = audio_signal[i] + decay_factor * feedback_delay[0]
            
    #         # Update delay line (feedback delay)
    #         feedback_delay = np.roll(feedback_delay, 1)
    #         feedback_delay[0] = output_sample
            
    #         # Store output sample
    #         output_signal[i] = output_sample
        
    #     return output_signal

    # def apply_delay(audio_signal, delay_time, feedback_gain):
    #     """Apply delay effect to the audio signal."""
    #     # Calculate delay length in samples
    #     delay_samples = int(delay_time * sample_rate)
        
    #     # Create a delay line with feedback
    #     delay_line = np.zeros(delay_samples)
    #     output_signal = np.zeros_like(audio_signal)
        
    #     for i in range(len(audio_signal)):
    #         # Calculate output sample (delay with feedback)
    #         output_sample = audio_signal[i] + feedback_gain * delay_line[0]
            
    #         # Update delay line (delay buffer)
    #         delay_line = np.roll(delay_line, 1)
    #         delay_line[0] = output_sample
            
    #         # Store output sample
    #         output_signal[i] = output_sample
        
    #     return output_signal

    # def apply_wave_folding(audio_signal, fold_factor):
    #     """Apply wave folding effect to the audio signal."""
    #     # Apply wave folding (fold the signal at specified fold factor)
    #     folded_signal = np.abs(np.mod(audio_signal, fold_factor) - fold_factor / 2)
        
    #     return folded_signal

    # def apply_analog_drive(audio_signal, drive_gain):
    #     """Apply analog-style drive effect to the audio signal."""
    #     # Apply soft clipping (tanh saturation) with drive gain
    #     driven_signal = np.tanh(drive_gain * audio_signal)
        
    #     return driven_signal

    # def apply_warm_saturation(audio_signal, saturation_amount):
    #     """Apply warm saturation effect to the audio signal."""
    #     # Apply soft clipping (tanh saturation) with adjustable amount
    #     saturated_signal = np.tanh(saturation_amount * audio_signal)
        
    #     return saturated_signal

    # def apply_pan(audio_signal, pan_position):
    #     """Apply panning effect to the audio signal (left-right balance)."""
    #     # Create stereo audio with specified pan position (-1 to 1, -1 = left, 1 = right)
    #     left_signal = np.sqrt(0.5) * (np.cos(pan_position * np.pi / 2) * audio_signal)
    #     right_signal = np.sqrt(0.5) * (np.sin(pan_position * np.pi / 2) * audio_signal)
        
    #     stereo_signal = np.vstack((left_signal, right_signal))
        
    #     return stereo_signal

    # def apply_clean_gain(audio_signal, gain_factor):
    #     """Apply clean gain (amplification) to the audio signal."""
    #     # Amplify the audio signal by a specified gain factor
    #     amplified_signal = gain_factor * audio_signal
        
    #     return amplified_signal

    # def waveguide_synthesis(self, tune, decay, body):
    #     """
    #     Apply waveguide synthesis to an input audio signal.

    #     Parameters:
    #     - self.source: Input audio signal as a numpy array
    #     - tune: tune control (frequency in Hz)
    #     - decay: Decay control (decay time in seconds)
    #     - body: body control (feedback amount)

    #     Returns:
    #     - numpy array containing the synthesized output signal
    #     """
    #     # Length of the delay line based on tune (adjust based on desired range)
    #     delay_length = int(self.sample_rate / tune)

    #     # Initialize delay line and output signal
    #     delay_line = np.zeros(delay_length)
    #     output_signal = np.zeros_like(self.source)

    #     # Process each sample in the input signal
    #     for i in range(len(self.source)):
    #         # Read from delay line (comb filter)
    #         output_sample = delay_line[0]

    #         # Update delay line with feedback (all-pass filter)
    #         delay_line[0] = self.source[i] + body * delay_line[0]

    #         # Move samples in delay line (implement tune control)
    #         delay_line = np.roll(delay_line, 1)

    #         # Apply decay envelope
    #         output_signal[i] = output_sample
    #         delay_line *= (1.0 - 1.0 / (decay * self.sample_rate))

    #     return output_signal

    # def apply_multi_channel_eq(audio_signal, sample_rate, band_gains, band_centers, band_qs):
    #     """
    #     Apply multi-channel equalization to the input audio signal using specified EQ parameters.

    #     Args:
    #         audio_signal (ndarray): Input audio signal (1D array).
    #         sample_rate (int): Sampling rate of the audio signal (in Hz).
    #         band_gains (list): List of gain values (in dB) for each EQ band.
    #         band_centers (list): List of center frequencies (in Hz) for each EQ band.
    #         band_qs (list): List of Q factors (bandwidth) for each EQ band.

    #     Returns:
    #         ndarray: Output audio signal after applying EQ.
    #     """
    #     num_bands = len(band_gains)
        
    #     # Initialize the combined filter coefficients
    #     sos = np.zeros((num_bands, 6))
        
    #     # Design peaking filters for each EQ band
    #     for i in range(num_bands):
    #         center_freq = band_centers[i]
    #         Q = band_qs[i]
    #         gain_db = band_gains[i]
            
    #         # Compute peaking filter coefficients (second-order section)
    #         sos[i, :], _ = iirpeak(center_freq, Q, sample_rate, gain_db=gain_db, output='sos')
        
    #     # Apply the EQ filters using cascaded second-order sections (SOS)
    #     output_signal = sosfilt(sos, audio_signal)
        
    #     return output_signal

    # def play_audio(audio):
    #     """Play the audio signal using sounddevice."""
    #     sd.play(audio, samplerate=sample_rate)
    #     sd.wait()

    def save_audio(self):
        """
        Save the audio signal to a WAV file.

        Parameters:
        - audio: numpy array containing the audio signal
        - filename: name of the output WAV file
        """
        # Scale audio to 16-bit integer range (-32768 to 32767)
        audio_int = (self.audio * 32767).astype(np.int16)
        
        # Save the audio to a WAV file
        wavfile.write(self.filename, self.sample_rate, audio_int)

