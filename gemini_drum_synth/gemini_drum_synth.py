import numpy as np

from dataclasses import dataclass

# import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import butter, lfilter, iirpeak, sosfilt, resample as sci_resample
from pathlib import Path
from pedalboard.io import AudioFile
from pedalboard import (
    Compressor,
    LadderFilter,
    Reverb,
    Delay,
    Chorus,
    Limiter,
    Bitcrush,
    Pedalboard,
    PitchShift,
    Resample,
    Distortion,
    Clipping,
)

# Sampling parameters
@dataclass
class Sound:
    sample_rate: int = 44100  # Sample rate (Hz)
    output_file_bitdepth: int = 16
    velocity: int = 90  # intensity of drum hit
    pan: int = 0
    level: float = 0.5

    def save_audio(self, filename):
        """
        Save the audio signal to a WAV file.

        Parameters:
        - audio: numpy array containing the audio signal
        - filename: name of the output WAV file
        """
        # Scale audio to 16-bit integer range (-32768 to 32767)
        audio_int = (self.audio * 32767).astype(np.int16)

        # Save the audio to a WAV file
        wavfile.write(self.filepath / f"{filename}.wav", self.sample_rate, audio_int)


@dataclass
class Layer:
    num_channels = 1
    bit_depth: int = 16
    sample_rate: int = 44100
    level: float = 0.5
    wave_guide_send: int = 0
    distortion_send: int = 0
    fx_bus_1_send: int = 0
    fx_bus_2_send: int = 0
    reverb_send: int = 0
    delay_send: int = 0
    filepath: Path = Path.home() / "drumsynth/samples"
    layer_audio: np.array = np.zeros(1)

    def pitch_change_semitones(self, audio_signal, semitones):
        """
        Change the pitch of an audio signal by a specified number of semitones.

        Parameters:
            audio_signal (ndarray): Input audio signal as a NumPy array.
            original_sr (int): Original sampling rate of the audio signal.
            semitones (float): Number of semitones to shift the pitch.
                Positive value for pitch increase (upwards), negative value for pitch decrease (downwards).

        Returns:
            ndarray: Audio signal with adjusted pitch (resampled to the target frequency).
        """
        # Calculate the frequency ratio based on the number of semitones
        frequency_ratio = 2 ** (semitones / 12)

        # Determine the target sample rate based on the frequency ratio
        target_sr = int(self.sample_rate * frequency_ratio)

        # Resample the audio signal to the target sample rate
        num_samples = int(len(audio_signal) * float(target_sr) / self.sample_rate)
        resampled_signal = sci_resample(audio_signal, num_samples)

        return resampled_signal

    # def play_audio(audio):
    #     """Play the audio signal using sounddevice."""
    #     sd.play(audio, samplerate=sample_rate)
    #     sd.wait()

    def normalize_audio(self, audio):
        """Normalize audio signal to between -1 and 1"""
        audio /= np.max(np.abs(audio))
        return audio

    def save_layer(self, filename):
        """
        Save the audio signal to a WAV file.

        Parameters:
        - audio: numpy array containing the audio signal
        - filename: name of the output WAV file
        """
        # Scale audio to 16-bit integer range (-32768 to 32767)
        audio_int = (self.layer_audio * 32767).astype(np.int16)

        # Save the audio to a WAV file
        wavfile.write(self.filepath / f"{filename}.wav", self.sample_rate, audio_int)


@dataclass
class SynthLayer(Layer):
    pitch: int = 0  # adjust pitch of  click sound in semitones (postive or negative)
    attack: float = 0.000001  # Duration of the attack of synthesized sound (seconds) to avoid divide by zero errors
    decay: float = 2.0  # Duration of the decay synthesized sound (seconds)

    def __post_init__(self):
        self.attack = self.attack if self.attack else 0.00001
        self.duration = self.attack + self.decay
        self.attack_samples = int(np.ceil(self.attack * self.sample_rate))
        self.decay_samples = int(self.decay * self.sample_rate)
        self.num_samples = int(self.attack_samples + self.decay_samples)
        self.env_t = np.linspace(0, 1, self.num_samples)
        self.att_t = np.linspace(0, 1, self.attack_samples)
        self.dec_t = np.linspace(0, 1, self.decay_samples)

    def gen_sine_wave(self, pitch_override=0):
        """
        Generate Sine Wave
        """
        if not pitch_override:
            pitch_override = self.frequency
        t = np.linspace(0, self.duration, self.num_samples, endpoint=False)
        sine_wave = np.sin(2 * np.pi * pitch_override * t)
        sine_wave /= np.max(np.abs(sine_wave))
        return sine_wave

    def gen_tri_wave(self, pitch_override=0):
        """
        Generate a triangle wave audio signal.

        Parameters:
            pitch_override (int): allow function to work with frequences that are not stored in the self.frequency property.

        Returns:
            ndarray: Generated triangle wave audio signal.
        """
        if not pitch_override:
            pitch_override = self.frequency
        t = np.linspace(
            0, self.duration, self.num_samples, endpoint=False
        )  # Time array

        # Calculate angular frequency in radians
        angular_freq = 2 * np.pi * pitch_override

        # Generate triangle wave using modulo operation
        triangle_wave = 2 * np.abs((t * angular_freq / (2 * np.pi) % 1) - 0.5) - 1

        return triangle_wave

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

    def gen_square_wave(self, pitch_override=0):
        """Generate Sine Wave

        Parameters:
            pitch_override (int): allow function to work with frequences that are not stored in the self.frequency property.

        Returns:
            ndarray: Generated triangle wave audio signal.
        """

        if not pitch_override:
            pitch_override = self.frequency
        t = np.linspace(0, self.duration, self.num_samples, endpoint=False)
        square_wave = np.sign(np.sin(2 * np.pi * pitch_override * t))
        return square_wave

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
        pink_noise = np.convolve(pink_noise, b, mode="same")
        pink_noise = np.convolve(pink_noise, a, mode="same")

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

    def gen_log_decay(self, degree=50):
        """Generate a logarithmic decay."""
        base = 0.95**degree
        log_decay = np.flip(np.logspace(1, 0, self.decay_samples, base=base))
        return (log_decay - np.min(log_decay)) / (np.max(log_decay) - np.min(log_decay))

    def gen_log_attack(self, degree=50):
        """Generate a logarithmic attack."""
        base = 0.95**degree
        log_attack = np.flip(np.logspace(1, 0, self.attack_samples, base=base))
        return (log_attack - np.min(log_attack)) / (
            np.max(log_attack) - np.min(log_attack)
        )

    def gen_lin_attack(self):
        """Generate a linear attack."""
        return np.linspace(0, 1, self.attack_samples)

    def gen_lin_decay(self):
        """Generate a linear decay."""
        return np.linspace(1, 0, self.decay_samples)

    def gen_lin_log_ad_env(self, degree=50):
        """Generate a linear attack / logrithmic decay envelope."""
        rise = self.gen_lin_attack()
        fall = self.gen_log_decay(degree=degree)
        envelope = np.concatenate([rise, fall])
        self.num_samples = rise.shape[0] + fall.shape[0]
        return envelope[: self.num_samples]

    def gen_log_log_ad_env(self, degree=50):
        rise = self.gen_log_attack(degree=degree)
        fall = self.gen_log_decay(degree=degree)
        envelope = np.concatenate([rise, fall])
        self.num_samples = rise.shape[0] + fall.shape[0]
        return envelope[: self.num_samples]

    def gen_lin_env(self, degree=50):
        """Generate a linear attack / logrithmic decay envelope."""
        rise = self.gen_lin_attack()
        fall = self.gen_lin_decay()
        envelope = np.concatenate([rise, fall])
        self.num_samples = rise.shape[0] + fall.shape[0]
        return envelope[: self.num_samples]

    def gen_double_peak_env(self, seperation=0.1, degree=50):
        """Generate a double peak envelope."""
        # t = np.linspace(0, 1, self.num_samples)
        shift_amt = int(self.num_samples * seperation)
        envelope = self.gen_lin_log_ad_env(degree=degree)
        zeros = np.zeros(shift_amt)
        ones = np.ones(int(self.num_samples - shift_amt))
        mask = np.concatenate((ones, zeros))
        reenvlope = np.roll(np.min((envelope, mask), axis=0), shift_amt)
        tail = envelope[self.num_samples - shift_amt :]
        remix = np.concatenate((np.max((envelope, reenvlope), axis=0), tail))
        self.num_samples = self.num_samples + shift_amt
        return remix


@dataclass
class ClickLayer(SynthLayer):
    """
    TODO: add level, sends, etc.
    Creates transient sounds alliveiating the need for frequency modulation to achieve this affect
    """

    click_type: str = 'N1'

    def __post_init__(self):
        super().__post_init__()
        self.attack = 0
        self.gen_click()

    def gen_click(self):
        """
        Generate a click sound waveform of the specified type.

        Parameters:
            click_type (str): Type of click sound ('simple', 'white_noise', 'impulse').
            duration (float): Duration of the click sound in seconds (default: 0.01 seconds).
            sample_rate (int): Sampling rate of the audio waveform (samples per second, default: 44100).

        Returns:
            ndarray: NumPy array containing the generated click sound waveform.
        """
        num_click_samples = int(self.duration * self.sample_rate)
        t = np.linspace(0, self.duration, num_click_samples, endpoint=False)
        click_envelope = np.exp(-5 * t)  # Exponential decay envelope
        # print(num_click_samples)

        if self.click_type == 'S1':
            # Generate a simple click (cosine wave envelope)
            self.layer_audio = np.cos(2 * np.pi * 1000 * t) * click_envelope

        elif self.click_type == 'N1':
            # Generate a burst of white noise
            self.layer_audio = np.random.randn(num_click_samples)

        elif self.click_type == 'N2':
            # Generate a burst of white noise with short envelop
            self.layer_audio = np.random.randn(num_click_samples) * click_envelope

        elif self.click_type == 'I1':
            # Generate an impulse (single-sample spike)
            self.layer_audio = np.zeros(num_click_samples)
            self.layer_audio[0] = 1.0

        elif self.click_type == 'M1':
            # Generate an metallic click (High-frequency sinusoidal component)
            high_freq_component = np.sin(2 * np.pi * 3000 * t)
            # Combine envelope with high-frequency component
            self.layer_audio = click_envelope * high_freq_component

        elif self.click_type == 'T1':
            # Envelope shaping for thud click
            low_freq_component = np.sin(
                2 * np.pi * 200 * t
            )  # Low-frequency sinusoidal component

            # Combine envelope with low-frequency component
            self.layer_audio = click_envelope * low_freq_component

        else:
            raise ValueError(
                f"Invalid click_type '{self.click_type}'. Choose from 'simple', 'white_noise', or 'impulse'."
            )

        if self.pitch:
            self.layer_audio = self.pitch_change_semitones(self.layer_audio, self.pitch)


@dataclass
class NoiseLayer(SynthLayer):
    """Drum Sound Layer based on the Synthesis in Nord Drum 3P"""

    # noise_type: str = 'white'
    filter_type: str = 'L2'  # low pass 4 pole
    resonance: int = 0  # max 20
    freq: int = 200  # cutoff frequency in Hz
    dynamic_filter: int = 0  # plus or minus 9
    # decay: int = 0  # max 50
    decay_type: str = "E"  #
    # drive: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        self.attack = 0
        # print('does this print')
        # print(self.num_samples)
        # self.duration = self.attack + self.decay
        filter_translate = {
            'L1': LadderFilter.LPF12,
            'L2': LadderFilter.LPF24,
            'H1': LadderFilter.HPF12,
            'H2': LadderFilter.HPF24,
            'B1': LadderFilter.BPF12,
            'B2': LadderFilter.BPF24,
        }
        decay_translate = {
            'E': self.gen_log_decay,
            'L': self.gen_linear_decay,
            'G': self.gen_gate_decay,
            'P': self.gen_punch_decay,
        }
        self.filter_mode = filter_translate.get(self.filter_type, 'L2')
        self.noise_decay_envelope = decay_translate.get(self.decay_type, 'E')()
        self.layer_audio = self.gen_white_noise()
        self.filter_noise()
        self.apply_noise_envelope()

    def filter_noise(self):
        """
        Generate a noise sound waveform of the specified type.

        Parameters:

        Returns:
            ndarray: NumPy array containing the generated click sound waveform.
        """
        board = Pedalboard([LadderFilter()])[0]
        board.mode = self.filter_mode
        board.cutoff_hz = self.freq
        board.drive = 1
        board.resonance = self.resonance

        if self.dynamic_filter:
            # FOR DYNAMIC FILTER
            # TODO CREATE CURVE AND ADAPT IT TO THE WEIRD LOW_PASS NEGATIVE HIGH PASS POSTIVE LOGIC. TEST WITH SOUNDS
            step_size_in_samples = 100
            output = []

            for i in range(0, self.num_samples, step_size_in_samples):
                #     chunk = af.read(step_size_in_samples)
                # if self.filter_type.startswith('L'):
                #     board.cutoff_hz = self.freq
                output.append(
                    board.process(
                        input_array=self.layer_audio[i : i + step_size_in_samples],
                        sample_rate=self.sample_rate,
                        reset=False,
                    )
                )
            self.layer_audio = np.concatenate(output, axis=0)
        else:
            self.layer_audio = board.process(
                input_array=self.layer_audio, sample_rate=self.sample_rate
            )

        #     # Set the reverb's "wet" parameter to be equal to the
        #     # percentage through the track (i.e.: make a ramp from 0% to 100%)
        # percentage_through_track = i / af.frames
        # board[0].cutoff_hz = percentage_through_track

        #         # Update our progress bar with the number of samples received:
        #         pbar.update(chunk.shape[1])

        #         # Process this chunk of audio, setting `reset` to `False`
        #         # to ensure that reverb tails aren't cut off
        #         output = board.process(chunk, af.samplerate, reset=False)
        #         o.write(output)

        # self.layer_audio = board.process(
        #     input_array=self.layer_audio, sample_rate=self.sample_rate
        # )

    def apply_noise_envelope(self):
        self.layer_audio = self.normalize_audio(
            self.noise_decay_envelope * self.layer_audio
        )


@dataclass
class ToneLayer(SynthLayer):

    wave: str = "A1"  # default sine wave
    second: int = 50  # second parameter, spectra if applicable
    third: int = 0  # third parameter of the wave also filter frequency
    dynamic_filter: int = 0
    # decay: int = 20
    decay_type: str = "dyn"
    dynamic_decay: int = 0
    bend: int = 0
    bend_time: int = 0
    pitch: int = 60  # default middle c

    def __post_init__(self):
        self.attack = 0

    def choose_spectra(self):
        self.spectra_options = {
            "A1": self.gen_sine_wave,  # analog-style sine wave
            "A2": self.gen_tri_wave,  # analog-style triangle wave
            "A3": self.gen_saw_wave,  # analog-style triangle wave
            "A4": self.gen_square_wave,  # analog-style square wave
            "A5": self.hp_square_wave,  # high pass filtered square wave
            # 'A6':self.gen_pulse_wave, #analog-style pulse wave
        }

    def gen_simulated_drum_head_sound(self, Lx, Ly, Nx, Ny, T, dt, c, damping_coeff):
        """
        Simulate a 2D waveguide model for a drum head with damping using finite difference method.

        Parameters:
            Lx (float): Length of the drum head in the x-direction (meters).
            Ly (float): Length of the drum head in the y-direction (meters).
            Nx (int): Number of grid points in the x-direction.
            Ny (int): Number of grid points in the y-direction.
            T (float): Total simulation time (seconds).
            dt (float): Time step size (seconds).
            c (float): Wave speed (meters/second).
            damping_coeff (float): Damping coefficient (0.0 for no damping, higher values for stronger damping).

        Returns:
            ndarray: Displacement field of the drum head over time (shape: (num_steps, Ny, Nx)).
        """
        # Initialize grid
        x = np.linspace(0, Lx, Nx)
        y = np.linspace(0, Ly, Ny)
        X, Y = np.meshgrid(x, y)

        # Initialize displacement field
        u = np.zeros((Ny, Nx))  # Current displacement field
        u_prev = np.zeros_like(u)  # Previous displacement field

        # Function to apply boundary conditions (fixed edges)
        def apply_boundary_conditions(u):
            u[:, 0] = 0.0  # Left boundary (fixed)
            u[:, -1] = 0.0  # Right boundary (fixed)
            u[0, :] = 0.0  # Bottom boundary (fixed)
            u[-1, :] = 0.0  # Top boundary (fixed)

        # Simulation loop
        num_steps = int(T / dt)
        displacement_history = []

        for step in range(num_steps):
            # Apply boundary conditions
            apply_boundary_conditions(u)

            # Update displacement field using finite difference method (wave equation with damping)
            u_next = (
                2 * u
                - u_prev
                + (c**2 * dt**2)
                * (
                    np.roll(u, 1, axis=0)
                    + np.roll(u, -1, axis=0)
                    + np.roll(u, 1, axis=1)
                    + np.roll(u, -1, axis=1)
                    - 4 * u
                )
                - 2 * damping_coeff * dt * (u - u_prev)
            )

            # Store current displacement field
            displacement_history.append(u.copy())

            # Update displacement fields for next time step
            u_prev = u
            u = u_next

        # Convert displacement history to numpy array
        displacement_history = np.array(displacement_history)

        return displacement_history

        # # Example usage:
        # if __name__ == "__main__":
        #     # Simulation parameters
        #     Lx = 1.0  # Length of the drum head in the x-direction (meters)
        #     Ly = 1.0  # Length of the drum head in the y-direction (meters)
        #     Nx = 50  # Number of grid points in the x-direction
        #     Ny = 50  # Number of grid points in the y-direction
        #     T = 2.0  # Total simulation time (seconds)
        #     dt = 0.0001  # Time step size (seconds)
        #     c = 100.0  # Wave speed (meters/second)
        #     damping_coeff = 0.05  # Damping coefficient (adjust to control decay, higher values for stronger damping)

        #     # Simulate drum head with damping
        #     displacement_history = simulate_drum_head(
        #         Lx, Ly, Nx, Ny, T, dt, c, damping_coeff
        #     )

    def frequency_modulation(self, carrier_freq, modulation_type, mod_freq, mod_amount):
        """Generate a frequency-modulated signal (FM)."""
        t = np.linspace(
            0, self.duration, int(self.sample_rate * self.duration), endpoint=False
        )

        # Modulation waveform generation based on modulation type
        if modulation_type == 'log_decay':
            modulation_waveform = self.gen_log_decay(
                len(t), attack_time=0.1, decay_time=0.5
            )
        elif modulation_type == 'sine':
            modulation_waveform = self.gen_sine_wave()
        elif modulation_type == 'noise':
            modulation_waveform = self.gen_white_noise()
        else:
            raise ValueError(
                "Invalid modulation type. Choose from 'log_decay', 'sine', or 'noise'."
            )

        # Carrier signal (sine wave)
        # carrier_signal = np.sin(2 * np.pi * carrier_freq * t)

        # Frequency modulation (FM) by scaling the carrier frequency with the modulation waveform
        fm_signal = np.sin(
            2 * np.pi * (carrier_freq + mod_amount * modulation_waveform) * t
        )

        return fm_signal


@dataclass
class GenericLayer(SynthLayer):
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

    src_type: str = "square"
    mod_type: str = "exp"
    env_type: str = "linear"
    frequency: int = 440
    detune: int = 10
    mod_amount: float = 1.0
    mod_rate: int = 220
    dynamic_filter: float = 0

    def __post_init__(self):
        self.duration = self.attack + self.decay
        self.num_samples = int(self.duration * self.sample_rate)
        self.noise_types = {
            "white": self.gen_white_noise,
            "brown": self.gen_brown_noise,
            "pink": self.gen_pink_noise,
            "blue": self.gen_blue_noise,
        }
        self.filter_types = {
            "low": self.apply_low_pass_filter,
            "high": self.apply_high_pass_filter,
            "band": self.apply_band_pass_filter,
        }
        self.osc_func = {
            "sine": self.gen_sine_wave,
            "saw": self.gen_saw_wave,
            "rev_saw": self.gen_rev_saw_wave,
            "square": self.gen_square_wave,
            "tri": self.gen_tri_wave,
            "white_noise": self.gen_white_noise,
            "hp_white_noise": self.gen_highpass_white_noise,
            "lp_white_noise": self.gen_lowpass_white_noise,
            "bp_white_noise": self.gen_bandpass_white_noise,
            "blue_noise": self.gen_blue_noise,
            "hp_blue_noise": self.gen_highpass_blue_noise,
            "lp_blue_noise": self.gen_lowpass_blue_noise,
            "bp_blue_noise": self.gen_bandpass_blue_noise,
            "brown_noise": self.gen_brown_noise,
            "hp_brown_noise": self.gen_highpass_brown_noise,
            "lp_brown_noise": self.gen_lowpass_brown_noise,
            "bp_brown_noise": self.gen_bandpass_brown_noise,
            "pink_noise": self.gen_pink_noise,
            "hp_pink_noise": self.gen_highpass_pink_noise,
            "lp_pink_noise": self.gen_lowpass_pink_noise,
            "bp_pink_noise": self.gen_bandpass_pink_noise,
        }
        self.env_type = {
            "log": self.gen_log_decay,
            "linear": self.gen_linear_decay,
            "double_peak": self.gen_double_peak,
        }
        self.apply_wave()
        # if self.b is None:
        #     self.b = 'Bravo'
        # if self.c is None:
        #     self.c = 'Charlie'

    def apply_wave(self):
        """Run code to generate source audio that will be used as basis for drum sounds"""
        self.layer_audio = self.osc_func[self.src_type]()

    def gen_filtered_noise(self, noise_type, filter_type, cutoff):
        noise = self.noise_types.get(noise_type, "white")()
        return self.allpass_filter(signal=noise)
        # return self.filter_types.get(filter_type)(
        #     signal=self.noise_types.get(noise_type)(), cutoff_frequency=cutoff
        # )

    def gen_highpass_white_noise(self):
        return self.gen_filtered_noise(
            noise_type="white", filter_type="high", cutoff=self.frequency
        )

    def gen_highpass_blue_noise(self):
        return self.gen_filtered_noise(
            noise_type="blue", filter_type="high", cutoff=self.frequency
        )

    def gen_highpass_pink_noise(self):
        return self.gen_filtered_noise(
            noise_type="pink", filter_type="high", cutoff=self.frequency
        )

    def gen_highpass_brown_noise(self):
        return self.gen_filtered_noise(
            noise_type="brown", filter_type="high", cutoff=self.frequency
        )

    def gen_lowpass_white_noise(self):
        return self.gen_filtered_noise(
            noise_type="white", filter_type="low", cutoff=self.frequency
        )

    def gen_lowpass_blue_noise(self):
        return self.gen_filtered_noise(
            noise_type="blue", filter_type="low", cutoff=self.frequency
        )

    def gen_lowpass_pink_noise(self):
        return self.gen_filtered_noise(
            noise_type="pink", filter_type="low", cutoff=self.frequency
        )

    def gen_lowpass_brown_noise(self):
        return self.gen_filtered_noise(
            noise_type="brown", filter_type="low", cutoff=self.frequency
        )

    def gen_bandpass_white_noise(self):
        return self.gen_filtered_noise(
            noise_type="white", filter_type="band", cutoff=self.frequency
        )

    def gen_bandpass_blue_noise(self):
        return self.gen_filtered_noise(
            noise_type="blue", filter_type="band", cutoff=self.frequency
        )

    def gen_bandpass_pink_noise(self):
        return self.gen_filtered_noise(
            noise_type="pink", filter_type="band", cutoff=self.frequency
        )

    def gen_bandpass_brown_noise(self):
        return self.gen_filtered_noise(
            noise_type="brown", filter_type="band", cutoff=self.frequency
        )

    def apply_low_pass_filter(self, signal, cutoff_frequency):
        """Apply a low-pass Butterworth filter to the signal."""
        nyquist_freq = 0.5 * self.sample_rate
        normal_cutoff = cutoff_frequency / nyquist_freq
        # b, a = butter(4, normal_cutoff, btype='low', analog=False)
        b, a = butter(4, normal_cutoff, btype="low", analog=True)
        filtered_signal = lfilter(b, a, signal)
        filtered_signal /= np.max(np.abs(filtered_signal))
        return filtered_signal

    def allpass_filter(self, signal, cutoff=440, filter_type="low"):
        cut_off_freq_mod = np.geomspace(20000, 20, signal.shape[0])

        allpass_signal = np.zeros_like(signal)

        dn_1 = 0

        for n in range(signal.shape[0]):
            break_frequency = cut_off_freq_mod[n]

            tan = np.tan(np.pi * break_frequency / self.sample_rate)

            a1 = (tan - 1) / (tan + 1)

            allpass_signal[n] = a1 + signal[n] + dn_1

            dn_1 = signal[n] - a1 * allpass_signal[n]

        if filter_type == "high":
            allpass_signal *= -1

        filtered_signal = (signal + allpass_signal) * 0.5

        return filtered_signal


# @dataclass
# class LayerEffects:

#     def apply_dynamic_bandpass_filter(self):
#         """Apply a dynamic bandpass filter with variable cutoff frequency and resonance (Q-factor)."""
#         # Calculate center frequency modulation based on velocity and intensity
#         max_frequency_change = 2000  # Maximum frequency change in Hz

#         # Calculate center frequency modulation based on velocity (0 to 1) and intensity (-1 to 1)
#         center_frequency_modulation = max_frequency_change * self.velocity * self.intensity

#         # Define center frequency for the bandpass filter
#         center_frequency = 3000 + center_frequency_modulation  # Base center frequency of 3000 Hz

#         # Calculate bandwidth modulation based on resonance
#         max_bandwidth = 2000  # Maximum bandwidth change in Hz

#         # Calculate bandwidth based on resonance (higher Q-factor for narrower bandwidth)
#         bandwidth = max_bandwidth * self.filter_resonanceresonance

#         # Calculate lower and upper cutoff frequencies for the bandpass filter
#         lower_cutoff = center_frequency - (bandwidth / 2)
#         upper_cutoff = center_frequency + (bandwidth / 2)

#         # Design a bandpass Butterworth filter with specified cutoff frequencies and order
#         order = 4  # Filter order
#         b_bandpass, a_bandpass = butter(order, [lower_cutoff, upper_cutoff], btype='bandpass', fs=self.sample_rate, output='ba')

#         # Apply the bandpass filter to the audio signal
#         filtered_signal = lfilter(b_bandpass, a_bandpass, self.source)

#         return filtered_signal

#     def apply_dynamic_lowpass_filter(audio_signal, sample_rate, velocity, intensity, resonance):
#         """Apply a dynamic low-pass filter with variable cutoff frequency and resonance (Q-factor)."""
#         # Calculate cutoff frequency modulation based on velocity and intensity
#         max_frequency_change = 2000  # Maximum frequency change in Hz

#         # Calculate cutoff frequency modulation based on velocity (0 to 1) and intensity (-1 to 1)
#         cutoff_modulation = max_frequency_change * velocity * intensity

#         # Define cutoff frequency for the low-pass filter
#         cutoff_frequency = 5000 + cutoff_modulation  # Base cutoff frequency of 5000 Hz

#         # Calculate Q-factor (resonance) for the low-pass filter
#         Q_factor = resonance  # Adjust resonance (higher Q_factor for more resonance)

#         # Design a low-pass Butterworth filter with specified cutoff and Q-factor
#         b_low, a_low = butter(4, cutoff_frequency, btype='low', fs=sample_rate, output='ba')

#         # Apply the low-pass filter to the audio signal
#         filtered_signal = lfilter(b_low, a_low, audio_signal)

#         return filtered_signal

#     def apply_dynamic_highpass_filter(audio_signal, sample_rate, velocity, intensity):
#         """Apply a dynamic high-pass filter with variable cutoff frequency."""
#         # Calculate cutoff frequency modulation based on velocity and intensity
#         max_frequency_change = 1000  # Maximum frequency change in Hz

#         # Calculate cutoff frequency modulation based on velocity (0 to 1) and intensity (-1 to 1)
#         cutoff_modulation = max_frequency_change * velocity * intensity

#         # Define cutoff frequency for the high-pass filter
#         cutoff_frequency = 1000 + cutoff_modulation  # Base cutoff frequency of 1000 Hz

#         # Design a high-pass Butterworth filter
#         order = 4  # Filter order
#         b_high, a_high = butter(order, cutoff_frequency, btype='high', fs=sample_rate)

#         # Apply the high-pass filter to the audio signal
#         filtered_signal = lfilter(b_high, a_high, audio_signal)

#         return filtered_signal

#     def apply_compressor(signal, threshold_db=-20.0, ratio=2.0, attack_ms=10, release_ms=100, makeup_gain_db=0.0, sidechain_signal=None, sample_rate=44100):
#         """
#         Apply audio compression to an input audio signal.

#         Parameters:
#             signal (ndarray): Input audio signal (numpy array).
#             threshold_db (float): Threshold level in dB for compression (default: -20.0 dB).
#             ratio (float): Compression ratio (default: 2.0).
#             attack_ms (float): Attack time in milliseconds (default: 10 ms).
#             release_ms (float): Release time in milliseconds (default: 100 ms).
#             makeup_gain_db (float): Makeup gain in dB (default: 0.0 dB).
#             sidechain_signal (ndarray): Optional sidechain signal for dynamic compression (default: None).
#             sample_rate (int): Sampling rate of the audio signal (default: 44100 Hz).

#         Returns:
#             ndarray: Compressed output audio signal.
#         """
#         # Convert threshold from dB to linear scale
#         threshold_lin = 10.0 ** (threshold_db / 20.0)

#         # Calculate attack and release time constants in samples
#         attack_samples = int(attack_ms * 0.001 * sample_rate)
#         release_samples = int(release_ms * 0.001 * sample_rate)

#         # Initialize arrays for envelope and output signal
#         envelope = np.zeros_like(signal)
#         output_signal = np.zeros_like(signal)

#         # Apply compression block-wise
#         block_size = len(signal)
#         num_blocks = len(signal) // block_size

#         for i in range(num_blocks):
#             start_idx = i * block_size
#             end_idx = (i + 1) * block_size

#             # Compute envelope using peak detection (sidechain if provided)
#             if sidechain_signal is not None:
#                 envelope[start_idx:end_idx] = np.abs(sidechain_signal[start_idx:end_idx])
#             else:
#                 envelope[start_idx:end_idx] = np.abs(signal[start_idx:end_idx])

#             # Apply compression to the envelope
#             for j in range(start_idx, end_idx):
#                 if envelope[j] > threshold_lin:
#                     # Calculate compression gain reduction
#                     gain_reduction = (envelope[j] - threshold_lin) / ratio

#                     # Apply attack/release envelope smoothing
#                     if gain_reduction > envelope[j - 1]:
#                         envelope[j] = self.attack_filter(envelope[j], envelope[j - 1], attack_samples)
#                     else:
#                         envelope[j] = self.release_filter(envelope[j], envelope[j - 1], release_samples)

#                     # Apply makeup gain and compression
#                     gain = threshold_lin + gain_reduction
#                     output_signal[j] = signal[j] * (gain / envelope[j])
#                 else:
#                     output_signal[j] = signal[j]

#         # Apply makeup gain to the output signal
#         output_signal *= 10.0 ** (makeup_gain_db / 20.0)

#         return output_signal

#     def attack_filter(self, current_value, previous_value, attack_samples):
#         """Apply attack envelope smoothing."""
#         alpha = np.exp(-1.0 / attack_samples)
#         return alpha * previous_value + (1.0 - alpha) * current_value

#     def release_filter(self, current_value, previous_value, release_samples):
#         """Apply release envelope smoothing."""
#         alpha = np.exp(-1.0 / release_samples)
#         return alpha * previous_value + (1.0 - alpha) * current_value

#     def apply_bit_crushing(audio_signal, bit_depth):
#         """Apply bit crushing effect to the audio signal."""
#         # Calculate the number of quantization levels based on bit depth
#         levels = 2 ** bit_depth

#         # Quantize the audio signal
#         quantized_signal = np.floor(audio_signal * levels) / levels

#         return quantized_signal

#     def apply_sample_reduction(audio_signal, reduction_factor):
#         """Apply sample reduction effect to the audio signal."""
#         # Reduce the sample rate by a specified reduction factor
#         reduced_signal = audio_signal[::reduction_factor]

#         return reduced_signal

#     def apply_spring_reverb(audio_signal, decay_factor, delay_length):
#         """Apply spring reverb effect to the audio signal."""
#         # Create a feedback delay line with a low-pass filter
#         feedback_delay = np.zeros(delay_length)
#         output_signal = np.zeros_like(audio_signal)

#         for i in range(len(audio_signal)):
#             # Calculate output sample (comb filter with feedback)
#             output_sample = audio_signal[i] + decay_factor * feedback_delay[0]

#             # Update delay line (feedback delay)
#             feedback_delay = np.roll(feedback_delay, 1)
#             feedback_delay[0] = output_sample

#             # Store output sample
#             output_signal[i] = output_sample

#         return output_signal

#     def apply_natural_reverb(audio_signal, decay_factor, delay_length):
#         """Apply natural reverb effect to the audio signal (simple comb filter)."""
#         # Create a feedback delay line (comb filter)
#         feedback_delay = np.zeros(delay_length)
#         output_signal = np.zeros_like(audio_signal)

#         for i in range(len(audio_signal)):
#             # Calculate output sample (comb filter with feedback)
#             output_sample = audio_signal[i] + decay_factor * feedback_delay[0]

#             # Update delay line (feedback delay)
#             feedback_delay = np.roll(feedback_delay, 1)
#             feedback_delay[0] = output_sample

#             # Store output sample
#             output_signal[i] = output_sample

#         return output_signal

#     def apply_delay(audio_signal, delay_time, feedback_gain):
#         """Apply delay effect to the audio signal."""
#         # Calculate delay length in samples
#         delay_samples = int(delay_time * sample_rate)

#         # Create a delay line with feedback
#         delay_line = np.zeros(delay_samples)
#         output_signal = np.zeros_like(audio_signal)

#         for i in range(len(audio_signal)):
#             # Calculate output sample (delay with feedback)
#             output_sample = audio_signal[i] + feedback_gain * delay_line[0]

#             # Update delay line (delay buffer)
#             delay_line = np.roll(delay_line, 1)
#             delay_line[0] = output_sample

#             # Store output sample
#             output_signal[i] = output_sample

#         return output_signal

#     def apply_wave_folding(audio_signal, fold_factor):
#         """Apply wave folding effect to the audio signal."""
#         # Apply wave folding (fold the signal at specified fold factor)
#         folded_signal = np.abs(np.mod(audio_signal, fold_factor) - fold_factor / 2)

#         return folded_signal

#     def apply_analog_drive(audio_signal, drive_gain):
#         """Apply analog-style drive effect to the audio signal."""
#         # Apply soft clipping (tanh saturation) with drive gain
#         driven_signal = np.tanh(drive_gain * audio_signal)

#         return driven_signal

#     def apply_warm_saturation(audio_signal, saturation_amount):
#         """Apply warm saturation effect to the audio signal."""
#         # Apply soft clipping (tanh saturation) with adjustable amount
#         saturated_signal = np.tanh(saturation_amount * audio_signal)

#         return saturated_signal

#     def apply_pan(audio_signal, pan_position):
#         """Apply panning effect to the audio signal (left-right balance)."""
#         # Create stereo audio with specified pan position (-1 to 1, -1 = left, 1 = right)
#         left_signal = np.sqrt(0.5) * (np.cos(pan_position * np.pi / 2) * audio_signal)
#         right_signal = np.sqrt(0.5) * (np.sin(pan_position * np.pi / 2) * audio_signal)

#         stereo_signal = np.vstack((left_signal, right_signal))

#         return stereo_signal

#     def apply_clean_gain(audio_signal, gain_factor):
#         """Apply clean gain (amplification) to the audio signal."""
#         # Amplify the audio signal by a specified gain factor
#         amplified_signal = gain_factor * audio_signal

#         return amplified_signal

#     def waveguide_synthesis(self, tune, decay, body):
#         """
#         Apply waveguide synthesis to an input audio signal.

#         Parameters:
#         - self.source: Input audio signal as a numpy array
#         - tune: tune control (frequency in Hz)
#         - decay: Decay control (decay time in seconds)
#         - body: body control (feedback amount)

#         Returns:
#         - numpy array containing the synthesized output signal
#         """
#         # Length of the delay line based on tune (adjust based on desired range)
#         delay_length = int(self.sample_rate / tune)

#         # Initialize delay line and output signal
#         delay_line = np.zeros(delay_length)
#         output_signal = np.zeros_like(self.source)

#         # Process each sample in the input signal
#         for i in range(len(self.source)):
#             # Read from delay line (comb filter)
#             output_sample = delay_line[0]

#             # Update delay line with feedback (all-pass filter)
#             delay_line[0] = self.source[i] + body * delay_line[0]

#             # Move samples in delay line (implement tune control)
#             delay_line = np.roll(delay_line, 1)

#             # Apply decay envelope
#             output_signal[i] = output_sample
#             delay_line *= (1.0 - 1.0 / (decay * self.sample_rate))

#         return output_signal

#     def apply_multi_channel_eq(audio_signal, sample_rate, band_gains, band_centers, band_qs):
#         """
#         Apply multi-channel equalization to the input audio signal using specified EQ parameters.

#         Args:
#             audio_signal (ndarray): Input audio signal (1D array).
#             sample_rate (int): Sampling rate of the audio signal (in Hz).
#             band_gains (list): List of gain values (in dB) for each EQ band.
#             band_centers (list): List of center frequencies (in Hz) for each EQ band.
#             band_qs (list): List of Q factors (bandwidth) for each EQ band.

#         Returns:
#             ndarray: Output audio signal after applying EQ.
#         """
#         num_bands = len(band_gains)

#         # Initialize the combined filter coefficients
#         sos = np.zeros((num_bands, 6))

#         # Design peaking filters for each EQ band
#         for i in range(num_bands):
#             center_freq = band_centers[i]
#             Q = band_qs[i]
#             gain_db = band_gains[i]

#             # Compute peaking filter coefficients (second-order section)
#             sos[i, :], _ = iirpeak(center_freq, Q, sample_rate, gain_db=gain_db, output='sos')

#         # Apply the EQ filters using cascaded second-order sections (SOS)
#         output_signal = sosfilt(sos, audio_signal)

#         return output_signal

# @dataclass
# class MixEffects:

#     def apply_dynamic_bandpass_filter(self):
#         """Apply a dynamic bandpass filter with variable cutoff frequency and resonance (Q-factor)."""
#         # Calculate center frequency modulation based on velocity and intensity
#         max_frequency_change = 2000  # Maximum frequency change in Hz

#         # Calculate center frequency modulation based on velocity (0 to 1) and intensity (-1 to 1)
#         center_frequency_modulation = max_frequency_change * self.velocity * self.intensity

#         # Define center frequency for the bandpass filter
#         center_frequency = 3000 + center_frequency_modulation  # Base center frequency of 3000 Hz

#         # Calculate bandwidth modulation based on resonance
#         max_bandwidth = 2000  # Maximum bandwidth change in Hz

#         # Calculate bandwidth based on resonance (higher Q-factor for narrower bandwidth)
#         bandwidth = max_bandwidth * self.filter_resonanceresonance

#         # Calculate lower and upper cutoff frequencies for the bandpass filter
#         lower_cutoff = center_frequency - (bandwidth / 2)
#         upper_cutoff = center_frequency + (bandwidth / 2)

#         # Design a bandpass Butterworth filter with specified cutoff frequencies and order
#         order = 4  # Filter order
#         b_bandpass, a_bandpass = butter(order, [lower_cutoff, upper_cutoff], btype='bandpass', fs=self.sample_rate, output='ba')

#         # Apply the bandpass filter to the audio signal
#         filtered_signal = lfilter(b_bandpass, a_bandpass, self.source)

#         return filtered_signal

#     def apply_dynamic_lowpass_filter(audio_signal, sample_rate, velocity, intensity, resonance):
#         """Apply a dynamic low-pass filter with variable cutoff frequency and resonance (Q-factor)."""
#         # Calculate cutoff frequency modulation based on velocity and intensity
#         max_frequency_change = 2000  # Maximum frequency change in Hz

#         # Calculate cutoff frequency modulation based on velocity (0 to 1) and intensity (-1 to 1)
#         cutoff_modulation = max_frequency_change * velocity * intensity

#         # Define cutoff frequency for the low-pass filter
#         cutoff_frequency = 5000 + cutoff_modulation  # Base cutoff frequency of 5000 Hz

#         # Calculate Q-factor (resonance) for the low-pass filter
#         Q_factor = resonance  # Adjust resonance (higher Q_factor for more resonance)

#         # Design a low-pass Butterworth filter with specified cutoff and Q-factor
#         b_low, a_low = butter(4, cutoff_frequency, btype='low', fs=sample_rate, output='ba')

#         # Apply the low-pass filter to the audio signal
#         filtered_signal = lfilter(b_low, a_low, audio_signal)

#         return filtered_signal

#     def apply_dynamic_highpass_filter(audio_signal, sample_rate, velocity, intensity):
#         """Apply a dynamic high-pass filter with variable cutoff frequency."""
#         # Calculate cutoff frequency modulation based on velocity and intensity
#         max_frequency_change = 1000  # Maximum frequency change in Hz

#         # Calculate cutoff frequency modulation based on velocity (0 to 1) and intensity (-1 to 1)
#         cutoff_modulation = max_frequency_change * velocity * intensity

#         # Define cutoff frequency for the high-pass filter
#         cutoff_frequency = 1000 + cutoff_modulation  # Base cutoff frequency of 1000 Hz

#         # Design a high-pass Butterworth filter
#         order = 4  # Filter order
#         b_high, a_high = butter(order, cutoff_frequency, btype='high', fs=sample_rate)

#         # Apply the high-pass filter to the audio signal
#         filtered_signal = lfilter(b_high, a_high, audio_signal)

#         return filtered_signal

#     def apply_compressor(signal, threshold_db=-20.0, ratio=2.0, attack_ms=10, release_ms=100, makeup_gain_db=0.0, sidechain_signal=None, sample_rate=44100):
#         """
#         Apply audio compression to an input audio signal.

#         Parameters:
#             signal (ndarray): Input audio signal (numpy array).
#             threshold_db (float): Threshold level in dB for compression (default: -20.0 dB).
#             ratio (float): Compression ratio (default: 2.0).
#             attack_ms (float): Attack time in milliseconds (default: 10 ms).
#             release_ms (float): Release time in milliseconds (default: 100 ms).
#             makeup_gain_db (float): Makeup gain in dB (default: 0.0 dB).
#             sidechain_signal (ndarray): Optional sidechain signal for dynamic compression (default: None).
#             sample_rate (int): Sampling rate of the audio signal (default: 44100 Hz).

#         Returns:
#             ndarray: Compressed output audio signal.
#         """
#         # Convert threshold from dB to linear scale
#         threshold_lin = 10.0 ** (threshold_db / 20.0)

#         # Calculate attack and release time constants in samples
#         attack_samples = int(attack_ms * 0.001 * sample_rate)
#         release_samples = int(release_ms * 0.001 * sample_rate)

#         # Initialize arrays for envelope and output signal
#         envelope = np.zeros_like(signal)
#         output_signal = np.zeros_like(signal)

#         # Apply compression block-wise
#         block_size = len(signal)
#         num_blocks = len(signal) // block_size

#         for i in range(num_blocks):
#             start_idx = i * block_size
#             end_idx = (i + 1) * block_size

#             # Compute envelope using peak detection (sidechain if provided)
#             if sidechain_signal is not None:
#                 envelope[start_idx:end_idx] = np.abs(sidechain_signal[start_idx:end_idx])
#             else:
#                 envelope[start_idx:end_idx] = np.abs(signal[start_idx:end_idx])

#             # Apply compression to the envelope
#             for j in range(start_idx, end_idx):
#                 if envelope[j] > threshold_lin:
#                     # Calculate compression gain reduction
#                     gain_reduction = (envelope[j] - threshold_lin) / ratio

#                     # Apply attack/release envelope smoothing
#                     if gain_reduction > envelope[j - 1]:
#                         envelope[j] = self.attack_filter(envelope[j], envelope[j - 1], attack_samples)
#                     else:
#                         envelope[j] = self.release_filter(envelope[j], envelope[j - 1], release_samples)

#                     # Apply makeup gain and compression
#                     gain = threshold_lin + gain_reduction
#                     output_signal[j] = signal[j] * (gain / envelope[j])
#                 else:
#                     output_signal[j] = signal[j]

#         # Apply makeup gain to the output signal
#         output_signal *= 10.0 ** (makeup_gain_db / 20.0)

#         return output_signal

#     def attack_filter(self, current_value, previous_value, attack_samples):
#         """Apply attack envelope smoothing."""
#         alpha = np.exp(-1.0 / attack_samples)
#         return alpha * previous_value + (1.0 - alpha) * current_value

#     def release_filter(self, current_value, previous_value, release_samples):
#         """Apply release envelope smoothing."""
#         alpha = np.exp(-1.0 / release_samples)
#         return alpha * previous_value + (1.0 - alpha) * current_value

#     def apply_bit_crushing(audio_signal, bit_depth):
#         """Apply bit crushing effect to the audio signal."""
#         # Calculate the number of quantization levels based on bit depth
#         levels = 2 ** bit_depth

#         # Quantize the audio signal
#         quantized_signal = np.floor(audio_signal * levels) / levels

#         return quantized_signal

#     def apply_sample_reduction(audio_signal, reduction_factor):
#         """Apply sample reduction effect to the audio signal."""
#         # Reduce the sample rate by a specified reduction factor
#         reduced_signal = audio_signal[::reduction_factor]

#         return reduced_signal

#     def apply_spring_reverb(audio_signal, decay_factor, delay_length):
#         """Apply spring reverb effect to the audio signal."""
#         # Create a feedback delay line with a low-pass filter
#         feedback_delay = np.zeros(delay_length)
#         output_signal = np.zeros_like(audio_signal)

#         for i in range(len(audio_signal)):
#             # Calculate output sample (comb filter with feedback)
#             output_sample = audio_signal[i] + decay_factor * feedback_delay[0]

#             # Update delay line (feedback delay)
#             feedback_delay = np.roll(feedback_delay, 1)
#             feedback_delay[0] = output_sample

#             # Store output sample
#             output_signal[i] = output_sample

#         return output_signal

#     def apply_natural_reverb(audio_signal, decay_factor, delay_length):
#         """Apply natural reverb effect to the audio signal (simple comb filter)."""
#         # Create a feedback delay line (comb filter)
#         feedback_delay = np.zeros(delay_length)
#         output_signal = np.zeros_like(audio_signal)

#         for i in range(len(audio_signal)):
#             # Calculate output sample (comb filter with feedback)
#             output_sample = audio_signal[i] + decay_factor * feedback_delay[0]

#             # Update delay line (feedback delay)
#             feedback_delay = np.roll(feedback_delay, 1)
#             feedback_delay[0] = output_sample

#             # Store output sample
#             output_signal[i] = output_sample

#         return output_signal

#     def apply_delay(audio_signal, delay_time, feedback_gain):
#         """Apply delay effect to the audio signal."""
#         # Calculate delay length in samples
#         delay_samples = int(delay_time * sample_rate)

#         # Create a delay line with feedback
#         delay_line = np.zeros(delay_samples)
#         output_signal = np.zeros_like(audio_signal)

#         for i in range(len(audio_signal)):
#             # Calculate output sample (delay with feedback)
#             output_sample = audio_signal[i] + feedback_gain * delay_line[0]

#             # Update delay line (delay buffer)
#             delay_line = np.roll(delay_line, 1)
#             delay_line[0] = output_sample

#             # Store output sample
#             output_signal[i] = output_sample

#         return output_signal

#     def apply_wave_folding(audio_signal, fold_factor):
#         """Apply wave folding effect to the audio signal."""
#         # Apply wave folding (fold the signal at specified fold factor)
#         folded_signal = np.abs(np.mod(audio_signal, fold_factor) - fold_factor / 2)

#         return folded_signal

#     def apply_analog_drive(audio_signal, drive_gain):
#         """Apply analog-style drive effect to the audio signal."""
#         # Apply soft clipping (tanh saturation) with drive gain
#         driven_signal = np.tanh(drive_gain * audio_signal)

#         return driven_signal

#     def apply_warm_saturation(audio_signal, saturation_amount):
#         """Apply warm saturation effect to the audio signal."""
#         # Apply soft clipping (tanh saturation) with adjustable amount
#         saturated_signal = np.tanh(saturation_amount * audio_signal)

#         return saturated_signal

#     def apply_pan(audio_signal, pan_position):
#         """Apply panning effect to the audio signal (left-right balance)."""
#         # Create stereo audio with specified pan position (-1 to 1, -1 = left, 1 = right)
#         left_signal = np.sqrt(0.5) * (np.cos(pan_position * np.pi / 2) * audio_signal)
#         right_signal = np.sqrt(0.5) * (np.sin(pan_position * np.pi / 2) * audio_signal)

#         stereo_signal = np.vstack((left_signal, right_signal))

#         return stereo_signal

#     def apply_clean_gain(audio_signal, gain_factor):
#         """Apply clean gain (amplification) to the audio signal."""
#         # Amplify the audio signal by a specified gain factor
#         amplified_signal = gain_factor * audio_signal

#         return amplified_signal

#     def waveguide_synthesis(self, tune, decay, body):
#         """
#         Apply waveguide synthesis to an input audio signal.

#         Parameters:
#         - self.source: Input audio signal as a numpy array
#         - tune: tune control (frequency in Hz)
#         - decay: Decay control (decay time in seconds)
#         - body: body control (feedback amount)

#         Returns:
#         - numpy array containing the synthesized output signal
#         """
#         # Length of the delay line based on tune (adjust based on desired range)
#         delay_length = int(self.sample_rate / tune)

#         # Initialize delay line and output signal
#         delay_line = np.zeros(delay_length)
#         output_signal = np.zeros_like(self.source)

#         # Process each sample in the input signal
#         for i in range(len(self.source)):
#             # Read from delay line (comb filter)
#             output_sample = delay_line[0]

#             # Update delay line with feedback (all-pass filter)
#             delay_line[0] = self.source[i] + body * delay_line[0]

#             # Move samples in delay line (implement tune control)
#             delay_line = np.roll(delay_line, 1)

#             # Apply decay envelope
#             output_signal[i] = output_sample
#             delay_line *= (1.0 - 1.0 / (decay * self.sample_rate))

#         return output_signal

#     def apply_multi_channel_eq(audio_signal, sample_rate, band_gains, band_centers, band_qs):
#         """
#         Apply multi-channel equalization to the input audio signal using specified EQ parameters.

#         Args:
#             audio_signal (ndarray): Input audio signal (1D array).
#             sample_rate (int): Sampling rate of the audio signal (in Hz).
#             band_gains (list): List of gain values (in dB) for each EQ band.
#             band_centers (list): List of center frequencies (in Hz) for each EQ band.
#             band_qs (list): List of Q factors (bandwidth) for each EQ band.

#         Returns:
#             ndarray: Output audio signal after applying EQ.
#         """
#         num_bands = len(band_gains)

#         # Initialize the combined filter coefficients
#         sos = np.zeros((num_bands, 6))

#         # Design peaking filters for each EQ band
#         for i in range(num_bands):
#             center_freq = band_centers[i]
#             Q = band_qs[i]
#             gain_db = band_gains[i]

#             # Compute peaking filter coefficients (second-order section)
#             sos[i, :], _ = iirpeak(center_freq, Q, sample_rate, gain_db=gain_db, output='sos')

#         # Apply the EQ filters using cascaded second-order sections (SOS)
#         output_signal = sosfilt(sos, audio_signal)

#         return output_signal

# class ProbablyDupes:
#         def gen_sine_wave(self, pitch_override=0):
#         """Generate Sine Wave"""
#         if not pitch_override:
#             pitch_override = self.frequency
#         t = np.linspace(0, self.duration, self.num_samples, endpoint=False)
#         sine_wave = np.sin(2 * np.pi * pitch_override * t)
#         sine_wave /= np.max(np.abs(sine_wave))
#         return sine_wave

#     def apply_high_pass_filter(self, signal, cutoff_frequency):
#         """Apply a high-pass Butterworth filter to the signal."""
#         nyquist_freq = 0.5 * self.sample_rate
#         normal_cutoff = cutoff_frequency / nyquist_freq
#         b, a = butter(4, normal_cutoff, btype="high", analog=True)
#         filtered_signal = lfilter(b=b, a=a, x=signal)
#         filtered_signal /= np.max(np.abs(filtered_signal))
#         return filtered_signal

#     def apply_band_pass_filter(self, signal, cutoff_frequency, width=100):
#         """Apply a dynamic bandpass filter with variable cutoff frequency and resonance (Q-factor)."""
#         # Calculate lower and upper cutoff frequencies for the bandpass filter
#         nyquist_freq = 0.5 * self.sample_rate
#         lower_cutoff = (cutoff_frequency - (width)) / nyquist_freq
#         upper_cutoff = (cutoff_frequency + (width)) / nyquist_freq
#         # Design a bandpass Butterworth filter with specified cutoff frequencies and order
#         order = 4  # Filter order
#         b_bandpass, a_bandpass = butter(
#             order,
#             [lower_cutoff, upper_cutoff],
#             btype="bandpass",
#             fs=self.sample_rate,
#             output="ba",
#         )

# Apply the bandpass filter to the audio signal
#     filtered_signal = lfilter(b_bandpass, a_bandpass, signal)
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
#     envelope = np.exp(-5 * (t - 0.5) ** 2 / (0.1 * self.attack**2))
#     envelope *= np.exp(-5 * t / self.decay)
#     return envelope

# def modulate_amplitude(self):
#     """Generate an amplitude-modulated signal (AM) using a modulation envelope."""
#     t = np.linspace(
#         0, self.duration, int(self.sample_rate * self.duration), endpoint=False
#     )

# Carrier signal (sine wave)
# carrier_signal = np.sin(2 * np.pi * self.source * t)

# Apply amplitude modulation (AM) with the modulation envelope
# am_signal = carrier_signal * self.modulation_env

# return am_signal

# def apply_envelope(self):
# Apply a simple exponential decay envelope
# attack_samples = int(sample_rate * attack_time)
# attack_ramp = np.linspace(0, 1, attack_samples)
# wave[:attack_samples] *= attack_ramp
# decay = np.exp(-t / decay_time)
# wave *= decay

# Generate amplitude-modulated (AM) signal using the selected modulation envelope
# am_signal = self.modulate_amplitude(self.spectra, self.env_type)

# Normalize the AM signal to the range [-1, 1]
#     am_signal /= np.max(np.abs(am_signal))

# def generate_log_decay_mod(self):
#     """Generate a logarithmic decay envelope."""
#     t = np.linspace(0, 1, self.num_samples)
#     mod_src = np.exp(-5 * (1 - t) / self.attack) * np.exp(-5 * t / self.decay)
#     return mod_src

# def generate_sine_wave(self):
#     """Generate a sine wave for mod, but other generator can be used."""
#     t = np.linspace(0, self.duration, self.num_samples, endpoint=False)
#     waveform = np.sin(2 * np.pi * self.frequency * t)
#     return waveform

# def generate_noise(self):
#     """Generate white noise for mod."""
#     noise = np.random.uniform(low=-1.0, high=1.0, size=self.num_samples)
#     return noise
