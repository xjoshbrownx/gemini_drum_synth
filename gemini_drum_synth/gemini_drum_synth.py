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
class SoundChannel:
    """
    A multi-layer (non-realtime) drum synthesizer that combines several drum-synth sound generation models and techniques as layers that can be mixed together and enhanced with FX.
    Can be used for percussion, drum, bass sounds.

    PARAMETERS:
    channel_filepath: Path = Path.home() / "drumsynth/samples"

    # AUDIO SETTINGS
    channel_sample_rate: int = 44100  # Sample rate (Hz)
    channel_bitdepth: int = 16

    # CHANNEL SETTINGS
    velocity: int = 90  # intensity of drum hit
    pan: int = 0
    level: float = 0.5

    # FX SETTINGS
    wave_guide_send: int = 0
    distortion_send: int = 0
    reverb_send: int = 0
    delay_send: int = 0
    fx_bus_1_send: int = 0
    fx_bus_2_send: int = 0
    """

    # ADMIN SETTINGS
    channel_audio = np.zeros(1)
    channel_filepath: Path = Path.home() / "drumsynth/samples"

    # AUDIO SETTINGS
    channel_sample_rate: int = 44100  # Sample rate (Hz)
    channel_bitdepth: int = 16

    # CHANNEL SETTINGS
    # velocity: int = 90  # intensity of drum hit NOT YET INTEGRATED INTO SOUND GENERATION
    pan: int = 0
    level: float = 0.5

    # FX SETTINGS
    wave_guide_send: int = 0
    distortion_send: int = 0
    reverb_send: int = 0
    delay_send: int = 0
    fx_bus_1_send: int = 0
    fx_bus_2_send: int = 0

    def __post_init__(self):
        self.layers: list = []  # list of layers of audio samples

    def add_nd_click_layer(
        self,
        layer_level: float = 1.0,
        click_type: str = 'N2',
        click_duration: float = 0.01,
    ):
        """
        Adds ND_ClickLayer object to Sound Channel object layer list.
        """
        layer = ND_ClickLayer(
            bit_depth=self.channel_bitdepth,
            sample_rate=self.channel_sample_rate,
            layer_level=layer_level,
            click_type=click_type,
            click_duration=click_duration,
        )
        self.layers.append(layer)

    def add_nd_noise_layer(
        self,
        layer_level: float = 1.0,
        pitch: int = 0,
        attack: float = 0.025,
        decay: float = 1.0,
        decay_type: str = 'E',
        filter_type: str = 'L2',
        resonance: float = 0.5,
        freq: int = 120,
        dynamic_filter: int = 5,
    ):
        """
        Adds ND_NoiseLayer object to Sound Channel object layer list.
        """
        layer = ND_NoiseLayer(
            bit_depth=self.channel_bitdepth,
            sample_rate=self.channel_sample_rate,
            layer_level=layer_level,
            pitch=pitch,
            attack=attack,
            decay=decay,
            decay_type=decay_type,
            filter_type=filter_type,
            resonance=resonance,
            freq=freq,
            dynamic_filter=dynamic_filter,
        )
        self.layers.append(layer)

    def add_vd_layer(
        self,
        layer_level: float,
        pitch: int,
        attack: float,
        decay: float,
        src_type: str,
        mod_type: str,
        env_type: str,
        noise_type: str,
        frequency: int,
        mod_amount: float,
        mod_rate: int,
        wave_guide_mix: float,
        wave_decay: float,
        wave_tone: float,
        wave_body: float,
    ):
        """
        Adds VD_GenericLayer object to Sound Channel object layer list.
        """
        layer = VD_GenericLayer(
            bit_depth=self.channel_bitdepth,
            sample_rate=self.channel_sample_rate,
            layer_level=layer_level,
            pitch=pitch,
            attack=attack,
            decay=decay,
            src_type=src_type,
            mod_type=mod_type,
            env_type=env_type,
            noise_type=noise_type,
            frequency=frequency,
            mod_amount=mod_amount,
            mod_rate=mod_rate,
            wave_guide_mix=wave_guide_mix,
            wave_decay=wave_decay,
            wave_tone=wave_tone,
            wave_body=wave_body,
        )
        self.layers.append(layer)

    def mix_signals(self, arr1, arr2):
        """
        Mixes signals (in form of numpy arrays) of diffrent lengths and returns an array of the length of the longest.
        """
        l = sorted((arr1, arr2), key=len)
        c = l[1].copy()
        c[: len(l[0])] += l[0]
        return c

    def mix_layers(self):
        # Placeholder for mixing audio samples from all layers
        for layer in self.layers:
            # Generate audio sample for the layer (e.g., using generate_sample method)
            layer.gen_layer_sound()
            layer.apply_level()
            self.channel_audio = self.mix_signals(layer.layer_audio, self.channel_audio)

            # Mix audio sample into the final track (e.g., add to list)

        self.channel_audio

    def save_channel_audio(self, filename):
        """
        Save the audio signal to a WAV file.

        Parameters:
        - audio: numpy array containing the audio signal
        - filename: name of the output WAV file
        """
        # Scale audio to 16-bit integer range (-32768 to 32767)
        audio_int = (self.channel_audio * 32767).astype(np.int16)

        # Save the audio to a WAV file
        wavfile.write(
            self.channel_filepath / f"{filename}.wav",
            self.channel_sample_rate,
            audio_int,
        )


@dataclass
class SynthLayer:
    # ADMIN SETTINGS
    layer_audio: np.array = np.zeros(1)
    filepath: Path = Path.home() / "drumsynth/samples"

    # AUDIO SETTINGS
    num_channels = 1
    bit_depth: int = 16
    sample_rate: int = 44100

    # LAYER SETTINGS
    layer_level: float = 0.5
    pitch: int = 0  # adjust pitch of sound in semitones 60 = 256hz, 69 = 440hz
    attack: float = 0.000001  # Duration of the attack of synthesized sound (seconds) to avoid divide by zero errors
    decay: float = 2.0  # Duration of the decay synthesized sound (seconds)

    def __post_init__(self):
        self._level_wrap_around()
        self.attack = (
            self.attack if self.attack else 0.00001
        )  # SET ATTACK TO NON-ZERO VALUE TO AVOID DIVIDE BY ZERO ERRORS
        self.attack_samples = int(np.floor(self.attack * self.sample_rate))
        self.decay_samples = int(np.floor(self.decay * self.sample_rate))
        # self.duration = self.attack + self.decay
        self.num_samples = int(self.attack_samples + self.decay_samples)
        self.att_t = self._gen_t(num_samples=self.attack_samples)
        self.dec_t = self._gen_t(num_samples=self.decay_samples)
        self.env_t = self._gen_t(num_samples=self.num_samples)

    def __str__(self):
        return self.create_long_layer_desc()

    def __repr__(self):
        return self.create_long_layer_desc()

    ####OUTPUT_METHODS####
    def apply_level(self):
        self.layer_audio *= self.layer_level

    def save_layer(self, filename=''):
        """
        Save the audio signal to a WAV file.

        Parameters:
        - audio: numpy array containing the audio signal
        - filename: name of the output WAV file
        """
        if not filename:
            filename = self.create_short_layer_desc()

        # Scale audio to 16-bit integer range (-32768 to 32767)
        audio_int = (self.layer_audio * 32767).astype(np.int16)

        # Save the audio to a WAV file
        wavfile.write(self.filepath / f"{filename}.wav", self.sample_rate, audio_int)

    def create_short_layer_desc(self):
        """Creates a shortened description of the layer object based on the arguments used to initialize the object, removing paths and arrays"""
        return '_'.join(
            [
                f'{"".join([l[0] if l else "" for l in x.split("_")])}-{y}'
                for x, y in self.__dict__.items()
                if not isinstance(y, (Path, np.ndarray))
            ]
        )

    def create_long_layer_desc(self):
        """Creates an explicit description of the layer object based on the arguments used to initialize the object, removing paths and arrays"""
        return '_'.join(
            [
                f'{x}: {y}'
                for x, y in self.__dict__.items()
                if not isinstance(y, (Path, np.ndarray))
            ]
        )

    def print_it_all(self, long=True):
        if long:
            print(self.create_long_layer_desc())
        else:
            print(self.create_short_layer_desc())

    ####HELPER_FUNCTIONS####

    def _wrap_around(self, setting):
        """Keeps settings between 0-1 by finding absolute value and applying modulo 1"""
        if setting > 1:
            return abs(setting) % 1.0
        else:
            return setting

    def _level_wrap_around(self):
        self.layer_level = self._wrap_around(self.layer_level)

    def _gen_t(self, num_samples=441):
        """
        Creates time domain array for envelope generators
        """
        return np.linspace(0, 1, num_samples, endpoint=True)

    def _translate_filter(self, filter_type="L2"):
        """
        Translates filter type values to uncalled filter methods that can be called with settings locally.
        """
        _filter_dict = {
            'L1': LadderFilter.LPF12,
            'L2': LadderFilter.LPF24,
            'H1': LadderFilter.HPF12,
            'H2': LadderFilter.HPF24,
            'B1': LadderFilter.BPF12,
            'B2': LadderFilter.BPF24,
        }
        return _filter_dict.get(filter_type.upper())

    ####AUDIO_HELPER_FUNCTIONS####

    def mix_signals(self, arr1, arr2):
        """
        Mixes signals (in form of numpy arrays) of diffrent lengths and returns an array of the length of the longest.
        """
        l = sorted((arr1, arr2), key=len)
        c = l[1].copy()
        c[: len(l[0])] += l[0]
        return c

    def midi_note_to_freq(self, pitch):
        """Convert midi notes to frequency"""
        return 2 ** ((pitch - 68) / 12) * 440

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

    ####NORMALIZATION_METHODS####

    def one_neg_one_normalization(self, signal):
        """
        Normalize audio signal to between -1 and 1.
        *low divide by zero risk
        """
        signal /= np.max(np.abs(signal))
        return signal

    def min_max_normalization(self, signal):
        return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

    def normalize_audio(self, signal):
        return 2 * self.min_max_normalization(signal) - 1

    ####WAVE_OSCILLATORS####

    def gen_sine_wave(self, freq_override=0):
        """
        Generate Sine Wave
        """

        if not freq_override:
            freq_override = self.frequency
        # t = np.linspace(0, self.duration, self.num_samples, endpoint=False)
        sine_wave = np.sin(2 * np.pi * freq_override * self.env_t)
        sine_wave /= np.max(np.abs(sine_wave))
        return sine_wave

    def gen_tri_wave(self, freq_override=0):
        """
        Generate a triangle wave audio signal.

        Parameters:
            freq_override (int): allow function to work with frequences that are not stored in the self.frequency property.

        Returns:
            ndarray: Generated triangle wave audio signal.
        """
        if not freq_override:
            freq_override = self.frequency
        # t = np.linspace(
        #     0, self.duration, self.num_samples, endpoint=False
        # )  # Time array

        # Calculate angular frequency in radians
        angular_freq = 2 * np.pi * freq_override

        # Generate triangle wave using modulo operation
        triangle_wave = (
            2 * np.abs((self.env_t * angular_freq / (2 * np.pi) % 1) - 0.5) - 1
        )

        return triangle_wave

    def gen_saw_wave(self, freq_override=0):
        """Generate Saw Wave"""
        if not freq_override:
            freq_override = self.frequency
        # t = np.linspace(0, self.duration, self.num_samples, endpoint=False)
        sawtooth_wave = 2 * (
            freq_override * self.env_t - np.floor(self.frequency * self.env_t + 0.5)
        )
        sawtooth_wave /= np.max(np.abs(sawtooth_wave))
        return sawtooth_wave

    def gen_rev_saw_wave(self, freq_override=0):
        """Generate reverse saw wave"""

        rev_saw_wave = self.gen_saw_wave(freq_override) * -1

    def gen_square_wave(self, freq_override=0):
        """Generate Sine Wave

        Parameters:
            freq_override (int): allow function to work with frequences that are not stored in the self.frequency property.

        Returns:
            ndarray: Generated triangle wave audio signal.
        """

        if not freq_override:
            freq_override = self.frequency
        # t = np.linspace(0, self.duration, self.num_samples, endpoint=False)
        square_wave = np.sign(np.sin(2 * np.pi * freq_override * self.env_t))
        return square_wave

    ####CLICK_SOUND_GENERATORS####

    def gen_click(self, click_type, click_duration):
        """
        Generate a click sound waveform of the specified type.

        Parameters:
            click_type (str): Type of click sound ('simple', 'white_noise', 'impulse').
            duration (float): Duration of the click sound in seconds (default: 0.01 seconds).
            sample_rate (int): Sampling rate of the audio waveform (samples per second, default: 44100).

        Returns:
            ndarray: NumPy array containing the generated click sound waveform.
        """
        self.num_click_samples = int(self.sample_rate * click_duration)
        click_t = self._gen_t(self.num_click_samples)
        click_env = self.gen_click_env(click_t=click_t)
        # Exponential decay envelope
        # print(num_click_samples)

        if click_type == 'S1':
            # Generate a simple click (cosine wave envelope)
            self.layer_audio = np.cos(2 * np.pi * 1000 * click_t) * click_env

        elif click_type == 'N1':
            # Generate a burst of white noise
            self.layer_audio = np.random.randn(self.num_click_samples)

        elif click_type == 'N2':
            # Generate a burst of white noise with short envelope
            self.layer_audio = np.random.randn(self.num_click_samples) * click_env

        elif click_type == 'I1':
            # Generate an impulse (single-sample spike)
            self.layer_audio = np.zeros(self.num_click_samples)
            self.layer_audio[0] = 1.0

        elif click_type == 'M1':
            # Generate an metallic click (High-frequency sinusoidal component)
            high_freq_component = np.sin(2 * np.pi * 3000 * click_t)
            # Combine envelope with high-frequency component
            self.layer_audio = click_env * high_freq_component

        elif click_type == 'T1':
            # Envelope shaping for thud click
            low_freq_component = np.sin(
                2 * np.pi * 200 * click_t
            )  # Low-frequency sinusoidal component

            # Combine envelope with low-frequency component
            self.layer_audio = click_env * low_freq_component

        elif click_type == 'T2':
            # Envelope shaping for thud click
            low_freq_component = np.sin(
                2 * np.pi * 200 * click_t
            )  # Low-frequency sinusoidal component

            # Combine envelope with low-frequency component
            self.layer_audio = click_env * low_freq_component

        else:
            raise ValueError(
                f"Invalid click_type '{click_type}'. Choose from 'S1', 'N1', 'N2', 'I1', 'M1', 'T1' or 'T2'"
            )

        if self.pitch:
            self.layer_audio = self.pitch_change_semitones(self.layer_audio, self.pitch)

    ####PHYSICAL_MODELLING_OSCILLATORS####

    def karplus_strong(self):
        """
        Generate a noise sound waveform of the specified type.

        Parameters:

        Returns:
            ndarray: NumPy array containing the generated click sound waveform.
        """
        output = []
        board = Pedalboard([Delay(), LadderFilter()])
        board[0].delay_seconds = 0.01
        board[0].feedback = 0.1
        board[0].mix = 0.5
        self.layer_audio = board(
            self.gen_click(click_type='N2'), sample_rate=self.sample_rate
        )
        # step_size_in_samples = 100
        # for i in range(0, self.num_samples-1, step_size_in_samples):
        #     #     chunk = af.read(step_size_in_samples)
        #     # if self.filter_type.startswith('L'):
        #     #     board.cutoff_hz = self.freq
        #     output.append(
        #         board.process(
        #             input_array=self.layer_audio[i : i + step_size_in_samples],
        #             sample_rate=self.sample_rate,
        #             reset=False,
        #         ))
        return self.layer_audio

    ####NOISE_GENERATORS####

    def gen_white_noise(self):
        """
        Generate white noise audio signal.

        Parameters:
            # duration (float): Duration of the noise signal in seconds.
            # sample_rate (int): Sampling rate of the noise signal (samples per second).

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

    ####ENVELOPE_COMPONENT_GENERATORS####

    def gen_log_decay(self, degree=50, decay_samples=0):
        """Generate a logarithmic decay."""
        base = 0.95**degree
        if not decay_samples:
            decay_samples = self.decay_samples
        log_decay = np.flip(np.logspace(1, 0, decay_samples, base=base))
        # return (log_decay - np.min(log_decay)) / (np.max(log_decay) - np.min(log_decay))
        return (
            # terany statement rewritten as multiline by black
            self.min_max_normalization(log_decay)
            if self.decay_samples
            else np.zeros(self.decay_samples)
        )

    def gen_log_attack(self, degree=50, attack_samples=0):
        """Generate a logarithmic attack."""
        base = 0.95**degree
        if not attack_samples:
            attack_samples = self.attack_samples
        log_attack = np.flip(np.logspace(0, 1, attack_samples, base=base))
        return (
            # terany statement rewritten as multiline by black
            self.min_max_normalization(log_attack)
            if self.attack_samples
            else np.zeros(self.attack_samples)
        )

    def gen_lin_attack(self, attack_samples=0):
        """Generate a linear attack."""
        if not attack_samples:
            attack_samples = self.attack_samples
        return np.linspace(0, 1, attack_samples)

    def gen_lin_decay(self, decay_samples=0):
        """Generate a linear decay."""
        if not decay_samples:
            decay_samples = self.decay_samples
        return np.linspace(1, 0, decay_samples)

    ####ENVELOPE_GENERATORS####

    def gen_click_env(self, click_t):
        return np.exp(-5 * click_t)

    def gen_lin_att_lin_dec_env(self):
        return self.gen_lin_env()

    def gen_log_dec_no_att_env(self):
        """Generates an envelop with no attack and a log decay for duration."""
        return self.gen_log_decay(decay_samples=self.num_samples)

    def gen_lin_att_log_dec_env(self, degree=50):
        """Generate a linear attack / logrithmic decay envelope."""
        rise = self.gen_lin_attack()
        fall = self.gen_log_decay(degree=degree)
        envelope = np.concatenate([rise, fall])
        self.num_samples = rise.shape[0] + fall.shape[0]
        return envelope[: self.num_samples]

    def gen_log_att_log_dec_env(self, degree=50):
        rise = self.gen_log_attack(degree=degree)
        fall = self.gen_log_decay(degree=degree)
        envelope = np.concatenate([rise, fall])
        self.num_samples = rise.shape[0] + fall.shape[0]
        return envelope[: self.num_samples]

    def gen_lin_env(self):
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
        envelope = self.gen_lin_att_log_dec_env(degree=degree)
        zeros = np.zeros(shift_amt)
        ones = np.ones(int(self.num_samples - shift_amt))
        mask = np.concatenate((ones, zeros))
        reenvlope = np.roll(np.min((envelope, mask), axis=0), shift_amt)
        tail = envelope[self.num_samples - shift_amt :]
        remix = np.concatenate((np.max((envelope, reenvlope), axis=0), tail))
        resampled_signal = sci_resample(remix, self.num_samples)
        # self.num_samples = self.num_samples + shift_amt
        return resampled_signal

    def gen_gate_env(self):
        # TODO
        return np.ones(self.num_samples)

    def gen_punch_decay(self):
        # TODO
        return self.gen_log_decay()

    ####FILTERS####

    def apply_filter(
        self, audio_signal, filter_type='L2', cutoff_hz=440.0, resonance=0, drive=1
    ):
        """
        Applies filter to audio signal

        Parameters:
            audio_signal (numpy.array): a numpy array representing an audio signal
            filter_type (str) default = L2: type of filter comprised of a 2 character code L2 = 24db per octave low-pass,
                character 1: l = low pass, h = high pass, b = bandpass
                character 2: 1 = 12db per octave, 2 = 24db per octave
            cutoff_hz (int) default = 100: Size of steps for filter to process in samples; should be at least double cutoff , filter_type = 'L2', cutoff_hz = 440., resonance = 0, drive = 1
            duration (float): Duration of the noise signal in seconds.
            sample_rate (int): Sampling rate of the noise signal (samples per second).

        Returns:
            no return, applies filter to self.layer_audio.
        """
        board = Pedalboard([LadderFilter()])[0]
        board.mode = self._translate_filter(filter_type)
        board.cutoff_hz = cutoff_hz
        board.drive = drive
        board.resonance = resonance
        return board.process(
            input_array=audio_signal,
            sample_rate=self.sample_rate,
            reset=False,
        )

    def apply_filter_to_layer(
        self, filter_type='L2', cutoff_hz=440.0, resonance=0, drive=1
    ):
        """
        Applies filter to layer audio

        Parameters:
            filter_type (str) default = L2: type of filter comprised of a 2 character code L2 = 24db per octave low-pass,
                character 1: l = low pass, h = high pass, b = bandpass
                character 2: 1 = 12db per octave, 2 = 24db per octave
            cutoff_hz (int) default = 100: Size of steps for filter to process in samples; should be at least double cutoff , filter_type = 'L2', cutoff_hz = 440., resonance = 0, drive = 1
            duration (float): Duration of the noise signal in seconds.
            sample_rate (int): Sampling rate of the noise signal (samples per second).

        Returns:
            no return, applies filter to self.layer_audio.
        """
        self.layer_audio = self.apply_filter(
            audio_signal=self.layer_audio,
            filter_type=filter_type,
            cutoff_hz=cutoff_hz,
            resonance=resonance,
            drive=drive,
        )

    def apply_dynamic_filter(
        self, step_size=100, filter_type='L2', cutoff_hz=440.0, resonance=0, drive=1
    ):
        """
        Applies filters with modulated parameters over the length of the signal
        """
        board = Pedalboard([LadderFilter()])[0]
        step_size = (self.sample_rate // cutoff_hz) * 2
        if self.dynamic_filter:
            # FOR DYNAMIC FILTER
            # TODO CREATE CURVE AND ADAPT IT TO THE WEIRD LOW_PASS NEGATIVE HIGH PASS POSTIVE LOGIC. TEST WITH SOUNDS
            if self.dynamic_filter > 0:
                self.filter_env = (
                    self.gen_log_decay(degree=self.dynamic_filter * 10)
                    * self.freq
                    * 0.05
                    * self.dynamic_filter
                ) + self.freq
            else:
                self.filter_env = self.freq - (
                    self.gen_log_decay(degree=self.dynamic_filter * -10)
                    * self.freq
                    * 0.05
                    * -self.dynamic_filter
                )
            output = []

            for i in range(0, self.num_samples - 1, step_size):
                board.mode = self._translate_filter(filter_type)
                board.cutoff_hz = cutoff_hz
                board.drive = drive
                board.resonance = resonance
                #     chunk = af.read(step_size)
                # if self.filter_type.startswith('L'):
                #     board.cutoff_hz = self.freq
                if isinstance(cutoff_hz, list):
                    board.cutoff_hz = self.filter_env[i]
                # print(board.cutoff_hz)
                output.append(
                    board.process(
                        input_array=self.layer_audio[i : i + step_size],
                        sample_rate=self.sample_rate,
                        reset=False,
                    )
                )
            self.layer_audio = np.concatenate(output, axis=0)
        else:
            self.layer_audio = board.process(
                input_array=self.layer_audio, sample_rate=self.sample_rate
            )

    ####MODULATION_FUNCTIONS####

    def modulate_frequency(self, carrier_signal, modulation_signal):
        """
        Creates an FM signal from carrier and modulation signals.

        Parameters:
        carrier_signal: (numpy array) signal to be modulated.
        modulation_signal: (numpy array) signal to use as basis for modulation.

        Returns:
        numpy array as an audio signal
        """
        return self.one_neg_one_normalization(
            np.sin(carrier_signal + modulation_signal)
        )

    ####FX_SENDS####

    def apply_pedal_fx(self, **kwargs):
        """
        Applies filter to audio signal

        Parameters:
            audio_signal (numpy.array): a numpy array representing an audio signal
            filter_type (str) default = L2: type of filter comprised of a 2 character code L2 = 24db per octave low-pass,
                character 1: l = low pass, h = high pass, b = bandpass
                character 2: 1 = 12db per octave, 2 = 24db per octave
            cutoff_hz (int) default = 100: Size of steps for filter to process in samples; should be at least double cutoff , filter_type = 'L2', cutoff_hz = 440., resonance = 0, drive = 1
            duration (float): Duration of the noise signal in seconds.
            sample_rate (int): Sampling rate of the noise signal (samples per second).

        Returns:
            no return, applies filter to self.layer_audio.
        """
        # board = Pedalboard([LadderFilter()])[0]
        # board.mode = self._translate_filter(filter_type)
        # board.cutoff_hz = cutoff_hz
        # board.drive = drive
        # board.resonance = resonance
        # return board.process(
        #     input_array=audio_signal,
        #     sample_rate=self.sample_rate,
        #     reset=False,
        # )
        pass

    def apply_wave_guide(self, audio_signal, wave_guide_mix, decay, body, tone):
        if wave_guide_mix:
            zero_tail = np.zeros(audio_signal.shape[0])
            audio_signal = np.concatenate([audio_signal, zero_tail])

            board = Pedalboard([Delay()])
            board.delay_seconds = tone
            board.feedback = decay
            board.mix = wave_guide_mix
            # body is low pass filter
            return board.process(
                input_array=audio_signal,
                sample_rate=self.sample_rate,
                reset=False,
            )
            # return audio_signal
        else:
            return audio_signal

    def apply_wave_guide2(self, audio_signal, wave_guide_mix, decay, body, tone):
        steps = int(tone * self.sample_rate // 1)
        segments = int(1 // tone)
        leftover = 1 % tone
        arr_len = audio_signal.shape[0]
        # print(arr_len)

        if wave_guide_mix:
            zero_tail_seg_list = [np.zeros(arr_len) for x in range(segments)]
            zero_tail_leftover = [np.zeros(int(arr_len * leftover) // 1)]
            audio_signal = np.concatenate(
                [audio_signal] + zero_tail_seg_list + zero_tail_leftover
            )
            for step in np.arange(0, arr_len, steps):
                # print(audio_signal.shape)
                rolled_signal = (np.roll(audio_signal, shift=steps)) * decay
                audio_signal += rolled_signal
            return audio_signal

    def gen_wave_guide(self, audio_signal, wave_guide_mix, decay, body, tone):
        """
        Generate a noise sound waveform of the specified type.

        Parameters:

        Returns:
            ndarray: NumPy array containing the generated click sound waveform.
        """
        board = Pedalboard([Delay(), LadderFilter()])
        board[0].delay_seconds = tone
        board[0].feedback = decay
        board[0].mix = wave_guide_mix
        board[1].cutoff_hz = body * 3000.0
        output_signal = board(audio_signal, sample_rate=self.sample_rate)
        return output_signal


@dataclass
class ND_ClickLayer(SynthLayer):
    """Creates transient click sounds for various percussion types"""

    click_type: str = 'N1'
    click_duration: float = 0.01  # typically range from .005 - .05 ms

    def __post_init__(self):
        self.attack = self.click_duration
        super().__post_init__()
        # self.gen_click(click_type=self.click_type, click_duration=self.click_duration)
        self.gen_layer_sound()

    def gen_layer_sound(self):
        self.gen_click(click_type=self.click_type, click_duration=self.click_duration)


@dataclass
class ND_NoiseLayer(SynthLayer):
    """Drum Sound Layer based on the Noise Layer Synthesis in Nord Drum 3P"""

    # noise_type: str = 'white'
    filter_type: str = 'L2'  # default low pass 4 pole
    resonance: int = 0  # max 20
    freq: int = 200  # cutoff frequency in Hz
    dynamic_filter: int = 0  # plus or minus 9
    # decay: int = 0  # max 50
    decay_type: str = "E"  # 'E': 'log','L': 'lin', 'G': 'gate', 'P': 'punch'
    # drive: float = 1.0

    def __post_init__(self):
        self.attack = 0
        super().__post_init__()
        # print('does this print')
        # print(self.num_samples)
        # self.duration = self.attack + self.decay
        self.gen_layer_sound()

    def gen_layer_sound(self):
        self.filter_mode = self._translate_filter(self.filter_type)
        self.noise_decay_envelope = self._translate_decay()()
        self.layer_audio = self.gen_white_noise()
        self.filter_noise()
        self.apply_noise_envelope()

    def _translate_decay(self):
        return {
            'E': self.gen_lin_att_log_dec_env,
            'L': self.gen_lin_att_lin_dec_env,
            'G': self.gen_gate_env,
            'P': self.gen_punch_decay,
        }.get(self.decay_type)

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
            if self.dynamic_filter > 0:
                self.filter_env = (
                    self.gen_log_decay(degree=self.dynamic_filter * 10)
                    * self.freq
                    * 0.05
                    * self.dynamic_filter
                ) + self.freq
            else:
                self.filter_env = self.freq - (
                    self.gen_log_decay(degree=self.dynamic_filter * -10)
                    * self.freq
                    * 0.05
                    * -self.dynamic_filter
                )
            output = []

            for i in range(0, self.num_samples - 1, step_size_in_samples):
                #     chunk = af.read(step_size_in_samples)
                # if self.filter_type.startswith('L'):
                #     board.cutoff_hz = self.freq
                board.cutoff_hz = self.filter_env[i]
                # print(board.cutoff_hz)
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
            self.noise_decay_envelope
            * self.layer_audio[: self.noise_decay_envelope.shape[0]]
        )


@dataclass
class ND_ToneLayer(SynthLayer):
    """
    INCOMPLETE NEED MANY TYPES OF SOUND GENERATION STILL
    Parameters:
    FROM GENERIC_VD
    - sample_rate: sample rate in number of individual samples
    - src_type: type of audio signal to be generated as a basis for the drum layer sound
    - mod_type: type of modulation being applied to signal if not specified in src_type
    - env_type: type of envelope to apply to signal: linear, exponential 1, exponential 2, exponential 3, punchy, double peak
    - level: audio level 0-1 of output signal
    - attack: time in seconds for signal to rise to peak
    - decay: time in seconds for signal to drop to persistent zero after peak
    - frequency: Fundamental frequency of the oscillator or in the case of noise the cutoff filter applied to noise (Hz)
    - mod_amount: amount or amplitude of modulation to effect oscillation signal
    - mod_rate: frequency of modulated signal
    # - detune: offers a detuning function that adds an oscillator of a fix difference frequency specified by this value

    """

    wave: str = "A1"  # default sine wave
    second: int = 50  # second parameter, spectra if applicable
    third: int = 0  # third parameter of the wave also filter frequency
    dynamic_filter: int = 0
    # decay: int = 20
    decay_type: str = "dyn"
    dynamic_decay: int = 0
    bend: int = 0
    bend_time: int = 0

    # pitch: int = 60  # default middle c

    def __post_init__(self):
        super().__post_init__()
        self.attack = 0
        self.wave_options = {
            "A1": self.gen_sine_wave,  # analog-style sine wave
            "A2": self.gen_tri_wave,  # analog-style triangle wave
            "A3": self.gen_saw_wave,  # analog-style triangle wave
            "A4": self.gen_square_wave,  # analog-style square wave
            # "A5": self.hp_square_wave,  # high pass filtered square wave
            # 'A6':self.gen_pulse_wave, #analog-style pulse wave
            "D1": self.drum_head_sound,
        }
        self.layer_audio = self.wave_options.get(self.wave, 'A1')()
        # self.duration = self.attack + self.decay
        # self.attack_samples = int(np.ceil(self.attack * self.sample_rate))
        # self.decay_samples = int(self.decay * self.sample_rate)
        # self.num_samples = int(self.attack_samples + self.decay_samples)

    def drum_head_sound(self, delay_sec=0.01, feedback_gain=0.5):
        """
        Simulate a drum head with delay and feedback using waveguide synthesis.

        Args:
            duration_sec (float): Duration of the drum sound in seconds.
            sample_rate_hz (int): Sampling rate in Hz (samples per second).
            delay_sec (float): Delay time in seconds for the feedback loop.
            feedback_gain (float): Feedback gain (0.0 to 1.0) for the delay loop.

        Returns:
            numpy.ndarray: Mono audio waveform as a NumPy array of floats.
        """
        duration_sec = self.duration
        sample_rate_hz = self.sample_rate
        # Constants
        c = 200.0  # Wave propagation speed (arbitrary value for demonstration)
        dt = 1.0 / sample_rate_hz  # Time step (inverse of sample rate)

        # Calculate number of samples
        num_samples = int(duration_sec * sample_rate_hz)

        # Initialize arrays
        audio_data = np.zeros(num_samples)
        delay_samples = int(delay_sec * sample_rate_hz)
        delay_line = np.zeros(delay_samples)

        # Simulation loop
        for t in range(num_samples):
            # Simulate wave propagation
            if t > 0:
                u_next = 2 * audio_data[t - 1] - audio_data[t - 2]
            else:
                u_next = 0.0  # Initial condition

            # Apply delay and feedback
            delayed_sample = delay_line[0]
            output_sample = u_next + feedback_gain * delayed_sample

            # Store output sample
            audio_data[t] = output_sample

            # Update delay line (circular buffer)
            delay_line = np.roll(delay_line, 1)
            delay_line[0] = u_next

        # Normalize audio data
        max_abs_value = np.max(np.abs(audio_data))
        if max_abs_value > 0.0:
            audio_data /= max_abs_value

        return audio_data

    def laplacian(self, u):
        """
        Compute discrete Laplacian (finite difference approximation).

        Args:
            u (numpy.ndarray): 2D grid representing wave displacement.

        Returns:
            numpy.ndarray: Laplacian of the input grid.
        """
        # Compute Laplacian using central difference method
        laplacian_u = (
            -4 * u
            + np.roll(u, 1, axis=0)
            + np.roll(u, -1, axis=0)
            + np.roll(u, 1, axis=1)
            + np.roll(u, -1, axis=1)
        )
        return laplacian_u

    # def frequency_modulation(self, carrier_signal, modulation_signal, mod_freq, mod_amount):
    #     """Generate a frequency-modulated signal (FM)."""
    #     t = np.linspace(
    #         0, self.duration, int(self.sample_rate * self.duration), endpoint=False
    #     )

    #     # Modulation waveform generation based on modulation type

    #     # Carrier signal (sine wave)
    #     # carrier_signal = np.sin(2 * np.pi * carrier_freq * t)

    #     # Frequency modulation (FM) by scaling the carrier frequency with the modulation waveform
    #     fm_signal = np.sin(
    #         2 * np.pi * (carrier_freq + mod_amount * modulation_signal) * t
    #     )

    #     return fm_signal


@dataclass
class VD_GenericLayer(SynthLayer):
    """
    Generates a layer of drum sounds using basic synthesis with a drum-oriented interface based on volca drum.

    Parameters:
    - sample_rate: sample rate in number of individual samples
    - src_type: type of audio signal to be generated as a basis for the drum layer sound
    - mod_type: type of modulation being applied to signal if not specified in src_type
    - env_type: type of envelope to apply to signal: lin, log, double peak
    - level: audio level 0-1 of output signal
    - attack: time in seconds for signal to rise to peak
    - decay: time in seconds for signal to drop to persistent zero after peak
    - frequency: Fundamental frequency of the oscillator or in the case of noise the cutoff filter applied to noise (Hz)
    - mod_amount: amount or amplitude of modulation to effect oscillation signal
    - mod_rate: frequency of modulated signal
    # - detune: offers a detuning function that adds an oscillator of a fix difference frequency specified by this value
    """

    src_type: str = "sine"  # options: 'sine', 'saw', 'hp_noise', 'lp_noise', 'bp_noise'
    mod_type: str = "exp"  # options: 'exp', 'sine', 'noise'
    env_type: str = "lin"  # options: 'lin', 'log', 'dp'
    noise_type: str = "white"  # options: 'white', 'brown', 'pink', 'blue'
    frequency: int = 440
    mod_amount: float = 1.0  # BETWEEN 0-1
    mod_rate: int = 220
    wave_guide_mix: float = 0.0  # BETWEEN 0-1
    wave_decay: float = 0.95  # BETWEEN 0-1
    wave_tone: float = 0.001  # BETWEEN 0-1
    wave_body: float = 0.0  # BETWEEN 0-1

    def __post_init__(self):
        super().__post_init__()
        self.mod_amount = self._wrap_around(self.mod_amount)
        self.wave_guide_mix = self._wrap_around(self.wave_guide_mix)
        self.wave_decay = self._wrap_around(self.wave_decay)
        self.wave_tone = self._wrap_around(self.wave_tone)
        self.wave_body = self._wrap_around(self.wave_body)
        self.gen_layer_sound()

    def gen_layer_sound(self):
        """Runs (or reruns) procedure to generate audio for layer array."""
        self.gen_mod_signal()
        self.gen_carrier_signal()
        self.gen_envelope()
        self.gen_layer()
        self.wave_guide_send()

    def gen_mod_signal(self):
        """
        Create mod signal
        TODO: ADD MOD RATE and CHANGE TO CONDITIONAL
        """
        mod_translate = {
            'exp': self.gen_log_dec_no_att_env,
            'sine': self.gen_sine_wave,
            'noise': self.gen_white_noise,
        }
        self.modulation_signal = mod_translate.get(self.mod_type)() * self.mod_amount

    def gen_carrier_signal(self):
        src_func = {
            "sine": self.gen_sine_wave,
            "saw": self.gen_saw_wave,
            # "rev_saw": self.gen_rev_saw_wave,
            # "square": self.gen_square_wave,
            # "tri": self.gen_tri_wave,
            "hp_noise": self.gen_highpass_noise,
            "lp_noise": self.gen_lowpass_noise,
            "bp_noise": self.gen_bandpass_noise,
        }
        """Generate carrier signal that will be used as basis for drum sounds"""
        self.carrier_signal = src_func.get(self.src_type)()

    def gen_envelope(self):
        envelopes = {
            'lin': self.gen_lin_att_lin_dec_env,
            'log': self.gen_log_att_log_dec_env,
            'dp': self.gen_double_peak_env,
        }
        self.envelope = envelopes.get(self.env_type)()

    def gen_layer(self):
        if self.mod_amount:
            self.layer_audio = (
                self.envelope
                * (
                    (
                        self.modulate_frequency(
                            self.carrier_signal, self.modulation_signal
                        )
                        * self.mod_amount
                    )
                    + ((1 - self.mod_amount) * self.carrier_signal)
                )
                / 2
            )
        else:
            self.layer_audio = self.carrier_signal * self.envelope

    def gen_filtered_noise(self, cutoff_hz, noise_type, filter_type='L1'):
        self.noise_types = {
            "white": self.gen_white_noise,
            "brown": self.gen_brown_noise,
            "pink": self.gen_pink_noise,
            "blue": self.gen_blue_noise,
        }
        noise_signal = self.noise_types.get(noise_type)()
        return self.apply_filter(
            audio_signal=noise_signal, cutoff_hz=cutoff_hz, filter_type=filter_type
        )

    def gen_highpass_noise(self):
        return self.gen_filtered_noise(
            cutoff_hz=self.frequency, noise_type=self.noise_type, filter_type="L2"
        )

    def gen_lowpass_noise(self):
        return self.gen_filtered_noise(
            cutoff_hz=self.frequency, noise_type=self.noise_type, filter_type="H2"
        )

    def gen_bandpass_noise(self):
        return self.gen_filtered_noise(
            cutoff_hz=self.frequency, noise_type=self.noise_type, filter_type="B2"
        )

    def wave_guide_send(self):
        """Sends audio through a wave guide delay to mix it back in"""
        if self.wave_guide_mix:
            wave_guide = self.gen_wave_guide(
                audio_signal=self.layer_audio,
                wave_guide_mix=self.wave_guide_mix,
                decay=self.wave_decay,
                body=self.wave_body,
                tone=self.wave_tone,
            )
            self.layer_audio = (self.layer_audio * (self.wave_guide_send / 2)) + (
                self.layer_audio * ((1 - self.wave_guide_send) / 2)
            )
