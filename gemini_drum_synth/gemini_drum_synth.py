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
    channel_pan: float = 0.5  # 0 is left 1 is right .5 is balanced
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
        """
        Initializes the SoundChannel object after dataclasses run.
        """
        self.layers: list = []  # list of layers of audio samples

    def num_float_to_int(self, v_float, v_int):
        """
        Converts a float value to an integer using multiplication.

        Parameters:
            v_float (float): The float value to convert.
            v_int (int): The integer to multiply with.

        Returns:
            int: The result of the conversion.
        """
        return np.floor(v_float * v_int)

    def add_nd_click_layer_rand(self):
        """
        Adds a random ND_ClickLayer object to the Sound Channel object layer list.
        """
        self.add_nd_click_layer_num(np.random.rand(3))

    def add_nd_click_layer_num(self, values: np.ndarray):
        """
        Adds an ND_ClickLayer object to the Sound Channel object layer list using an array of 3 floating point values.

        Parameters:
            values (np.ndarray): An array of values used to configure the ND_ClickLayer object.

        Returns:
            None
        """

        layer = ND_ClickLayer(
            bit_depth=self.channel_bitdepth,
            sample_rate=self.channel_sample_rate,
            layer_level=values[0],
            click_type={
                0: 'N1',
                1: 'N2',
                2: 'S1',
                3: 'I1',
                4: 'M1',
                5: 'T1',
                6: 'T2',
            }.get(np.floor(values[1] * 7)),
            click_duration=np.abs(np.log(values[2]) / 50) + 0.005,
        )
        self.layers.append(layer)

    def add_nd_click_layer(
        self,
        layer_level: float = 1.0,
        click_type: str = 'N2',
        click_duration: float = 0.01,
    ):
        """
        Adds an ND_ClickLayer object to the Sound Channel object layer list with specified parameters.

        Args:
            layer_level (float, optional): The level of the layer. Defaults to 1.0.
            click_type (str, optional): The type of click. Defaults to 'N2'.
            click_duration (float, optional): The duration of the click. Defaults to 0.01.
        """
        layer = ND_ClickLayer(
            bit_depth=self.channel_bitdepth,
            sample_rate=self.channel_sample_rate,
            layer_level=layer_level,
            click_type=click_type,
            click_duration=click_duration,
        )
        self.layers.append(layer)

    def add_nd_noise_layer_rand(self) -> None:
        """
        Adds a randomized ND_NoiseLayer object to the Sound Channel object layer list.

        Parameters:
            None

        Returns:
            None
        """
        self.add_nd_noise_layer_num(np.random.rand(9))

    def add_nd_noise_layer_num(self, values: np.ndarray) -> None:
        """
        Adds ND_NoiseLayer object to Sound Channel object layer list.

        Parameters:
            values (np.ndarray): Array of 9 random values used to configure the ND_NoiseLayer object.

        Returns:
            None
        """
        layer = ND_NoiseLayer(
            bit_depth=self.channel_bitdepth,
            sample_rate=self.channel_sample_rate,
            layer_level=values[0],
            pitch=self.num_float_to_int(values[1], 120) + 1,
            attack=values[2] ** 3,
            decay=np.abs(np.log(values[3])),
            decay_type={0: 'E', 1: 'L', 2: 'G'}.get(
                self.num_float_to_int(values[4], 3)
            ),  # 3:'P'
            filter_type={0: 'L1', 1: 'L2', 2: 'B1', 3: 'B2', 4: 'H1', 5: 'H2'}.get(
                self.num_float_to_int(values[5], 6)
            ),
            resonance=values[6],
            freq=self.num_float_to_int(values[7], 10000) + 20,
            dynamic_filter=self.num_float_to_int(values[8], 9),
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
    ) -> None:
        """
        Adds ND_NoiseLayer object to Sound Channel object layer list.

        Parameters:
            layer_level (float, optional): The level of the layer. Defaults to 1.0.
            pitch (int, optional): The pitch of the noise. Defaults to 0.
            attack (float, optional): The attack time. Defaults to 0.025.
            decay (float, optional): The decay time. Defaults to 1.0.
            decay_type (str, optional): The type of decay. Defaults to 'E'.
            filter_type (str, optional): The type of filter. Defaults to 'L2'.
            resonance (float, optional): The resonance of the filter. Defaults to 0.5.
            freq (int, optional): The frequency of the noise. Defaults to 120.
            dynamic_filter (int, optional): The dynamic filter type. Defaults to 5.

        Returns:
            None
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
    ) -> None:
        """
        Adds VD_GenericLayer object to Sound Channel object layer list.

        Parameters:
            layer_level (float): The level of the layer.
            pitch (int): The pitch of the layer.
            attack (float): The attack time.
            decay (float): The decay time.
            src_type (str): The source type.
            mod_type (str): The modulation type.
            env_type (str): The envelope type.
            noise_type (str): The noise type.
            frequency (int): The frequency of the layer.
            mod_amount (float): The modulation amount.
            mod_rate (int): The modulation rate.
            wave_guide_mix (float): The wave guide mix.
            wave_decay (float): The wave decay.
            wave_tone (float): The wave tone.
            wave_body (float): The wave body.

        Returns:
            None
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

    def _mono_to_stereo(self) -> None:
        """
        Convert mono audio to stereo.

        Parameters:
            None

        Returns:
            None
        """
        if not self.channel_pan - 0.5:
            self.channel_audio = np.tile(self.layer_audio, (2, 1)).T

    def mix_signals(self, arr1, arr2):
        """
        Mixes signals (in form of numpy arrays) of different lengths and returns an array of the length of the longest.

        Parameters:
            arr1 (np.ndarray): The first array of audio signals.
            arr2 (np.ndarray): The second array of audio signals.

        Returns:
            np.ndarray: The mixed array of audio signals.
        """
        l = sorted((arr1, arr2), key=len)
        c = l[1].copy()
        c[: len(l[0])] += l[0]
        return c

    def mix_layers(self) -> None:
        """
        Mixes audio samples from all layers and generates the final track.

        Parameters:
            None

        Returns:
            None
        """
        for layer in self.layers:
            # Generate audio sample for the layer (e.g., using generate_sample method)
            layer.gen_layer_sound()
            layer.apply_level()
            self.channel_audio = self.mix_signals(layer.layer_audio, self.channel_audio)

        # Mix audio sample into the final track (e.g., add to list)
        # self.channel_audio

    def save_channel_audio(self, filename):
        """
        Save the audio signal to a WAV file.

        Parameters:
            filename (str): The name of the output WAV file.

        Returns:
            None
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
    """
    SynthLayer: Generates a layer of a percussive synthesizer.

    Attributes:
        layer_audio (np.array): Audio data for the layer.
        filepath (Path): Path to the directory where samples are stored.
        num_channels (int): Number of audio channels.
        bit_depth (int): Bit depth of the audio data.
        sample_rate (int): Sampling rate of the audio data.
        layer_level (float): Volume level of the layer (between 0 and 1).
        pitch (int): Pitch adjustment of the sound in semitones.
        attack (float): Duration of the attack of the synthesized sound in seconds.
        decay (float): Duration of the decay of the synthesized sound in seconds.
        attack_samples (int): Number of attack samples calculated based on the attack duration and sample rate.
        decay_samples (int): Number of decay samples calculated based on the decay duration and sample rate.
        num_samples (int): Total number of samples in the layer.
        att_t (np.array): Time domain array for the attack envelope.
        dec_t (np.array): Time domain array for the decay envelope.
        env_t (np.array): Time domain array for the overall envelope.

    Examples:
        >>> layer = SynthLayer(layer_audio=np.zeros(44100), filepath=Path.home() / "drumsynth/samples")
        >>> layer.save_layer(filename='my_layer')
    """

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
        """
        Initialize the SynthLayer object after __init__.

        Parameters:
            None

        Returns:
            None
        """
        self._level_wrap_around()
        self.attack = (
            self.attack if self.attack else 0.00001
        )  # SET ATTACK TO NON-ZERO VALUE TO AVOID DIVIDE BY ZERO ERRORS
        self.prep_layer()

    def __str__(self):
        """
        Return a string representation of the SynthLayer object.

        Parameters:
            None

        Returns:
            str: String representation of the SynthLayer object.
        """
        return self._create_long_layer_desc()

    def __repr__(self):
        """
        Return a string representation of the SynthLayer object.

        Parameters:
            None

        Returns:
            str: String representation of the SynthLayer object.
        """
        return self._create_long_layer_desc()

    def prep_layer(self):
        """
        Prepare the layer by calculating envelope parameters.

        Parameters:
            None

        Returns:
            None
        """
        self.attack_samples = int(np.floor(self.attack * self.sample_rate))
        self.decay_samples = int(np.floor(self.decay * self.sample_rate))
        # self.duration = self.attack + self.decay
        self.num_samples = int(self.attack_samples + self.decay_samples)
        self.att_t = self._gen_t(num_samples=self.attack_samples)
        self.dec_t = self._gen_t(num_samples=self.decay_samples)
        self.env_t = self._gen_t(num_samples=self.num_samples)

    ####OUTPUT_METHODS####
    def apply_level(self):
        self.layer_audio *= self.layer_level

    def save_layer(self, filename='', filetypes=['wav']):
        """
        Save the audio signal to a WAV file.

        Parameters:
            filename (str): Name of the output WAV file. If not provided, a default name will be generated.
            TODO filetypes (list): List of strings containing the extensions that will be exported
        Returns:
            None
        """
        if not filename:
            filename = self._create_short_layer_desc()

        # Scale audio to 16-bit integer range (-32768 to 32767)
        audio_int = (self.layer_audio * 32767).astype(np.int16)

        # Save the audio to a WAV file
        wavfile.write(self.filepath / f"{filename}.wav", self.sample_rate, audio_int)

    def _create_short_layer_desc(self):
        """
        Create a shortened description of the layer.

        Parameters:
            None

        Returns:
            str: Shortened description of the layer.
        """
        return '_'.join(
            [
                f'{"".join([l[0] if l else "" for l in x.split("_")])}-{y}'
                for x, y in self.__dict__.items()
                if not isinstance(y, (Path, np.ndarray))
            ]
        )

    def _create_long_layer_desc(self):
        """
        Create a long description of the SynthLayer object.

        Parameters:
            None

        Returns:
            str: Long description of the SynthLayer object.
        """
        return ', '.join(
            [
                f'{x}: {y}'
                for x, y in self.__dict__.items()
                if not isinstance(y, (Path, np.ndarray))
            ]
        )

    def print_it_all(self, long=True):
        """
        Print a description of the layer.

        Parameters:
            long (bool): If True, print a detailed description; if False, print a shortened description.

        Returns:
            None
        """
        if long:
            print(self._create_long_layer_desc())
        else:
            print(self._create_short_layer_desc())

    ####HELPER_FUNCTIONS####

    def _wrap_around(self, setting):
        """
        Keep a value within 0 to 1.

        Parameters:
            setting (float): Input value to wrap around.

        Returns:
            float: Wrapped value within 0 to 1.
        """

        if setting > 1:
            return abs(setting) % 1.0
        else:
            return setting

    def _level_wrap_around(self) -> None:
        """
        Wrap around the layer level value to ensure it falls within the range [0, 1].

        Parameters:
            None

        Returns:
            None
        """
        self.layer_level = self._wrap_around(self.layer_level)

    def _gen_t(self, num_samples=441) -> np.ndarray:
        """
        Generate a time domain array based on the number of samples.

        Parameters:
            num_samples (int): Number of samples.

        Returns:
            np.array: Time domain array.
        """
        return np.linspace(0, 1, num_samples, endpoint=True)

    def _translate_filter(self, filter_type="L2"):
        """
        Translates filter type values to uncalled filter methods that can be called with settings locally.

        Parameters:
            filter_type (str): Type of filter to translate (e.g., 'L1', 'L2', 'H1', 'H2', 'B1', 'B2'). Defaults to 'L2'.

        Returns:
            Callable: A callable method corresponding to the specified filter type.
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
        Mixes signals (in the form of numpy arrays) of different lengths and returns an array of the length of the longest.

        Parameters:
            arr1 (np.ndarray): First array of audio signals.
            arr2 (np.ndarray): Second array of audio signals.

        Returns:
            np.ndarray: Mixed array of audio signals.
        """
        l = sorted((arr1, arr2), key=len)
        c = l[1].copy()
        c[: len(l[0])] += l[0]
        return c

    def midi_note_to_freq(self, pitch):
        """
        Convert MIDI note value to frequency.

        Parameters:
            pitch (int): MIDI note value.

        Returns:
            float: Corresponding frequency in Hz.
        """
        return 2 ** ((pitch - 68) / 12) * 440

    def pitch_change_semitones(self, audio_signal, semitones):
        """
        Change the pitch of an audio signal by a specified number of semitones.

        Parameters:
            audio_signal (ndarray): Input audio signal as a NumPy array.
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
        Normalize audio signal to between -1 and 1 by dividing array by max absolute value in the array.

        Parameters:
            signal (np.ndarray): Input audio signal as a NumPy array.

        Returns:
            np.ndarray: Normalized audio signal between -1 and 1.
        """
        signal /= np.max(np.abs(signal))
        return signal

    def min_max_normalization(self, signal):
        """
        Normalize audio signal using min-max normalization.

        Parameters:
            signal (np.ndarray): Input audio signal as a NumPy array.

        Returns:
            np.ndarray: Normalized audio signal using min-max normalization.
        """
        return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

    def normalize_audio(self, signal):
        """
        Normalize audio signal to between -1 and 1 using min-max normalization scaled.

        Parameters:
            signal (np.ndarray): Input audio signal as a NumPy array.

        Returns:
            np.ndarray: Normalized audio signal between -1 and 1.
        """
        return 2 * self.min_max_normalization(signal) - 1

    ####WAVE_OSCILLATORS####

    def gen_sine_wave(self, freq_override=0):
        """
        Generate a sine wave audio signal.

        Parameters:
            freq_override (float): Frequency of the sine wave in Hz (default: 0, uses self.frequency).

        Returns:
            ndarray: Generated sine wave audio signal.
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
            freq_override (float): Frequency of the triangle wave in Hz (default: 0, uses self.frequency).

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
        """
        Generate a sawtooth wave audio signal.

        Parameters:
            freq_override (float): Frequency of the sawtooth wave in Hz (default: 0, uses self.frequency).

        Returns:
            ndarray: Generated sawtooth wave audio signal.
        """
        if not freq_override:
            freq_override = self.frequency
        # t = np.linspace(0, self.duration, self.num_samples, endpoint=False)
        sawtooth_wave = 2 * (
            freq_override * self.env_t - np.floor(self.frequency * self.env_t + 0.5)
        )
        sawtooth_wave /= np.max(np.abs(sawtooth_wave))
        return sawtooth_wave

    def gen_rev_saw_wave(self, freq_override=0):
        """
        Generate a reverse sawtooth wave audio signal.

        Parameters:
            freq_override (float): Frequency of the reverse sawtooth wave in Hz (default: 0, uses self.frequency).

        Returns:
            ndarray: Generated reverse sawtooth wave audio signal.
        """
        rev_saw_wave = self.gen_saw_wave(freq_override) * -1

    def gen_square_wave(self, freq_override=0):
        """
        Generate a square wave audio signal.

        Parameters:
            freq_override (float): Frequency of the square wave in Hz (default: 0, uses self.frequency).

        Returns:
            ndarray: Generated square wave audio signal.
        """
        if not freq_override:
            freq_override = self.frequency
        # t = np.linspace(0, self.duration, self.num_samples, endpoint=False)
        square_wave = np.sign(np.sin(2 * np.pi * freq_override * self.env_t))
        return square_wave

    ####CLICK_SOUND_GENERATORS####

    def gen_click(self, click_type, click_duration):
        """
        Generate a click sound waveform of the specified type. Based loosely on the Nord Drum Click Sound Layer.

        Parameters:
            click_type (str): Type of click sound ('S1', 'N1', 'N2', 'I1', 'M1', 'T1', 'T2').
            click_duration (float): Duration of the click sound in seconds.

        Raises:
            ValueError: If an invalid click_type is provided.

        Returns:
            None
        """
        self.num_click_samples = int(self.sample_rate * click_duration)
        click_t = self._gen_t(self.num_click_samples)
        click_env = self.gen_click_env(click_t=click_t)

        ### !!! #### WHEN ADDING TO THIS LIST CHANGE sound channel numeric input as well ####

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
            None

        Returns:
            ndarray: Generated Karplus-Strong plucked string sound waveform.
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
        Generate white noise audio signal the length of duration of the sample.

        Parameters:
            None

        Returns:
            ndarray: Generated white noise audio signal.
        """
        return np.random.normal(scale=1, size=self.num_samples)

    def gen_pink_noise(self):
        """
        Generate pink noise audio signal (1/f noise) the length of duration of the sample.

        Parameters:
            None

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
        Generate brown noise audio signal the length of duration of the sample.

        Parameters:
            None

        Returns:
            ndarray: Generated brown noise audio signal.
        """
        brown_noise = np.random.randn(self.num_samples).cumsum()

        # Normalize brown noise to stay within -1 to 1
        brown_noise /= np.max(np.abs(brown_noise))

        return brown_noise

    def gen_blue_noise(self):
        """
        Generate blue noise (azure noise) audio signal the legnth of the duration of the sample.

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
        """
        Generate a logarithmic decay envelope.

        Parameters:
            degree (int): Degree of logarithmic decay (default: 50).

        Returns:
            ndarray: Generated logarithmic decay envelope.
        """
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
        """
        Generate a logarithmic attack envelope.

        Parameters:
            degree (int): Degree of logarithmic attack (default: 50).

        Returns:
            ndarray: Generated logarithmic attack envelope.
        """
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
        """
        Generate a linear attack envelope.

        Returns:
            ndarray: Generated linear attack envelope.
        """
        if not attack_samples:
            attack_samples = self.attack_samples
        return np.linspace(0, 1, attack_samples)

    def gen_lin_decay(self, decay_samples=0):
        """
        Generate a linear decay envelope.

        Returns:
            ndarray: Generated linear decay envelope.
        """
        if not decay_samples:
            decay_samples = self.decay_samples
        return np.linspace(1, 0, decay_samples)

    ####ENVELOPE_GENERATORS####

    def gen_click_env(self, click_t):
        """
        Generate a simple exponential click envelope.

        Parameters:
            click_t (ndarray): Time array for the click envelope.

        Returns:
            ndarray: Exponential click envelope.
        """
        return np.exp(-5 * click_t)

    def gen_lin_att_lin_dec_env(self):
        """
        Generate a linear attack / linear decay envelope.

        Returns:
            ndarray: Generated envelope.
        """
        return self.gen_lin_env()

    def gen_log_dec_no_att_env(self):
        """
        Generate a logarithmic decay envelope without attack.

        Returns:
            ndarray: Generated envelope.
        """
        return self.gen_log_decay(decay_samples=self.num_samples)

    def gen_lin_att_log_dec_env(self, degree=50):
        """
        Generate a linear attack / logarithmic decay envelope.

        Parameters:
            degree (int): Degree of logarithmic decay (default: 50).

        Returns:
            ndarray: Generated envelope.
        """
        rise = self.gen_lin_attack()
        fall = self.gen_log_decay(degree=degree)
        envelope = np.concatenate([rise, fall])
        self.num_samples = rise.shape[0] + fall.shape[0]
        return envelope[: self.num_samples]

    def gen_log_att_log_dec_env(self, degree=50):
        """
        Generate a logarithmic attack / logarithmic decay envelope.

        Parameters:
            degree (int): Degree of logarithmic attack and decay (default: 50).

        Returns:
            ndarray: Generated envelope.
        """
        rise = self.gen_log_attack(degree=degree)
        fall = self.gen_log_decay(degree=degree)
        envelope = np.concatenate([rise, fall])
        self.num_samples = rise.shape[0] + fall.shape[0]
        return envelope[: self.num_samples]

    def gen_lin_env(self):
        """
        Generate a linear attack / linear decay envelope.

        Returns:
            ndarray: Generated envelope.
        """
        rise = self.gen_lin_attack()
        fall = self.gen_lin_decay()
        envelope = np.concatenate([rise, fall])
        self.num_samples = rise.shape[0] + fall.shape[0]
        return envelope[: self.num_samples]

    def gen_double_peak_env(self, seperation=0.1, degree=50):
        """
        Generate a double peak envelope.

        Parameters:
            separation (float): Separation between peaks as a fraction of the total length (default: 0.1).
            degree (int): Degree of logarithmic decay (default: 50).

        Returns:
            ndarray: Generated envelope.
        """
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
        """
        Generate a gate envelope.

        Returns:
            ndarray: Gate envelope.
        """
        return np.ones(self.num_samples)

    def gen_punch_decay(self):
        """
        Generate a punchy decay envelope.

        Returns:
            ndarray: Generated envelope.
        """
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
        """
        CONSIDER REMOVING !!!!!!!!
        NOT WORKING WELL !!!!!!!!!!
        Apply waveguide synthesis to the audio signal.

        Parameters:
            audio_signal (ndarray): Input audio signal as a NumPy array.
            wave_guide_mix (float): Waveguide mix parameter (0 to 1).
            decay (float): Feedback decay for the waveguide.
            body (float): Body of the waveguide (low-pass filter).
            tone (float): Delay time for the waveguide in seconds.

        Returns:
            ndarray: Processed audio signal after applying waveguide synthesis.
        """
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
        """
        CONSIDER REMOVING !!!!!!!!
        NOT WORKING WELL !!!!!!!!!!
        Apply waveguide synthesis (alternate method) to the audio signal.

        Parameters:
            audio_signal (ndarray): Input audio signal as a NumPy array.
            wave_guide_mix (float): Waveguide mix parameter (0 to 1).
            decay (float): Feedback decay for the waveguide.
            body (float): Body of the waveguide (low-pass filter).
            tone (float): Delay time for the waveguide in seconds.

        Returns:
            ndarray: Processed audio signal after applying waveguide synthesis.
        """
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
        Apply waveguide synthesis using Pedalboard to the audio signal.

        Parameters:
            audio_signal (ndarray): Input audio signal as a NumPy array.
            wave_guide_mix (float): Waveguide mix parameter (0 to 1).
            decay (float): Feedback decay for the waveguide.
            body (float): Body of the waveguide (low-pass filter).
            tone (float): Delay time for the waveguide in seconds.

        Returns:
            ndarray: Processed audio signal after applying waveguide synthesis.
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
    """A layer for generating transient click sounds used in percussion.

    Parameters:
            click_type (str): The type of click sound to generate ('N1', 'N2', 'S1', 'I1', 'M1', 'T1', 'T2').
            click_duration (float): The duration of the click sound in seconds.

        Returns:
            None
    """

    click_type: str = 'N1'
    click_duration: float = 0.01  # typically range from .005 - .05 ms

    def __post_init__(self):
        """
        Initializes the ND_ClickLayer instance after the base SynthLayer has been initialized.

        Parameters:
            click_type (str): The type of click sound to generate ('N1', 'N2', 'S1', 'I1', 'M1', 'T1', 'T2').
            click_duration (float): The duration of the click sound in seconds.

        Returns:
            None
        """
        self.attack = self.click_duration
        super().__post_init__()
        # self.gen_click(click_type=self.click_type, click_duration=self.click_duration)
        self.gen_layer_sound()

    def gen_layer_sound(self):
        """
        Generate the audio sample for the ND_ClickLayer.

        This method generates a transient click sound based on the specified click_type and click_duration.

        Parameters:
            None

        Returns:
            None
        """
        self.gen_click(click_type=self.click_type, click_duration=self.click_duration)


@dataclass
class ND_NoiseLayer(SynthLayer):
    """
    Drum Sound Layer based on the Noise Layer Synthesis in Nord Drum 3P.

    Attributes:
        filter_type (str): Type of filter for the noise layer ('L1', 'L2', 'H1', 'H2', 'B1', 'B2').
        resonance (int): Resonance of the filter (0 to 20).
        freq (int): Cutoff frequency of the filter in Hz.
        dynamic_filter (int): Dynamic filter value (-9 to 9).
        decay_type (str): Type of decay for the noise envelope ('E' for logarithmic, 'L' for linear, 'G' for gate, 'P' for punch).

    Examples:
        >>> layer = ND_NoiseLayer(filter_type='L2', resonance=10, freq=500, dynamic_filter=3, decay_type='E')
    """

    # noise_type: str = 'white'
    filter_type: str = 'L2'  # default low pass 4 pole
    resonance: int = 0  # max 20
    freq: int = 200  # cutoff frequency in Hz
    dynamic_filter: int = 0  # plus or minus 9
    # decay: int = 0  # max 50
    decay_type: str = "E"  # 'E': 'log','L': 'lin', 'G': 'gate', 'P': 'punch'
    # drive: float = 1.0

    def __post_init__(self):
        """
        Initialize the ND_NoiseLayer after the base SynthLayer has been initialized.

        This method sets the attack to 0 and generates the layer sound.

        Parameters:
            None

        Returns:
            None
        """
        self.attack = 0
        super().__post_init__()
        # print('does this print')
        # print(self.num_samples)
        # self.duration = self.attack + self.decay
        self.gen_layer_sound()

    def gen_layer_sound(self):
        """
        Generate the audio sample for the ND_NoiseLayer.

        This method sets up the noise layer by translating decay type,
        generating white noise, filtering the noise, and applying the noise envelope.

        Parameters:
            None

        Returns:
            None
        """
        self.filter_mode = self._translate_filter(self.filter_type)
        self.noise_decay_envelope = self._translate_decay()()
        self.layer_audio = self.gen_white_noise()
        self.filter_noise()
        self.apply_noise_envelope()

    def _translate_decay(self):
        """
        Translate decay type string to corresponding envelope generation method.

        Returns:
            Callable: Envelope generation method based on decay type.
        """
        return {
            'E': self.gen_lin_att_log_dec_env,
            'L': self.gen_lin_att_lin_dec_env,
            'G': self.gen_gate_env,
            'P': self.gen_punch_decay,
        }.get(self.decay_type)

    def filter_noise(self):
        """
        Generate a white noise sound waveform, and apply filtering to the noise layer.

        This method uses a Pedalboard with a LadderFilter to process the noise audio.

        Parameters:
            None

        Returns:
            None
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
        """
        Apply the noise envelope to the noise layer.

        This method applies the generated noise decay envelope to the noise audio.

        Parameters:
            None

        Returns:
            None
        """
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

    Attributes:
        src_type (str): Type of audio signal to be generated as a basis for the drum layer sound.
        mod_type (str): Type of modulation being applied to the signal if not specified in src_type.
        env_type (str): Type of envelope to apply to the signal ('lin', 'log', 'dp').
        noise_type (str): Type of noise ('white', 'brown', 'pink', 'blue').
        frequency (int): Fundamental frequency of the oscillator or cutoff frequency in Hz for noise.
        mod_amount (float): Amount or amplitude of modulation to affect oscillation signal (0-1).
        mod_rate (int): Frequency of the modulated signal.
        wave_guide_mix (float): Mix amount for the wave guide delay (0-1).
        wave_decay (float): Decay parameter for the wave guide delay (0-1).
        wave_tone (float): Tone parameter for the wave guide delay (0-1).
        wave_body (float): Body parameter for the wave guide delay (0-1).

    Methods:
        __post_init__(): Initializes the VD_GenericLayer after the base SynthLayer has been initialized.
        gen_layer_sound(): Generates the audio sample for the VD_GenericLayer.
        gen_mod_signal(): Creates the modulation signal based on mod_type.
        gen_carrier_signal(): Generates the carrier signal based on src_type.
        gen_envelope(): Generates the envelope based on env_type.
        gen_layer(): Generates the layer audio by applying modulation and envelope to the carrier signal.
        gen_filtered_noise(cutoff_hz, noise_type, filter_type='L1'): Generates filtered noise.
        gen_highpass_noise(): Generates high-pass filtered noise.
        gen_lowpass_noise(): Generates low-pass filtered noise.
        gen_bandpass_noise(): Generates band-pass filtered noise.
        wave_guide_send(): Sends audio through a wave guide delay and mixes it back in.

    Examples:
        >>> layer = VD_GenericLayer(src_type='sine', mod_type='exp', env_type='lin', noise_type='white', frequency=440, mod_amount=0.5)
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
        """
        Initialize the VD_GenericLayer after the base SynthLayer has been initialized.

        This method wraps around certain attributes, sets default values, and generates the layer sound.

        Parameters:
            None

        Returns:
            None
        """
        super().__post_init__()
        self.mod_amount = self._wrap_around(self.mod_amount)
        self.wave_guide_mix = self._wrap_around(self.wave_guide_mix)
        self.wave_decay = self._wrap_around(self.wave_decay)
        self.wave_tone = self._wrap_around(self.wave_tone)
        self.wave_body = self._wrap_around(self.wave_body)
        self.gen_layer_sound()

    def gen_layer_sound(self):
        """
        Generate the audio sample for the VD_GenericLayer after resetting attributes.

        This method calls other methods to generate modulation signal, carrier signal, envelope,
        and then generates the layer audio.

        Parameters:
            None

        Returns:
            None
        """
        self.gen_mod_signal()
        self.gen_carrier_signal()
        self.gen_envelope()
        self.gen_layer()
        self.wave_guide_send()

    def gen_mod_signal(self):
        """
        Create the modulation signal based on mod_type.
        This method selects a modulation signal generation function based on mod_type.
        TODO FIX FM TECHNIQUE

        Parameters:
            None

        Returns:
            None
        """
        mod_translate = {
            'exp': self.gen_log_dec_no_att_env,
            'sine': self.gen_sine_wave,
            'noise': self.gen_white_noise,
        }
        self.modulation_signal = mod_translate.get(self.mod_type)() * self.mod_amount

    def gen_carrier_signal(self):
        """
        Generate the carrier signal based on src_type.

        This method selects a carrier signal generation function based on src_type.

        Parameters:
            None

        Returns:
            None
        """
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
        self.carrier_signal = src_func.get(self.src_type)()

    def gen_envelope(self):
        """
        Generate the envelope based on env_type.

        This method selects an envelope generation function based on env_type.

        Parameters:
            None

        Returns:
            None
        """
        envelopes = {
            'lin': self.gen_lin_att_lin_dec_env,
            'log': self.gen_log_att_log_dec_env,
            'dp': self.gen_double_peak_env,
        }
        self.envelope = envelopes.get(self.env_type)()

    def gen_layer(self):
        """
        Generate the layer audio by applying modulation and envelope to the carrier signal.

        This method applies modulation and envelope to the carrier signal to generate the final layer audio.

        Parameters:
            None

        Returns:
            None
        """
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

    def gen_filtered_noise(
        self, cutoff_hz: int, noise_type: str, filter_type: str = 'L1'
    ):
        """
        Generate filtered noise.

        This method generates noise of specified type and applies filtering based on cutoff frequency and filter type.

        Parameters:
            cutoff_hz (int): Cutoff frequency in Hz for the filter.
            noise_type (str): Type of noise ('white', 'brown', 'pink', 'blue').
            filter_type (str): Type of filter to apply ('L1', 'L2', 'H1', 'H2', 'B1', 'B2').

        Returns:
            ndarray: NumPy array containing the generated filtered noise.
        """
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
        """
        Generate high-pass filtered noise.

        This method generates high-pass filtered noise based on the specified frequency and noise type.

        Parameters:
            None

        Returns:
            ndarray: NumPy array containing the generated high-pass filtered noise.
        """
        return self.gen_filtered_noise(
            cutoff_hz=self.frequency, noise_type=self.noise_type, filter_type="L2"
        )

    def gen_lowpass_noise(self):
        """
        Generate low-pass filtered noise.

        This method generates low-pass filtered noise based on the specified frequency and noise type.

        Parameters:
            None

        Returns:
            ndarray: NumPy array containing the generated low-pass filtered noise.
        """
        return self.gen_filtered_noise(
            cutoff_hz=self.frequency, noise_type=self.noise_type, filter_type="H2"
        )

    def gen_bandpass_noise(self):
        """
        Generate band-pass filtered noise.

        This method generates band-pass filtered noise based on the specified frequency and noise type.

        Parameters:
            None

        Returns:
            ndarray: NumPy array containing the generated band-pass filtered noise.
        """
        return self.gen_filtered_noise(
            cutoff_hz=self.frequency, noise_type=self.noise_type, filter_type="B2"
        )

    def wave_guide_send(self):
        """
        Send audio through a wave guide delay and mix it back in.

        This method applies wave guide delay to the layer audio and mixes it back in based on wave guide parameters.

        Parameters:
            None

        Returns:
            None
        """
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
