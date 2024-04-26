# def gen_highpass_blue_noise(self):
#     return self.gen_filtered_noise(
#         noise_type="blue", filter_type="high", cutoff=self.frequency
#     )

# def gen_highpass_pink_noise(self):
#     return self.gen_filtered_noise(
#         noise_type="pink", filter_type="high", cutoff=self.frequency
#     )

# def gen_highpass_brown_noise(self):
#     return self.gen_filtered_noise(
#         noise_type="brown", filter_type="high", cutoff=self.frequency
#     )

# def gen_lowpass_white_noise(self):
#     return self.gen_filtered_noise(
#         noise_type="white", filter_type="low", cutoff=self.frequency
#     )

# def gen_lowpass_blue_noise(self):
#     return self.gen_filtered_noise(
#         noise_type="blue", filter_type="low", cutoff=self.frequency
#     )

# def gen_lowpass_pink_noise(self):
#     return self.gen_filtered_noise(
#         noise_type="pink", filter_type="low", cutoff=self.frequency
#     )

# def gen_lowpass_brown_noise(self):
#     return self.gen_filtered_noise(
#         noise_type="brown", filter_type="low", cutoff=self.frequency
#     )

# def gen_bandpass_white_noise(self):
#     return self.gen_filtered_noise(
#         noise_type="white", filter_type="band", cutoff=self.frequency
#     )

# def gen_bandpass_blue_noise(self):
#     return self.gen_filtered_noise(
#         noise_type="blue", filter_type="band", cutoff=self.frequency
#     )

# def gen_bandpass_pink_noise(self):
#     return self.gen_filtered_noise(
#         noise_type="pink", filter_type="band", cutoff=self.frequency
#     )

# def gen_bandpass_brown_noise(self):
#     return self.gen_filtered_noise(
#         noise_type="brown", filter_type="band", cutoff=self.frequency
#     )


# @dataclass
# class LayerEffects:

# def drum_head_sound(self, strike_position=0, head_size=.2):
#     """
#     Simulates a drum sound using waveguide synthesis.

#     Args:
#         pitch: The pitch of the drum in Hz.
#         strike_position: A value between 0 and 1 representing the position of the strike
#                         relative to the center of the drum head (0 = center, 1 = edge).
#         head_size: The diameter of the drum head in meters.
#         sample_rate: The desired sample rate of the audio in Hz (default: 44100).
#         duration: The desired duration of the sound in seconds (default: 1).

#     Returns:
#         A NumPy array containing the audio waveform and the sample rate.
#     """

#     # Calculate wavelength based on pitch
#     wavelength = self.sample_rate / self.frequency

#     # Calculate delay based on strike position and head size
#     delay_samples = int(strike_position * head_size * self.sample_rate / wavelength)

#     # Initialize the audio buffer
#     audio_buffer = np.zeros(self.num_samples)

#     # Impulse at the strike position
#     audio_buffer[delay_samples] = 1
#     noise_duration = 0.01  # seconds
#     noise_samples = int(noise_duration * self.sample_rate)
#     noise = np.random.rand(noise_samples) - 0.5

#     # Apply an envelope to the noise burst
#     envelope = np.linspace(1, 0, noise_samples)  # Fade in and out

#     # Apply noise to the audio buffer at the strike position
#     audio_buffer[delay_samples:delay_samples + noise_samples] += noise * envelope

#     # Two-tap waveguide filter coefficients
#     a1 = -0.95
#     a2 = 0.9

#     # Apply waveguide filter
#     for n in range(delay_samples + 1, self.num_samples):
#         audio_buffer[n] = a1 * audio_buffer[n - 1] + a2 * audio_buffer[n - delay_samples - 1]

#     # Normalize and scale audio
#     audio_buffer = audio_buffer / np.max(np.abs(audio_buffer))
#     audio_buffer *= 0.3

#     return audio_buffer


# def laplacian(self, u):
#     """
#     Compute discrete Laplacian (finite difference approximation).

#     Args:
#         u (numpy.ndarray): 2D grid representing wave displacement.

#     Returns:
#         numpy.ndarray: Laplacian of the input grid.
#     """
#     # Compute Laplacian using central difference method
#     laplacian_u = (
#         -4 * u
#         + np.roll(u, 1, axis=0)
#         + np.roll(u, -1, axis=0)
#         + np.roll(u, 1, axis=1)
#         + np.roll(u, -1, axis=1)
#     )
#     return laplacian_u

# def simulate_drum_head(self):
#     """
#     Simulate a drum head using waveguide synthesis.

#     Args:
#         duration_sec (float): Duration of the drum sound in seconds.
#         sample_rate_hz (int): Sampling rate in Hz (samples per second).

#     Returns:
#         numpy.ndarray: Mono audio waveform as a NumPy array of floats.
#     """
#     # Constants
#     Lx = 40  # Grid size along x-axis
#     Ly = 40  # Grid size along y-axis
#     c = 200  # Wave propagation speed (arbitrary value for demonstration)
#     dt = 1.0 / self.sample_rate  # Time step (inverse of sample rate)

#     # Initialize grid and state variables
#     u = np.zeros((Lx, Ly))  # Displacement grid
#     u_prev = np.zeros((Lx, Ly))  # Previous time step

#     # # Simulation parameters
#     # duration_samples = int(self.duration * self.sample_rate)
#     # num_frames = duration_samples  # Total frames to simulate

#     # Simulation loop
#     audio_data = []
#     for frame in range(self.num_samples):
#         # Update wave equation (finite difference method)
#         # u_next = 2 * u - u_prev + (c ** 2) * self.laplacian(u)
#         u_next = 2 * u - u_prev + (c ** 2) * self.laplacian(u) * (dt ** 2)

#         # Boundary conditions (simple fixed boundary)
#         u_next[:, 0] = 0  # Left boundary (clamp)
#         u_next[:, -1] = 0  # Right boundary (clamp)
#         u_next[0, :] = 0  # Top boundary (clamp)
#         u_next[-1, :] = 0  # Bottom boundary (clamp)

#         # Output from the center of the grid (simulated drum head)
#         audio_data.append(u_next[Lx // 2, Ly // 2])

#         # Update state for the next time step
#         u_prev = u
#         u = u_next

#     # Normalize audio data and convert to NumPy array
#     audio_data = np.array(audio_data)
#     # / np.max(np.abs(audio_data))

#     return audio_data

# def gen_simulated_drum_head_sound(self, Lx=.5, Ly=.5, Nx=40, Ny=40, T=None, dt=.01, c=340., damping_coeff=.001):
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
#     if not T:
#         T = self.duration
#     # Initialize grid
#     x = np.linspace(0, Lx, Nx)
#     y = np.linspace(0, Ly, Ny)
#     X, Y = np.meshgrid(x, y)

#     # Initialize displacement field
#     u = np.zeros((Ny, Nx))  # Current displacement field
#     u_prev = np.zeros_like(u)  # Previous displacement field

#     # Function to apply boundary conditions (fixed edges)
#     def apply_boundary_conditions(u):
#         u[:, 0] = 0.0  # Left boundary (fixed)
#         u[:, -1] = 0.0  # Right boundary (fixed)
#         u[0, :] = 0.0  # Bottom boundary (fixed)
#         u[-1, :] = 0.0  # Top boundary (fixed)

#     # Simulation loop
#     num_steps = int(T / dt)
#     displacement_history = []

#     for step in range(num_steps):
#         # Apply boundary conditions
#         apply_boundary_conditions(u)

#         # Update displacement field using finite difference method (wave equation with damping)
#         u_next = (
#             2 * u
#             - u_prev
#             + (c**2 * dt**2)
#             * (
#                 np.roll(u, 1, axis=0)
#                 + np.roll(u, -1, axis=0)
#                 + np.roll(u, 1, axis=1)
#                 + np.roll(u, -1, axis=1)
#                 - 4 * u
#             )
#             - 2 * damping_coeff * dt * (u - u_prev)
#         )

#         # Store current displacement field
#         displacement_history.append(u.copy())

#         # Update displacement fields for next time step
#         u_prev = u
#         u = u_next

#     # Convert displacement history to numpy array
#     displacement_history = np.array(displacement_history)

#     return displacement_history

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
#         def gen_sine_wave(self, freq_override=0):
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
