import numpy as np

# from gemini_drum_synth.gemini_drum_synth import SoundChannel as sc, VD_GenericLayer as vd, ND_ClickLayer as cl, ND_NoiseLayer as nl
from gemini_drum_synth.gemini_drum_synth import SynthLayer as sl

####OUTPUT_METHODS####
def test_apply_level():
    sl_al = sl()

    sl_al.layer_audio = sl_al.gen_sine_wave(freq_override=500)
    assert np.max(sl_al.layer_audio) >= 0.9
    assert np.min(sl_al.layer_audio) <= -0.9
    # sl_al.layer_audio = np.ones(sl_al.num_samples)
    sl_al.layer_level = 0.1
    sl_al.apply_level()
    # print(sl_al.layer_audio)
    assert np.max(sl_al.layer_audio) <= 0.1
    assert np.min(sl_al.layer_audio) >= -0.1


# def test_save_layer():
#     pass


def test_prep_layer():

    sl_pl = sl()

    sl_pl.attack = 0.1
    sl_pl.decay = 0.2
    sl_pl.prep_layer()
    assert sl_pl.attack_samples < sl_pl.decay_samples
    assert sl_pl.attack_samples < sl_pl.num_samples
    assert sl_pl.num_samples > sl_pl.decay_samples
    assert sl_pl.attack_samples + sl_pl.decay_samples == sl_pl.num_samples
    assert sl_pl.att_t[0] == 0
    assert sl_pl.att_t[-1] == 1
    assert sl_pl.att_t[sl_pl.attack_samples - 1] == 1
    assert sl_pl.dec_t[0] == 0
    assert sl_pl.dec_t[-1] == 1
    assert sl_pl.dec_t[sl_pl.decay_samples - 1] == 1
    assert sl_pl.att_t.shape[0] < sl_pl.dec_t.shape[0]
    assert sl_pl.env_t.shape[0] > sl_pl.dec_t.shape[0]
    assert sl_pl.att_t.shape[0] + sl_pl.dec_t.shape[0] == sl_pl.env_t.shape[0]
    sl_pl.attack = 2
    sl_pl.decay = 0.2
    sl_pl.prep_layer()
    assert sl_pl.attack_samples > sl_pl.decay_samples
    assert sl_pl.attack_samples < sl_pl.num_samples
    assert sl_pl.num_samples > sl_pl.decay_samples
    assert sl_pl.attack_samples + sl_pl.decay_samples == sl_pl.num_samples
    assert sl_pl.att_t[0] == 0
    assert sl_pl.att_t[-1] == 1
    assert sl_pl.att_t[sl_pl.attack_samples - 1] == 1
    assert sl_pl.dec_t[0] == 0
    assert sl_pl.dec_t[-1] == 1
    assert sl_pl.dec_t[sl_pl.decay_samples - 1] == 1
    assert sl_pl.att_t.shape[0] > sl_pl.dec_t.shape[0]
    assert sl_pl.env_t.shape[0] > sl_pl.dec_t.shape[0]
    assert sl_pl.att_t.shape[0] + sl_pl.dec_t.shape[0] == sl_pl.env_t.shape[0]


def test_create_short_layer_desc():
    sl_csl = sl(
        bit_depth=16,
        sample_rate=44100,
        layer_level=0.99,
        pitch=60,
        attack=0.2,
        decay=2.25,
    )
    assert (
        sl_csl._create_short_layer_desc()
        == 'bd-16_sr-44100_ll-0.99_p-60_a-0.2_d-2.25_as-8820_ds-99225_ns-108045'
    )


def test_create_long_layer_desc():
    sl_csl = sl(
        bit_depth=16,
        sample_rate=44100,
        layer_level=0.99,
        pitch=60,
        attack=0.2,
        decay=2.25,
    )
    sl_csl._create_long_layer_desc() == 'bit_depth: 16, sample_rate: 44100, layer_level: 0.99, pitch: 60, attack: 0.2, decay: 2.25, attack_samples: 8820, decay_samples: 99225, num_samples: 108045'


slt = sl()


def test_get_t():
    assert slt._gen_t(num_samples=50).shape[0] == 50
    assert slt._gen_t(num_samples=100)[-1] == 1
    assert slt._gen_t(num_samples=100)[0] == 0


####NORMALIZATION_METHODS####


def test_one_neg_one_normalization():
    test_norm = np.random.randint(low=-5, high=5, size=300) * 1.00001
    assert np.max(slt.one_neg_one_normalization(signal=test_norm)) <= 1
    assert np.min(slt.one_neg_one_normalization(signal=test_norm)) >= -1


def test_min_max_normalization():
    test_norm = np.random.randint(low=-5, high=5, size=300) * 1.00001
    assert np.max(slt.min_max_normalization(signal=test_norm)) <= 1
    assert np.min(slt.min_max_normalization(signal=test_norm)) >= -1


def test_normalize_audio():
    test_norm = np.random.randint(low=-5, high=5, size=300) * 1.00001
    assert np.max(slt.normalize_audio(signal=test_norm)) <= 1
    assert np.min(slt.normalize_audio(signal=test_norm)) >= -1
