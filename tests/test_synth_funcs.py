from gemini_drum_synth import gemini_drum_synth as gds

slo = gds.SynthLayer()


def test_get_t():
    assert slo.get_t(num_samples=50).shape[0] == 50
    assert slo.get_t(num_samples=100)[-1] == 1
    assert slo.get_t(num_samples=100)[0] == 0
