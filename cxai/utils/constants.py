from zennit.rules import Epsilon, Gamma, Flat, WSquare

CLASS_IDX_MAPPER = {"pop": 0, "metal": 1, "disco": 2, "blues": 3, "reggae": 4, "classical": 5, "rock": 6, "hiphop": 7, "country": 8, "jazz": 9}

CLASS_IDX_MAPPER_TOY = {'class1': 0, 'class2': 1}

# parameters to perform DSTFT
AUDIO_PARAMS = {
    'gtzan': {'sample_rate': 16000, 
        'n_fft': 800, 
        'hop_length': 360, 
        'n_mels': 128, 
        'slice_length': 3, 
        'mel_width': 128
    }, 
    'toy': {'sample_rate': 16000, 
        'n_fft': 480, 
        'hop_length': 240, 
        'n_mels': 64, 
        'mel_width': 64
    }
}


# name maps for standard atribution with zennit

LRP_NAME_MAP_GTZAN = [
    # feature extractor
    (['features.0'], WSquare(stabilizer=1e-7)),
    (['features.3'], Gamma(gamma=0.4, stabilizer=1e-7)),
    (['features.6'], Gamma(gamma=0.4, stabilizer=1e-7)),
    (['features.9'], Gamma(gamma=0.4/2, stabilizer=1e-7)),
    (['features.12'], Gamma(gamma=0.4/4, stabilizer=1e-7)),
    # classification head
    (['classifier.0'], Epsilon(epsilon=1e-7)),
    (['classifier.3'], Epsilon(epsilon=1e-7)),
    (['classifier.6'], Epsilon(epsilon=1e-7))
]

LRP_NAME_MAP_TOY = [
    # feature extractor
    (['features.0'], Flat(stabilizer=1e-7)),
    (['features.3'], Gamma(gamma=0.8, stabilizer=1e-7)),
    (['features.6'], Gamma(gamma=0.8, stabilizer=1e-7)),
    (['features.9'], Gamma(gamma=0.8, stabilizer=1e-7)),
    (['features.12'], Gamma(gamma=0.8, stabilizer=1e-7)),
    # classification head
    (['classifier.0'], Epsilon(epsilon=1e-7)),
    (['classifier.2'], Epsilon(epsilon=1e-7)),
    (['classifier.4'], Epsilon(epsilon=1e-7))
]


# TODO: layer idx to dimension mapper (d=num filters)
