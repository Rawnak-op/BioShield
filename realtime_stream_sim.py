from signal_generator import generate_sample

def ecg_stream(label=0):
    while True:
        noisy, clean = generate_sample(label, return_clean=True)
        for n, c in zip(noisy, clean):
            yield n, c, label
