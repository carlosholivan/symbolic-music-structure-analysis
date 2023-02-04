from pathlib import Path
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import pandas as pd


import pretty_midi as pm
from pretty_midi import TimeSignature
from musicaiz.rhythm import NoteLengths
from musicaiz.datasets import BPSFH


def bps_to_midi(file, time_sig, dest_path):
    notes_df = pd.read_csv(file, header=None)

    # Initialize empty Musa object
    midi = pm.PrettyMIDI(resolution=96)

    midi.instruments.append(
        pm.Instrument(
            program=0,
        )
    )
    midi.instruments[0].notes = []

    num = int(time_sig.split("/")[0])
    den = int(time_sig.split("/")[1])
    midi.time_signature_changes.append(
        TimeSignature(num, den, 0)
    )

    if den == 4:
        note = NoteLengths.QUARTER
    elif den == 8:
        note = NoteLengths.EIGHT

    upbeat = 0
    for row in range(notes_df.shape[0]):  # loop in rows
        pitch = int(notes_df.iloc[row, 1])
        onset_quarter = notes_df.iloc[row, 0]
        duration_quarter = notes_df.iloc[row, 3]

        # if the onset is negative it's an upbeat before bar 1
        if onset_quarter < 0:
            upbeat = midi.time_signature_changes[0].numerator + onset_quarter
            start = upbeat * note.ticks()
            end = start + int((duration_quarter) * note.ticks())
        else:
            start = int((onset_quarter + upbeat + 1) * note.ticks())
            end = start + int((duration_quarter) * note.ticks())

        # no duration, no note will be written
        if start == end:
            continue

        midi.instruments[0].notes.append(
            pm.Note(pitch=pitch, start=midi.tick_to_time(int(start)), end=midi.tick_to_time(int(end)), velocity=127)
        )

    midi.write(dest_path)

data_dir = Path("H:/INVESTIGACION/Datasets/functional-harmony-master/BPS_FH_Dataset")
for file in data_dir.glob("*"):
    if file.is_file():
        continue
    filename = file.stem
    file = Path(data_dir, f"{filename}/notes.csv")
    ts = BPSFH.TIME_SIGS[filename]
    dest_path = f"{data_dir}/{filename}/{filename}.mid"
    bps_to_midi(file, ts, dest_path)
    print(f"Converted file {filename} to MIDI")

# Plot
#from musicaiz import plotters, loaders
#musa = loaders.Musa("1.mid")
#plot = plotters.Pianoroll(musa)
#plot.plot_instruments([0], 0, 5, show=True)