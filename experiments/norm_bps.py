# This script detects the boundaries (or change points) of a midi file using 
# the Norm algorithm. It calculates the Precision, Recall and F1-score of the algorithm.
# The script also generate a plot with the amount of beats that the predicted
# boundaries differ from the ground truth boundaries. 
# This only works with the BPS-FH dataset.

from musicaiz.features.self_similarity import get_segment_boundaries
from musicaiz.features.rhythm import get_ioi
from musicaiz.rhythm import TimeSignature
from musicaiz.datasets import bps_fh

from pathlib import Path
import sys
sys.path.insert(1, "../")
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def get_ioi(notes):
    iois = []
    for i, next in zip(notes, notes[1:]):
        iois.append(next.start_sec - i.start_sec)
    return iois

def get_pitches(notes):
    return [note.pitch for note in notes]

def get_local_direction(
    notes,
):
    """Notes must be sorted by note on."""
    directions = []
    pitches = get_pitches(notes)
    for i, pitch in enumerate(pitches):
        if i == 0:
            continue
        if pitches[i-1] > pitches[i]:
            direction = -1
        if pitches[i-1] < pitches[i]:
            direction = 1
        if pitches[i-1] == pitches[i]:
            direction = 0
        directions.append(direction)
    return directions

def znormalization(ts):
    """
    ts - each column of ts is a time series (np.ndarray)
    """
    mus = ts.mean(axis = 0)
    stds = ts.std(axis = 0)
    return (ts - mus) / stds


def msa_norm(path, csv, tol, alpha1_values, threshold1_values, window2_values, threshold2_values, plot=True):
    precs_alg1, recs_alg1, fscores_alg1 = [], [], []
    precs_alg2, recs_alg2, fscores_alg2 = [], [], []

    files = list(path.glob("*/*.mid")) 

    for n_file, file_dir in enumerate(files):

        filename = file_dir.stem
        midi = Musa(file_dir)
        print(f"Processing file {filename} with alpha1 = {alpha1_values}, threshold1 = {threshold1_values}, window2 = {window2_values}, threshold2 = {threshold2_values} and tolernance={tolerance}")
        
        iois = get_ioi(midi.notes)
        local_dir = get_local_direction(midi.notes)
        vector = [local_dir[i] + iois[i] for i in range(len(iois))]
        vector = np.asarray(vector)

        ###############################################################
        ####################### Normalize IOIs ########################
        ###############################################################
        zts = znormalization(vector)

        from scipy import spatial, signal
        window = alpha1_values * (len(midi.notes)/15)
        
        def peak_picking(array, threshold, window):
            #Peaks detection - sliding window
            lamda = round(window) #window length
            peaks_position = signal.find_peaks(
                array,
                height=threshold,
                distance=lamda,
                width=1,
            )[0] #array of peaks
            peaks_values = signal.find_peaks(
            array,
            height=threshold,
            distance=lamda,
            width=1,
            )[1]['peak_heights'] #array of peaks

            #Adding elements 1 and N' to the begining and end of the array
            if peaks_position[0] != 0:
                peaks_position = np.concatenate([[0], peaks_position])
            if peaks_position[-1] != array.shape[0] - 1:
                peaks_position = np.concatenate([peaks_position, [array.shape[0]-1]])
            return peaks_position

        try:
            all_peaks = peak_picking(zts, threshold=threshold1_values, window = window)
        except:
            all_peaks = [0]

        print(f"Processing file {filename}")

        notes_position = all_peaks
        notes_position = np.asarray(notes_position)
        
        # Convert peaks (number of notes) in eights so we can compare vs ground truth
        predictions = notes_position

        import mir_eval

        # Prepare matrix with ground truth
        ref = np.empty((len(csv[filename])-1, 2), dtype=int) 
        for idx, row in enumerate(csv[filename].iterrows()):
            if idx == 0:
                continue
            if int(TIME_SIGS[filename].split("/")[1]) == 4:
                ref[idx-1, 0] = row[1][0] 
                ref[idx-1, 1] = row[1][1] 
            elif int(TIME_SIGS[filename].split("/")[1]) == 8:
                ref[idx-1, 0] = row[1][0] 
                ref[idx-1, 1] = row[1][1]
        
        ref = np.delete(ref, 0, 0)
        
        predictions = [midi.notes[n].beat_idx for n in predictions] 
        predictions = list(dict.fromkeys(predictions))

        est = np.empty((len(predictions)-1, 2), dtype=int)
        for j in range(len(predictions)):
            if j + 1 == len(predictions):
                break
            est[j, 0] = predictions[j]  # iois are the difference of 2 notes, so we add 1 note
            est[j, 1] = predictions[j+1]
        
        # We measure with a threshold that correspond to the 8th notes in one bar
        eights_bar = TimeSignature(TIME_SIGS[filename.split(".")[0]]).eights
        if int(TIME_SIGS[filename.split(".")[0]].split("/")[1]) == 4:
            eights = 2
        elif int(TIME_SIGS[filename.split(".")[0]].split("/")[1]) == 8:
            eights = 1
    
        if tolerance == "1_beat":
            eights = eights
        if tolerance == "1_bar":
            eights = eights_bar
        elif tolerance == "2_bars":
            eights = 2 * eights_bar
        
        prec, rec, fscore = mir_eval.segment.detection(ref, est, window=eights, beta=1.0, trim=False)

        precs_alg1.append(prec)
        recs_alg1.append(rec)
        fscores_alg1.append(fscore)

        print("Now, clean peaks with SSM")

        # Now compute distance between consecutive peaks
        #
        # Group notes in the predicted segments
        notes_segments = []
        for i in range(len(all_peaks)):
            if i + 1 == len(all_peaks):
                notes_segments.append(midi.notes[all_peaks[i]:-1])
                break
            notes_segments.append(midi.notes[all_peaks[i]:all_peaks[i+1]])
        
        # Distance
        dists = []
        for i in range(len(notes_segments)):
            if i + 1 == len(all_peaks):
                dists.append(0.0)
                break
            d = distance.euclidean(sum(get_ioi(notes_segments[i])), sum(get_ioi(notes_segments[i+1])))
            dists.append(d)

        ssm = np.empty((len(notes_segments), len(notes_segments)))
        for i in range(len(notes_segments)):
            for j in range(len(notes_segments)):
                ssm[i, j] = distance.euclidean(sum(get_ioi(notes_segments[i])), sum(get_ioi(notes_segments[j])))
        
        try:
            boundaries = get_segment_boundaries(ssm, threshold=threshold2_values, window = window2_values)
        except:
            boundaries = [0]

        peak_boundaries_pred = []
        for i in boundaries:
            peak_boundaries_pred.append(all_peaks[i])

        notes_position = []
        for i in range(len(peak_boundaries_pred)):
            count = peak_boundaries_pred[i]
            note_idx = peak_boundaries_pred[i]
            notes_position.append(count)
        notes_position = notes_position[:-1]
        notes_position = np.asarray(notes_position)
        
        # convert notes to eights
        predictions = notes_position
        predictions = [midi.notes[n].beat_idx for n in predictions] 
        predictions = list(dict.fromkeys(predictions))

        # Prepare matrix with estimations
        
        if ref[-1,1] not in predictions:
            predictions.append(ref[-1,1])

        est = np.empty((len(predictions)-1, 2), dtype=int)

        for j in range(len(predictions)):
            if j + 1 == len(predictions):
                break
            est[j, 0] = predictions[j]  # iois are the difference of 2 notes, so we add 1 note
            est[j, 1] = predictions[j+1]

        if est.shape[0] != 0:
            est = np.delete(est, 0, 0)
        
        est = np.sort(est)
        print("ground truth:", ref)
        print("predicted:", est)

        prec, rec, fscore = mir_eval.segment.detection(ref, est, window=eights, beta=1.0, trim=False)

        precs_alg2.append(prec)
        recs_alg2.append(rec)
        fscores_alg2.append(fscore)
      
    return precs_alg1, recs_alg1, fscores_alg1, precs_alg2, recs_alg2, fscores_alg2


from musicaiz.loaders import Musa

dataset = Path("C:/Users/Usuario/Documents/Universidad/1 trabajo/BPS-FH/BPS_FH_Dataset")
TIME_SIGS = bps_fh.BPSFH.TIME_SIGS
dataset_name = "bps"
level = "high" # "high", "mid" or "low"
csv = bps_fh.BPSFH(dataset).parse_anns(level)
tolerance = ["1_beat", "1_bar"] 

# Optimal values for high level
alpha1_values = 0.5
threshold1_values = 0.8
window2_values = 2
threshold2_values = 0.5

# Optimal values for mid level
#alpha1_values = 0.2
#threshold1_values = 0.5
#window2_values = 2
#threshold2_values = 0.5

# Optimal values for low level
#alpha1_values = 0.05
#threshold1_values = 0.05
#window2_values = 2
#threshold2_values = 0.2

precisions = np.array([])
recalls = np.array([])
fscores = np.array([])

precisions_alg1_mean = np.array([])
recalls_alg1_mean = np.array([])
fscores_alg1_mean = np.array([])

precisions_alg1_std = np.array([])
recalls_alg1_std = np.array([])
fscores_alg1_std = np.array([])

precisions_std = np.array([])
recalls_std = np.array([])
fscores_std = np.array([])

for tol in tolerance:
    precs_alg1, recs_alg1, fscores_alg1, precs_alg2, recs_alg2, fscores_alg2  = msa_norm(dataset, csv, tol, alpha1_values, threshold1_values, window2_values, threshold2_values, plot=True)
    
    precs_alg1 = np.asarray(precs_alg1)
    recs_alg1 = np.asarray(recs_alg1)
    fscores_alg1 = np.asarray(fscores_alg1)

    precs_alg2 = np.asarray(precs_alg2)
    recs_alg2 = np.asarray(recs_alg2)
    fscores_alg2 = np.asarray(fscores_alg2)

    precs_alg1_mean = precs_alg1.mean()
    recs_alg1_mean = recs_alg1.mean()
    fscoress_alg1_mean = fscores_alg1.mean()

    precs_alg2_mean = precs_alg2.mean()
    recs_alg2_mean = recs_alg2.mean()
    fscores_alg2_mean = fscores_alg2.mean()

    precs_alg1_std = precs_alg1.std()
    recs_alg1_std = recs_alg1.std()
    fscoress_alg1_std = fscores_alg1.std()

    precs_alg2_std = precs_alg2.std()
    recs_alg2_std = recs_alg2.std()
    fscores_alg2_std = fscores_alg2.std()

    precisions_alg1_mean = np.concatenate((precisions_alg1_mean, [precs_alg1_mean]))
    recalls_alg1_mean = np.concatenate((recalls_alg1_mean, [recs_alg1_mean]))
    fscores_alg1_mean = np.concatenate((fscores_alg1_mean, [fscoress_alg1_mean]))

    precisions_alg1_std = np.concatenate((precisions_alg1_std, [precs_alg1_std]))
    recalls_alg1_std = np.concatenate((recalls_alg1_std, [recs_alg1_std]))
    fscores_alg1_std = np.concatenate((fscores_alg1_std, [fscoress_alg1_std]))

    precisions = np.concatenate((precisions, [precs_alg2_mean]))
    recalls = np.concatenate((recalls, [recs_alg2_mean]))
    fscores = np.concatenate((fscores, [fscores_alg2_mean]))

    precisions_std = np.concatenate((precisions_std, [precs_alg2_std]))
    recalls_std = np.concatenate((recalls_std, [recs_alg2_std]))
    fscores_std = np.concatenate((fscores_std, [fscores_alg2_std]))

    with open(f"results_norm_{dataset_name}_{level}.txt", 'a') as f:
            
        f.write(f"------ RESULTS FOR alpha1 = {alpha1_values}, threshold1 = {threshold1_values}, window2 = {window2_values}, threshold2 = {threshold2_values} and tolerance {tolerance} ----------\n")
        f.write(f"p: {precisions_alg1_mean} +/- {precisions_alg1_std}\n")
        f.write(f"r: {recalls_alg1_mean} +/- {recalls_alg1_std}\n")
        f.write(f"f1: {fscores_alg1_mean} +/- {fscores_alg1_std}\n")
        f.write(f"p': {precisions} +/- {precisions_std}\n")
        f.write(f"r': {recalls} +/- {recalls_std}\n")
        f.write(f"f1': {fscores} +/- {fscores_std}\n")
        f.write(f"\n")
        
    plt.show()
