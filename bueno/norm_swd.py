# This script detects the boundaries (or change points) of a midi file using 
# the Norm algorithm. It calculates the precision, recall and f1-score of the algorithm.
# The script also generate a plot with the amount of beats that the predicted
# boundaries differ from the ground truth boundaries. 
# This only works with the SWD dataset.

from musicaiz.features.self_similarity import get_segment_boundaries
from musicaiz.features.rhythm import get_ioi
from musicaiz.rhythm import TimeSignature

from pathlib import Path
import sys
sys.path.insert(1, "../")
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import pandas as pd
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def parse_anns(midis_path, anns_path) -> dict: 
    """Converts the bar index float 
    annotations in 8th notes."""
    table = {}
    empiezo_en_cero = True

    for file in list(midis_path.glob("*.mid")):

        if file.stem not in valid_files:
            continue
        table[file.name.split(".")[0]] = [] 

        gt = pd.read_csv( 
            Path(anns_path, file.name.split(".")[0] + ".csv"),
            sep=';', header=None, engine='python'
        )
        gt_aux = gt

        if(empiezo_en_cero): 
            gt_aux[0][1] = 0

        key = gt_aux[2][1:] 
        gt_aux[2][1:] = key 

        print(file.name.split(".")[0], TIME_SIGS[file.name.split(".")[0]])

        for i, ann in enumerate(gt_aux[0]):
            if i == 0: 
                continue
            if "." not in str(ann):
                ann = float(ann) 
                bar_idx = int(float(("0." + str(ann).split(".")[0]))) 
                float_beat_idx = float("0." + str(ann).split(".")[1]) 
            else:
                bar_idx = int(float(str(ann).split(".")[0]))-1 
                float_beat_idx = float("0." + str(ann).split(".")[1]) 
            
            timesig = TimeSignature(TIME_SIGS[file.name.split(".")[0]])
            eights_per_bar = timesig.eights
            eight_idx = round(eights_per_bar * float_beat_idx) 
            total_eights = bar_idx * eights_per_bar + eight_idx  
            gt_aux[0][i] = int(total_eights) 
            
        for i, ann in enumerate(gt_aux[1]):
            if i == 0:
                continue
        
            # Repeat the code below but for the end annotations
            if "." not in str(ann):
                ann = float(ann)
                bar_idx = int(float(("0." + str(ann).split(".")[0])))
                float_beat_idx = float("0." + str(ann).split(".")[1])
            else:
                bar_idx = int(float(str(ann).split(".")[0]))-1
                float_beat_idx = float("0." + str(ann).split(".")[1])
            # if annotation is .999 it means that the bar is finished
            # so we don't compute the fraction, we directly compute the exact number of bars
            # when the bar is finished
            if "99" in str(float_beat_idx):
                float_beat_idx = 0
                bar_idx = bar_idx + 1

            timesig = TimeSignature(TIME_SIGS[file.name.split(".")[0]])
            eights_per_bar = timesig.eights
            eight_idx = round(eights_per_bar * float_beat_idx)
            total_eights = bar_idx * eights_per_bar + eight_idx
            gt_aux[1][i] = int(total_eights)

        table[file.name.split(".")[0]] = gt_aux

    return table


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

def msa_norm(table, tol, alpha1_values, threshold1_values, window2_values, threshold2_values, plot=True):
    precs_alg1, recs_alg1, fscores_alg1 = [], [], []
    precs_alg2, recs_alg2, fscores_alg2 = [], [], []

    for n_file, file_dir in enumerate(list(midis_path.glob("*.mid"))):

        filename = file_dir.stem
        if filename not in valid_files:
            continue
        
        midi = Musa(str(midis_path) + "/" + filename + ".mid", quantize=False)
        print(f"Processing file {filename} with alpha1 = {alpha1_values}, threshold1 = {threshold1_values}, window2 = {window2_values}, threshold2 = {threshold2_values} and tolernance={tolerance}")
        
        iois = get_ioi(midi.notes)
        local_dir = get_local_direction(midi.notes)
        vector = [local_dir[i] + iois[i] for i in range(len(iois))]
        vector = np.asarray(vector)

        ###############################################################
        ####################### Normalize IOIs ########################
        ###############################################################
        zts = znormalization(vector)

        from scipy import signal
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
        ref = np.empty((len(table[filename])-1, 2), dtype=int) 
        for idx, row in enumerate(table[filename].iterrows()): 
            if idx == 0:
                continue
            if int(TIME_SIGS[filename].split("/")[1]) == 4:
                ref[idx-1, 0] = row[1][0] / 2
                ref[idx-1, 1] = row[1][1] / 2
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
        eights_bar = int(midi.time_signature_changes[-1]["time_sig"].eights)
        if int(midi.time_signature_changes[-1]["time_sig"].denom) == 4:
            eights = 2
        elif int(midi.time_signature_changes[-1]["time_sig"].denom) == 8:
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
            # If threshold is too high that the algorithm does not find peaks try with threshold=0
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
        
        ref = np.delete(ref, 0, 0)

        if est.shape[0] != 0:
            est = np.delete(est, 0, 0)
        
        prec, rec, fscore = mir_eval.segment.detection(ref, est, window=eights, beta=1.0, trim=False)

        precs_alg2.append(prec)
        recs_alg2.append(rec)
        fscores_alg2.append(fscore)
      
    return precs_alg1, recs_alg1, fscores_alg1, precs_alg2, recs_alg2, fscores_alg2

from musicaiz.loaders import Musa

dataset = "C:/Users/Usuario/Documents/Universidad/1 trabajo/SWD"
midis_path = Path(dataset, "01_RawData/score_midi")
anns_path = Path(dataset, "02_Annotations/ann_score_structure")
table = parse_anns(midis_path, anns_path)

TIME_SIGS = {
        "Schubert_D911-01": "2/4", "Schubert_D911-02": "6/8", "Schubert_D911-03": "4/4", "Schubert_D911-04": "4/4", 
        "Schubert_D911-05": "3/4", "Schubert_D911-06": "3/4", "Schubert_D911-07": "2/4", "Schubert_D911-08": "3/4",
        "Schubert_D911-09": "3/8", "Schubert_D911-10": "2/4", "Schubert_D911-12": "2/4",
        "Schubert_D911-13": "6/8", "Schubert_D911-14": "3/4", "Schubert_D911-15": "2/4", "Schubert_D911-16": "3/4",
        "Schubert_D911-17": "12/8", "Schubert_D911-18": "4/4", "Schubert_D911-19": "6/8", "Schubert_D911-20": "2/4",
        "Schubert_D911-21": "4/4", "Schubert_D911-22": "2/4", "Schubert_D911-23": "3/4", "Schubert_D911-24": "3/4",
    }

valid_files = list(TIME_SIGS.keys()) # coge los nombres de los archivos

tolerance = ["1_beat", "1_bar"]

# Optimal values for norm algorithm
alpha1_values = 0.6
threshold1_values = 1
window2_values = 2
threshold2_values = 0.5

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
    precs_alg1, recs_alg1, fscores_alg1, precs_alg2, recs_alg2, fscores_alg2  = msa_norm(table, tol, alpha1_values, threshold1_values, window2_values, threshold2_values, plot=True)
    
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

    precisionss_alg1_std = np.concatenate((precisions_alg1_std, [precs_alg1_std]))
    recalls_alg1_std = np.concatenate((recalls_alg1_std, [recs_alg1_std]))
    fscores_alg1_std = np.concatenate((fscores_alg1_std, [fscoress_alg1_std]))

    precisions = np.concatenate((precisions, [precs_alg2_mean]))
    recalls = np.concatenate((recalls, [recs_alg2_mean]))
    fscores = np.concatenate((fscores, [fscores_alg2_mean]))

    precisions_std = np.concatenate((precisions_std, [precs_alg2_std]))
    recalls_std = np.concatenate((recalls_std, [recs_alg2_std]))
    fscores_std = np.concatenate((fscores_std, [fscores_alg2_std]))

    from baseline import create_baseline

    dataset_name = "bps"
    level = "mid"        
    precs_alg_base, recs_alg_base, fscores_alg_base = create_baseline(dataset_name, level, tol)

    precs_alg_base = np.asarray(precs_alg_base)
    recs_alg_base = np.asarray(recs_alg_base)
    fscores_alg_base = np.asarray(fscores_alg_base)

    baseline_precs = precs_alg_base.mean()
    baseline_recs = recs_alg_base.mean()
    baseline_fscores = fscores_alg_base.mean()

    baseline_precs_std = precs_alg_base.std()
    baseline_recs_std = recs_alg_base.std()
    baseline_fscores_std = fscores_alg_base.std()

    precisions = np.concatenate(([baseline_precs], precisions))
    recalls = np.concatenate(([baseline_recs], recalls))
    fscores = np.concatenate(([baseline_fscores], fscores))

    precisions_std = np.concatenate(([baseline_precs_std], precisions_std))
    recalls_std = np.concatenate(([baseline_recs_std], recalls_std))
    fscores_std = np.concatenate(([baseline_fscores_std], fscores_std))

    with open(f"norm_{dataset_name}_{level}.txt", 'a') as f:

            for i in range(len(precisions)):
                if i == 0:
                    f.write(f"------ RESULTS FOR BASELINE for tolerance {tolerance} ----------\n")
                    f.write(f"P': {precisions[i]} +/- {precisions_std[i]} \n")
                    f.write(f"R': {recalls[i]} +/- {recalls_std[i]} \n")
                    f.write(f"F1': {fscores[i]} +/- {fscores_std[i]} \n")
                    f.write(f"\n")

                else:
                    f.write(f"------ RESULTS FOR alpha1 = {alpha1_values}, threshold1 = {threshold1_values}, window2 = {window2_values}, threshold2 = {threshold2_values} and tolerance {tolerance} ----------\n")
                    f.write(f"P: {precisions_alg1_mean[i-1]} +/- {precisions_alg1_std[i-1]}\n")
                    f.write(f"R: {recalls_alg1_mean[i-1]} +/- {recalls_alg1_std[i-1]}\n")
                    f.write(f"F1: {fscores_alg1_mean[i-1]} +/- {fscores_alg1_std[i-1]}\n")
                    f.write(f"P': {precisions[i]} +/- {precisions_std[i]}\n")
                    f.write(f"R': {recalls[i]} +/- {recalls_std[i]}\n")
                    f.write(f"F1': {fscores[i]} +/- {fscores_std[i]}\n")
                    f.write(f"\n")
        
    x = np.arange(len(window2_values)+1)
    fig, ax = plt.subplots()
    fig.set_dpi(300)
    fig.set_size_inches(6,2)

    bar_width = 0.2

    bar_prec = ax.bar(x - bar_width, precisions, bar_width, label='P', color='#3498DB')
    bar_rec = ax.bar(x, recalls, bar_width, label='R', color='#1F618D')
    bar_fscore = ax.bar(x + bar_width, fscores, bar_width, label='F1', color='#CACACA')


    # add errorbar
    for i in range(len(bar_prec)):
        ax.errorbar(bar_prec[i].get_x() + bar_prec[i].get_width()/2, precisions[i], yerr=precisions_std[i], fmt='none', ecolor='black', capsize=0, capthick = 0.1)
    for i in range(len(bar_rec)):
        ax.errorbar(bar_rec[i].get_x() + bar_rec[i].get_width()/2, recalls[i], yerr=recalls_std[i], fmt='none', ecolor='black', capsize=0, capthick = 0.1)
    for i in range(len(bar_fscore)):
        ax.errorbar(bar_fscore[i].get_x() + bar_fscore[i].get_width()/2, fscores[i], yerr=fscores_std[i], fmt='none', ecolor='black', capsize=0, capthick = 0.1)

    ax.set_xticks(x)
    ax.set_xticklabels(['baseline'] + window2_values)
    ax.legend(prop={'size': 6})
    plt.savefig(f"{dataset}/norm_swd_{tol}", dpi=300)
    plt.show()
