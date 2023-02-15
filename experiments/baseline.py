# This script calculates the baseline for both the SWD and BPS-FH datasets.
# The baseline is a segmentation of the piece into different segments of equal length
# depending on the level and the number of notes
# It also calculates the precision, recall and F-score of the baseline

from musicaiz.loaders import Musa
from musicaiz.rhythm import TimeSignature
from musicaiz.datasets import bps_fh

import numpy as np
import pandas as pd
from pathlib import Path
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def parse_anns(midis_path, anns_path, valid_files,TIME_SIGS) -> dict: 
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
            denom_bar = int(TIME_SIGS[file.name.split(".")[0]].split("/")[1])
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


def msa_norm(level, midi_file, csv, tolerance, valid_files, TIME_SIGS, swd = False): 

    if swd:
        files = list(midi_file.glob("*.mid"))
    else:
        files = list(midi_file.glob("*/*.mid"))

    precs_alg_base, recs_alg_base, fscores_alg_base = [], [], []
    baseline_precs = np.array([])
    baseline_recs = np.array([])
    baseline_fscores = np.array([])

    for n_file, file_dir in enumerate(files):

        filename = file_dir.stem 
        print(f"Processing baseline for file {filename}")
        
        if swd:
            if filename not in valid_files:
                continue

            ref = np.empty((len(csv[filename])-1, 2), dtype=int) 
            for idx, row in enumerate(csv[filename].iterrows()): 
                if idx == 0:
                    continue 
                if int(TIME_SIGS[filename].split("/")[1]) == 4:
                    ref[idx-1, 0] = row[1][0] / 2
                    ref[idx-1, 1] = row[1][1] / 2
                elif int(TIME_SIGS[filename].split("/")[1]) == 8:
                    ref[idx-1, 0] = row[1][0] 
                    ref[idx-1, 1] = row[1][1]
        else:

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

        if swd:
            n_segments = 8
        else:
            if level == "low":
                n_segments = 46
            elif level == "mid":
                n_segments = 14
            elif level == "high":
                n_segments = 4

       # Create a baseline with n_segments of equal length
        musa_obj = Musa(file_dir)
        
        length = musa_obj.beats[-1].global_idx
        segment_size = length / n_segments
        baseline = np.empty((n_segments, 2), dtype=int)
        for i in range(n_segments):
            baseline[i, 0] = int(i * segment_size)
            baseline[i, 1] = int((i + 1) * segment_size)
        baseline = np.delete(baseline, 0, 0)
        print("Ground Truth:", ref)
        print("Baseline:", baseline)

        # Calculate prec, rec and fscore for the baseline
        import mir_eval
        prec_base, rec_base, fscore_base = mir_eval.segment.detection(ref, baseline, window=eights, beta=1.0, trim=False)
        
        print("---Results for boundaries for the baseline---")
        print(f"P={prec_base}")
        print(f"R={rec_base}")
        print(f"F={fscore_base}")

        precs_alg_base.append(prec_base)
        recs_alg_base.append(rec_base)
        fscores_alg_base.append(fscore_base)
    
    baseline_precs = np.asarray(precs_alg_base)
    baseline_recs = np.asarray(recs_alg_base)
    baseline_fscores = np.asarray(fscores_alg_base)

    return baseline_precs, baseline_recs, baseline_fscores


def create_baseline(dataset_name, level, tolerance):

    if dataset_name == "swd":

        TIME_SIGS = {
            "Schubert_D911-01": "2/4", "Schubert_D911-02": "6/8", "Schubert_D911-03": "4/4", "Schubert_D911-04": "4/4", 
            "Schubert_D911-05": "3/4", "Schubert_D911-06": "3/4", "Schubert_D911-07": "2/4", "Schubert_D911-08": "3/4",
            "Schubert_D911-09": "3/8", "Schubert_D911-10": "2/4", "Schubert_D911-12": "2/4",
            "Schubert_D911-13": "6/8", "Schubert_D911-14": "3/4", "Schubert_D911-15": "2/4", "Schubert_D911-16": "3/4",
            "Schubert_D911-17": "12/8", "Schubert_D911-18": "4/4", "Schubert_D911-19": "6/8", "Schubert_D911-20": "2/4",
            "Schubert_D911-21": "4/4", "Schubert_D911-22": "2/4", "Schubert_D911-23": "3/4", "Schubert_D911-24": "3/4",
        }

        valid_files = list(TIME_SIGS.keys()) 
        dataset = "C:/Users/Usuario/Documents/Universidad/1 trabajo/SWD"
        midi_file = Path(dataset, "01_RawData/score_midi")
        anns_file = Path(dataset, "02_Annotations/ann_score_structure")
        csv = parse_anns(midi_file, anns_file, valid_files, TIME_SIGS)

        for tol in tolerance:
            baseline_precs, baseline_recs, baseline_fscores = msa_norm(level, midi_file, csv, tol, valid_files, TIME_SIGS, swd = True)
            
            with open(f"baseline_{dataset_name}.txt", 'a') as f:
                    f.write(f"------ RESULTS FOR BASELINE FOR DATASET {dataset_name}, LEVEL {level} AND TOLERANCE {tol} ----------\n")
                    f.write(f"P: {baseline_precs.mean()} +/- {baseline_precs.std()}\n")
                    f.write(f"R: {baseline_recs.mean()} +/- {baseline_recs.std()}\n")
                    f.write(f"F1: {baseline_fscores.mean()} +/- {baseline_fscores.std()}\n")
                    f.write(f"\n")

    elif dataset_name == "bps":
        midi_file = Path("C:/Users/Usuario/Documents/Universidad/1 trabajo/BPS-FH/BPS_FH_Dataset")
        csv = bps_fh.BPSFH(midi_file).parse_anns(level)
        TIME_SIGS = bps_fh.BPSFH.TIME_SIGS
        valid_files = []
        
        for tol in tolerance:
            baseline_precs, baseline_recs, baseline_fscores = msa_norm(level, midi_file, csv, tol, valid_files, TIME_SIGS, swd=False)

            with open(f"baseline_{dataset_name}.txt", 'a') as f:
                    f.write(f"------ RESULTS for baseline for dataset {dataset_name}, level {level} and tolerance {tol} ----------\n")
                    f.write(f"P: {baseline_precs.mean()} +/- {baseline_precs.std()}\n")
                    f.write(f"R: {baseline_recs.mean()} +/- {baseline_recs.std()}\n")
                    f.write(f"F1: {baseline_fscores.mean()} +/- {baseline_fscores.std()}\n")
                    f.write(f"\n")
    return baseline_precs, baseline_recs, baseline_fscores

# dataset = "swd"
# level = "mid"  
# tolerance = ["1_beat","1_bar"]
# create_baseline(dataset,level,tolerance)
