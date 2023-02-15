# This script detects the boundaries (or change points) of a midi file using 
# the pelt and window algorithms, applied to a graph representation. 
# It calculates the precision, recall and f1-score of the algorithm.
# The script also generate a plot with the amount of beats that the predicted
# boundaries differ from the ground truth boundaries. 
# This only works with the SWD dataset.

from musicaiz.loaders import Musa
from musicaiz.rhythm import TimeSignature
from musicaiz.features import PeltArgs, WindowArgs, StructurePrediction, StructurePredictionWindow

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Create constant with the name of the file and the tempo
TIME_SIGS = {
        "Schubert_D911-01": "2/4", "Schubert_D911-02": "6/8", "Schubert_D911-03": "4/4", "Schubert_D911-04": "4/4", 
        "Schubert_D911-05": "3/4", "Schubert_D911-06": "3/4", "Schubert_D911-07": "2/4", "Schubert_D911-08": "3/4",
        "Schubert_D911-09": "3/8", "Schubert_D911-10": "2/4", "Schubert_D911-12": "2/4",
        "Schubert_D911-13": "6/8", "Schubert_D911-14": "3/4", "Schubert_D911-15": "2/4", "Schubert_D911-16": "3/4",
        "Schubert_D911-17": "12/8", "Schubert_D911-18": "4/4", "Schubert_D911-19": "6/8", "Schubert_D911-20": "2/4",
        "Schubert_D911-21": "4/4", "Schubert_D911-22": "2/4", "Schubert_D911-23": "3/4", "Schubert_D911-24": "3/4",
    }

valid_files = list(TIME_SIGS.keys()) # coge los nombres de los archivos
#valid_files = ["Schubert_D911-03", "Schubert_D911-10", "Schubert_D911-13", "Schubert_D911-16","Schubert_D911-18", "Schubert_D911-19", "Schubert_D911-22"]
#tresillos_files = ["Schubert_D911-16", "Schubert_D911-18", "Schubert_D911-19"]
#valid_files = ["Schubert_D911-01"]
#Escribir por pantalla la constante TIME_SIGS separada por coma
#print(TIME_SIGS)


def parse_anns(midis_path, anns_path) -> dict: #construye tabla de etiquetas
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

def create_baseline(musa_obj):
    # Get the length of the MIDI file
    length = musa_obj.beats[-1].global_idx
    # Divide the length into 5 segments
    segment_size = length / 5
    # Create an empty array to store the baseline values
    baseline = np.empty((5, 2), dtype=int)
    for i in range(5):
        baseline[i, 0] = int(i * segment_size)
        baseline[i, 1] = int((i + 1) * segment_size)
    baseline = np.delete(baseline, 0, 0)
    print(baseline)
    return baseline


def msa_norm(csv, tolerance, alpha_values, jump_values, alg, plot, baseline=False): 
    files = list(midi_file.glob("*.mid")) 

    precisions = np.array([])
    recalls = np.array([])
    fscores = np.array([])
   
    baseline_precs = np.array([])
    baseline_recs = np.array([])
    baseline_fscores = np.array([])
   
    baseline_precs_std =np.array([])
    baseline_recs_std = np.array([])
    baseline_fscores_std = np.array([])

    prec_matrix = np.empty((0, len(jump_values)), float)
    rec_matrix = np.empty((0, len(jump_values)), float)
    fscore_matrix = np.empty((0, len(jump_values)), float)
    
    precs_alg_base, recs_alg_base, fscores_alg_base = [], [],[]

    border_matrix = np.zeros((len(csv), len(csv)), float)
    ground_truth_matrix = np.zeros((len(csv), len(csv)), float)
    i = 0
    for n_file, file_dir in enumerate(files):
        precs_alg, recs_alg, fscores_alg = [], [],[] 

        filename = file_dir.stem 
        
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
        
        ref = np.delete(ref, 0, 0)
        
        
        # Measure with a threshold that correspond to the 8th notes in one bar
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

        musa_obj = Musa(file_dir)

        if baseline:
            print(f"Processing baseline for file {filename}")
            
            base = create_baseline(musa_obj) 

            # Calculate prec, rec and fscore for the baseline
            import mir_eval
            prec_base, rec_base, fscore_base = mir_eval.segment.detection(ref, base, window=eights, beta=1.0, trim=False)
            print("---Results for boundaries for the baseline---")
            print(f"P={prec_base}")
            print(f"R={rec_base}")
            print(f"F={fscore_base}")

            precs_alg_base.append(prec_base)
            recs_alg_base.append(rec_base)
            fscores_alg_base.append(fscore_base)

        if alg == "pelt":
            sp = StructurePrediction(file_dir)
        elif alg == "window":
            sp = StructurePredictionWindow(file_dir)

        for alpha in alpha_values:
            for jump in jump_values:
                print(f"Processing file {filename} with tolerance {tolerance}, alpha {alpha} and jump {jump}")
                if alg == "pelt":
                    minsize = alpha*(len(musa_obj.notes)/15)
                    jump =  int(round(jump * minsize))
                    penalty= 0.7
                    pelt_args = PeltArgs(
                        penalty=penalty,
                        model="rbf",
                        minsize=minsize, 
                        jump=jump, 
                    )
                elif alg == "window":
                    width = int(alpha*(len(musa_obj.notes)/15))
                    penalty= 0.5
                    pelt_args = WindowArgs(
                        penalty=penalty,
                        model="rbf",
                        width=width,
                    )

                result = sp.beats(pelt_args)

                import mir_eval
                predictions = result

                # Prepare matrix with ground truth
                est = np.empty((len(predictions)-1, 2), dtype=int) 
                for j in range(len(predictions)):
                    if j + 1 == len(predictions):
                        break
                    est[j, 0] = predictions[j]
                    est[j, 1] = predictions[j+1]
            
                print("Ground truth:", ref)
                print("Predicted:", est)

                prec, rec, fscore = mir_eval.segment.detection(ref, est, window=eights, beta=1.0, trim=False)
                print("---Results for boundaries---")
                print(f"P={prec}")
                print(f"R={rec}")
                print(f"F={fscore}")

                precs_alg.append(prec)
                recs_alg.append(rec)
                fscores_alg.append(fscore)

        predictions = np.array(predictions)
        predictions.resize(border_matrix.shape[1])
        border_matrix[i, :] = predictions

        ref = np.array([ref[0, 0]] + [row[1] for row in ref])
        ref.resize(border_matrix.shape[1])
        ground_truth_matrix[i, :] = ref
        i = i + 1

        # Append the row to the corresponding matrix
        prec_matrix = np.vstack([prec_matrix, precs_alg])
        rec_matrix = np.vstack([rec_matrix, recs_alg])
        fscore_matrix = np.vstack([fscore_matrix, fscores_alg])

        precs_alg = np.asarray(precs_alg)
        recs_alg = np.asarray(recs_alg)
        fscores_alg = np.asarray(fscores_alg)
        
 
    precisions = np.mean(prec_matrix, axis=0)
    recalls = np.mean(rec_matrix, axis=0)
    fscores = np.mean(fscore_matrix, axis=0)  
    
    baseline_precs = np.asarray(precs_alg_base).mean()
    baseline_recs = np.asarray(recs_alg_base).mean()
    baseline_fscores = np.asarray(fscores_alg_base).mean()

    # Concatenate baseline and algorithm results    
    precisions = np.concatenate(([baseline_precs], precisions))
    recalls = np.concatenate(([baseline_recs], recalls))
    fscores = np.concatenate(([baseline_fscores], fscores))

    precisions_std = np.std(prec_matrix, axis=0)
    recalls_std = np.std(rec_matrix, axis=0)
    fscores_std = np.std(fscore_matrix, axis=0)

    baseline_precs_std = np.asarray(precs_alg_base).std()
    baseline_recs_std =  np.asarray(recs_alg_base).std()
    baseline_fscores_std = np.asarray(fscores_alg_base).std()
    
    precisions_std = np.concatenate(([baseline_precs_std], precisions_std))
    recalls_std = np.concatenate(([baseline_recs_std], recalls_std))
    fscores_std = np.concatenate(([baseline_fscores_std], fscores_std))

    return precisions, recalls, fscores, precisions_std, recalls_std, fscores_std, border_matrix, ground_truth_matrix

def create_plot(tolerance, alpha_values, jump_values, alg, csv):

    precisions, recalls, fscores, precisions_std, recalls_std, fscores_std, border_matrix, ground_truth_matrix = msa_norm(csv, tolerance, alpha_values, jump_values, alg, plot=True, baseline=True)
    x = np.arange(len(jump_values)+1)
    fig, ax = plt.subplots()
    fig.set_dpi(300)
    fig.set_size_inches(6,2)
    with open(f"swd_{alg}.txt", 'a') as f:
        for i in range(len(precisions)):
            if i == 0:
                f.write(f"------ RESULTS FOR BASELINE for tolerance {tol} ----------\n")
                f.write(f"P: {precisions[i]} +/- {precisions_std[i]}\n")
                f.write(f"R: {recalls[i]} +/- {recalls_std[i]}\n")
                f.write(f"F1: {fscores[i]} +/- {fscores_std[i]}\n")
                f.write(f"\n")
                
            else:
                f.write(f"------ RESULTS for tolerance {tol}, alpha {alpha_values[i-1]}, jump {jump_values} and penalty 0.5 ----------\n")
                f.write(f"P: {precisions[i]} +/- {precisions_std[i]}\n")
                f.write(f"R: {recalls[i]} +/- {recalls_std[i]}\n")
                f.write(f"F1: {fscores[i]} +/- {fscores_std[i]}\n")
                f.write(f"\n")

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
    ax.set_xticklabels(['baseline'] + jump_values)
    ax.legend(prop={'size': 6})

    return plt, border_matrix, ground_truth_matrix


def compare_borders(ground_truth_matrix, border_matrix, dictionary = False):

   matrix = []
   beats_dict = dict()

   for z in range(len(border_matrix)):
      result = []
      count = 0

      # Load the vectors of each row, corresponding to each file
      print("Procesando fichero ", z)
      predicted = border_matrix[z,:]
      ground_truth = ground_truth_matrix[z,:]

      non_zero_indices_predicted = np.nonzero(predicted)
      predicted = predicted[non_zero_indices_predicted]
      non_zero_indices_ground_truth = np.nonzero(ground_truth)
      ground_truth = ground_truth[non_zero_indices_ground_truth]

      # Calculate the predicted boundaries closest to the ground truth
      for i in range(len(ground_truth)):
         if count <= len(ground_truth):
            for j in range(len(predicted)):
               num = abs(ground_truth[i]-predicted[j])  
               matrix = np.append(matrix, num)
            index = np.argmin(matrix)
            value = predicted[index]
            if value not in result:
               result = np.append(result, value)
               result = result.astype(int)
               count = count + 1
               if dictionary:
                  diff = abs(ground_truth[i]-predicted[index])  
                  if diff not in beats_dict:
                     beats_dict[diff] = 1
                  else:
                     beats_dict[diff] = beats_dict[diff] + 1
            matrix = []
      
      print("Fronteras originales")
      print(ground_truth)
      print("Fronteras predichas")
      print(result)
   return beats_dict


dataset = "C:/Users/Usuario/Documents/Universidad/1 trabajo/SWD"
midi_file = Path(dataset, "01_RawData/score_midi")
anns_file = Path(dataset, "02_Annotations/ann_score_structure")
csv = parse_anns(midi_file, anns_file)
tolerance = ["1_beat","1_bar"] 
alg = "pelt" # "window" or "pelt"

# Change alpha values
#alpha_values = [1,1.3,1.6,1.9]
#jump_values = [0.15]

# Optimal values for pelt algorithm
jump_values = [0.15]
alpha_values = [0.6]
penalty = 0.7

# Optimal values for window algorithm
#alpha_values = [1]
#jump_values = [0]
#penalty = 0.5

for tol in tolerance:
    breakpoint()
    plt, border_matrix, ground_truth_matrix = create_plot(tol, alpha_values, jump_values, alg, csv)
    plt.savefig(f"{dataset}/{alg}_{tol}_jump", dpi=300)
plt.show()

border_matrix = np.array(border_matrix)
border_matrix = border_matrix.astype(int)
ground_truth_matrix = np.array(ground_truth_matrix)
ground_truth_matrix = ground_truth_matrix.astype(int)

beats_dict = compare_borders(ground_truth_matrix, border_matrix, dictionary = True)  
print(beats_dict)        

x = list(beats_dict.keys())
y = list(beats_dict.values())

fig, ax = plt.subplots(figsize=(6,3))
ax.bar(x, y)
plt.savefig(f"{dataset}/errors_swd_{alg}_{tol}.png", dpi=300)
plt.show()




