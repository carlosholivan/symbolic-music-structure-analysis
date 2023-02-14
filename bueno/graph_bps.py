# This script detects the boundaries (or change points) of a midi file using 
# the pelt and window algorithms, applied to a graph representation. 
# It calculates the precision, recall and f1-score of the algorithm.
# The script also generate a plot with the amount of beats that the predicted
# boundaries differ from the ground truth boundaries. 
# This only works with the BPS-FH dataset.


from musicaiz.loaders import Musa
from musicaiz.rhythm import TimeSignature
from musicaiz.datasets import bps_fh
from musicaiz.features import PeltArgs
from musicaiz.features import StructurePrediction, StructurePredictionWindow

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def msa_norm(alg, csv, path, tolerance, level, alpha_values, jump_values, penalty_values, plot, baseline=False): 
    
    files = list(path.glob("*/*.mid")) 
    precisions = np.array([])
    recalls = np.array([])
    fscores = np.array([])
   
    prec_matrix = np.empty((0, len(alpha_values)), float)
    rec_matrix = np.empty((0, len(alpha_values)), float)
    fscore_matrix = np.empty((0, len(alpha_values)), float)

    border_matrix = np.zeros((len(csv), len(csv)), float)
    ground_truth_matrix = np.zeros((len(csv), len(csv)), float)
    i = 0
    for n_file, file_dir in enumerate(files):
        precs_alg, recs_alg, fscores_alg = [], [],[] 
        filename = file_dir.stem 

        ref = np.empty((len(csv[filename])-1, 2), dtype=int) 
        for idx, row in enumerate(csv[filename].iterrows()): 
            if idx == 0:
                continue 
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

        if alg == "pelt":
            sp = StructurePrediction(file_dir)
        if alg == "window":
            sp = StructurePredictionWindow(file_dir)
       
        for alpha in alpha_values:
            for jump in jump_values:
                print(f"Processing file {filename} with tolerance {tolerance}, alpha {alpha}, jump {jump} and penalty {penalty_values}")
                minsize = alpha*(len(musa_obj.notes)/15)
                jump =  int(round(jump * minsize))
                if jump < 1:
                    jump = 1
                penalty= penalty_values
                pelt_args = PeltArgs(
                    penalty=penalty,
                    model="rbf",
                    minsize=minsize, 
                    jump=jump, 
                )
            
                result = sp.beats(pelt_args) 
            
                import mir_eval
                predictions = result
                predictions = list(dict.fromkeys(predictions))
        
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
   
    precisions_std = np.std(prec_matrix, axis=0)
    recalls_std = np.std(rec_matrix, axis=0)
    fscores_std = np.std(fscore_matrix, axis=0)
            
    return precisions, recalls, fscores, precisions_std, recalls_std, fscores_std, border_matrix, ground_truth_matrix


def create_plot(tolerance, level, alpha_values, jump_values, penalty_values, path, csv, alg):
        
    precisions, recalls, fscores, precisions_std, recalls_std, fscores_std, border_matrix, ground_truth_matrix = msa_norm(alg, csv, path, tolerance, level, alpha_values, jump_values, penalty_values, plot=True, baseline=True)
    
    from baseline import create_baseline
    dataset_name = "bps"       
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

    with open(f"graph_bps_{alg}_{level}.txt", 'a') as f:

        for i in range(len(precisions)):
            if i == 0:
                f.write(f"------ RESULTS FOR BASELINE for tolerance {tol} ----------\n")
                f.write(f"P: {precisions[i]} +/- {precisions_std[i]}\n")
                f.write(f"R: {recalls[i]} +/- {recalls_std[i]}\n")
                f.write(f"F1: {fscores[i]} +/- {fscores_std[i]}\n")
                f.write(f"\n")
                
            else:
                f.write(f"------ RESULTS for tolerance {tol}, alpha {alpha_values[i-1]}, jump {jump_values} and penalty {penalty_values} ----------\n")
                f.write(f"P: {precisions[i]} +/- {precisions_std[i]}\n")
                f.write(f"R: {recalls[i]} +/- {recalls_std[i]}\n")
                f.write(f"F1: {fscores[i]} +/- {fscores_std[i]}\n")
                f.write(f"\n")
    
    
    x = np.arange(len(alpha_values)+1)
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
    ax.set_xticklabels(['baseline'] + alpha_values)  
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

TIME_SIGS = bps_fh.BPSFH.TIME_SIGS
dataset = Path("C:/Users/Usuario/Documents/Universidad/1 trabajo/BPS-FH/BPS_FH_Dataset")
tolerance = ["1_beat", "1_bar"]
alg = "pelt" # "pelt" or "window"
level = "mid" # "high", "mid" or "low"
csv = bps_fh.BPSFH(dataset).parse_anns(level)

# Optimal values for pelt and high level
alpha_values = [2.3]
jump_values = [1.5]
penalty_values = 4

# Optimal values for pelt mid level
#alpha_values = [1]
#jump_values = [0.01]
#penalty_values = 0.5

# Optimal values for pelt low level
#alpha_values = [0.1]
#jump_values = [0.15]
#penalty_values = 0.1

# Optimal values for window and high level
#alpha_values = [1]
#jump_values = [0]
#penalty_valuess = 4

# Optimal values for window and mid level
#alpha_values = [1]
#jump_values = [0]
#penalty_valuess = 0.5

# Optimal values for window and low level
#alpha_values = [1]
#jump_values = [0]
#penalty_valuess = 0.1

for tol in tolerance:
    plt, border_matrix, ground_truth_matrix = create_plot(tol, level, alpha_values, jump_values, penalty_values, dataset, csv, alg)
    plt.savefig(f"{dataset}/bps_{alg}_{tol}_alpha_{level}", dpi=300)
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
plt.savefig(f"{dataset}/errors_bps_{alg}_{tol}_{level}.png", dpi=300)
plt.show()




