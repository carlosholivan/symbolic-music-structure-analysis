from pathlib import Path
import os
import numpy as np
import pandas as pd
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


#_______Beethoven___________

from musicaiz.datasets.bps_fh import BPSFH

dataset = Path("H:/INVESTIGACION/Datasets/functional-harmony-master/BPS_FH_Dataset")

data = BPSFH(dataset)

anns = ["high", "mid", "low"]

print("_______Beethoven_________")
for ann in anns:
    csv = data.parse_anns(anns=ann)

    counts = []
    for file in list(dataset.glob("*/*.mid")):
        # count boundaries
        counts.append(len(csv[file.stem].index))

    mean_counts_file = np.asarray(counts).mean()
    std_counts_file = np.asarray(counts).std()
    print("Total number of boundaries: ", sum(counts))
    print("Mean number of boundaries per file: ", mean_counts_file)
    print("Standard deviation of boundaries per file: ", std_counts_file)


#_______SWD___________
from musicaiz.rhythm import TimeSignature

# Create constant with the name of the file and the tempo
TIME_SIGS = {
        "Schubert_D911-01": "2/4", "Schubert_D911-02": "6/8", "Schubert_D911-03": "4/4", "Schubert_D911-04": "4/4", 
        "Schubert_D911-05": "3/4", "Schubert_D911-06": "3/4", "Schubert_D911-07": "2/4", "Schubert_D911-08": "3/4",
        "Schubert_D911-09": "3/8", "Schubert_D911-10": "2/4", "Schubert_D911-12": "2/4",
        "Schubert_D911-13": "6/8", "Schubert_D911-14": "3/4", "Schubert_D911-15": "2/4", "Schubert_D911-16": "3/4",
        "Schubert_D911-17": "12/8", "Schubert_D911-18": "4/4", "Schubert_D911-19": "6/8", "Schubert_D911-20": "2/4",
        "Schubert_D911-21": "4/4", "Schubert_D911-22": "2/4", "Schubert_D911-23": "3/4", "Schubert_D911-24": "3/4",
    }

valid_files = list(TIME_SIGS.keys())

def parse_anns(midis_path, anns_path) -> dict:
    """Converts the bar index float 
    annotations in 8th notes."""
    table = {}

    empiezo_en_cero = True

    for file in list(midis_path.glob("*.mid")):
        if file.stem not in valid_files:
            continue
        table[file.name.split(".")[0]] = [] #crea una tabla vacía

        gt = pd.read_csv( # lee el fichero csv
            Path(anns_path, file.name.split(".")[0] + ".csv"),
            sep=';', header=None, engine='python'
        )
        gt_aux = gt

        if(empiezo_en_cero): #para que empiece en cero siempre, aunque no lo haga en el fichero
            gt_aux[0][1] = 0

        key = gt_aux[2][1:] #completar el resto de la tabla
        gt_aux[2][1:] = key #gt_aux es la tabla con las estructuras

        for i, ann in enumerate(gt_aux[0]):
            if i == 0: # se salta el start de la tabla
                continue
            
            if "." not in str(ann): #si no hay punto, se añade y lo divide en entero y decimal
                ann = float(ann) # convierte 0 en 0.0
                bar_idx = int(float(("0." + str(ann).split(".")[0]))) # 0.0 parte entera, lo convierte a entero = 0
                float_beat_idx = float("0." + str(ann).split(".")[1]) # 0.0 parte decimal
            else: # hace lo mismo pero ya había parte entera y parte decimal
                bar_idx = int(float(str(ann).split(".")[0]))-1 # solo la parte entera
                float_beat_idx = float("0." + str(ann).split(".")[1]) # 0.X la parte decimal
            num_bar = int(TIME_SIGS[file.name.split(".")[0]].split("/")[0])
            denom_bar = int(TIME_SIGS[file.name.split(".")[0]].split("/")[1])
            #eights_per_bar = num_bar * 8 / denom_bar # Número de corcheas por compás 
            timesig = TimeSignature(TIME_SIGS[file.name.split(".")[0]])
            eights_per_bar = timesig.eights
            #eights_per_bar = file_object.time_signature_changes[-1]["time_sig"].eights # Número de corcheas por compás
            eight_idx = round(eights_per_bar * float_beat_idx) # número de corcheas que hay en la parte decimal
            total_eights = bar_idx * eights_per_bar + eight_idx  # 2 8th notes in one beat. Se suma las corcheas de la parte entera y de la parte decimal
            gt_aux[0][i] = int(total_eights) # convierte a entero el valor de eights y lo guarda en la tabla
            

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
            num_bar = int(TIME_SIGS[file.name.split(".")[0]].split("/")[0])
            denom_bar = int(TIME_SIGS[file.name.split(".")[0]].split("/")[1])
            #eights_per_bar = num_bar * 8 / denom_bar # Número de corcheas por compás 
            timesig = TimeSignature(TIME_SIGS[file.name.split(".")[0]])
            eights_per_bar = timesig.eights
            eight_idx = round(eights_per_bar * float_beat_idx)
            if denom_bar % 4 == 0:
                total_eights = bar_idx * eights_per_bar + eight_idx
            elif denom_bar % 8 == 0:
                total_eights = bar_idx * eights_per_bar + eight_idx
            else:
                raise ValueError("Not valid time sig.")
            gt_aux[1][i] = int(total_eights)
    
        table[file.name.split(".")[0]] = gt_aux
    return table

bps_dataset = Path("H:/INVESTIGACION/Datasets/Shubert_Winterreise")
midis_path = Path(bps_dataset, "01_RawData/score_midi")
anns_path = Path(bps_dataset, "02_Annotations/ann_score_structure")

csv = parse_anns(midis_path, anns_path)

print("_______SWD_________")
counts = []
for key in csv.keys():
    # count boundaries
    counts.append(len(csv[key].index))

mean_counts_file = np.asarray(counts).mean()
std_counts_file = np.asarray(counts).std()
print("Total number of boundaries: ", sum(counts))
print("Mean number of boundaries per file: ", mean_counts_file)
print("Standard deviation of boundaries per file: ", std_counts_file)