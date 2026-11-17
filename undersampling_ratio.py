# UNDERSAMPLING, TIENE SOLO I BORDI DEI PIXEL DELLE CLASSI MAGGIORI 
# KERNEL DIVERSI PER CLASSI DIVERSE
# USA IL RATIO PER PENALIZZARE ANCORA DI PIU' LE CLASSI DOMINANTI

import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm
from glob import glob

# ======================================================
# CONFIGURAZIONE
# ======================================================

DATASET_ROOT = r"./data/Harmonized_dense"
OUTPUT_ROOT  = r"./data/Harmonized_dense_undersample_ratio_only_train_2"

GT_KEYWORD = "_L2R_cl_"
GT_EXT = ".tif"          # cambia in ".png" se le GT sono png

IGNORE_INDEX = 0

# Kernel per classe dominante
# 7 (Marine Water) -> k=15
# 10 (Turbid Water) -> k=25
# 11 (Shallow Water) -> k=25
KERNEL_BY_CLASS = {
    # 2: 25, # dense
    7: 15,
    10: 35,
    11: 25
}

# Percentuale di pixel da TENERE dentro il contesto (per classe)
KEEP_RATIO_BY_CLASS = {
    # 2: 0.30, # dense
    7: 0.30,
    10: 0.30, # MIRKO: l'ho rimessa io e ho messo 0.30 al posto di 0.20
    11: 0.30
}

# KERNEL_BY_CLASS = {
#     7: 15,
#     10: 35,
#     11: 25
# }

# # Percentuale di pixel da TENERE dentro il contesto (per classe)
# KEEP_RATIO_BY_CLASS = {
#     7: 0.50,
#     10: 0.20, # MIRKO: l'ho rimessa io e ho messo 0.50 al posto di 0.20
#     11: 0.50
# }


# Seed per rendere riproducibile il sottocampionamento casuale
RANDOM_SEED = 0


# ======================================================
# FUNZIONI
# ======================================================

def dilate(mask: np.ndarray, k: int) -> np.ndarray:
    kernel = np.ones((k, k), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)


def keep_random_subset(mask: np.ndarray, keep_ratio: float, rng: np.random.Generator) -> np.ndarray:
    """
    Dato un mask booleano, tiene solo una frazione (keep_ratio) dei True.
    Ritorna un nuovo mask booleano con un sottoinsieme casuale.
    """
    if keep_ratio >= 1.0:
        return mask

    idx = np.flatnonzero(mask)
    n = idx.size
    if n == 0:
        return mask

    k = int(np.floor(n * keep_ratio))
    if k <= 0:
        return np.zeros_like(mask, dtype=bool)

    sel = rng.choice(idx, size=k, replace=False)
    out = np.zeros_like(mask, dtype=bool)
    out.flat[sel] = True
    return out


def undersample_gt(gt: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Boundary-aware undersampling con kernel e keep_ratio per classe:
    - crea una maschera 'minority' (tutto tranne ignore e classi dominanti)
    - per ogni classe dominante:
        * calcola contesto = dilate(minority, kernel)
        * keep = (classe & contesto)
        * tiene solo una percentuale keep_ratio di keep
        * il resto della classe -> IGNORE
    """
    gt_new = gt.copy()

    # Unione di tutte le classi dominanti
    dominant_union = np.zeros_like(gt, dtype=bool)
    for cls in KERNEL_BY_CLASS.keys():
        dominant_union |= (gt == cls)

    # Minoritarie: tutto tranne ignore e dominanti
    minority = (~dominant_union) & (gt != IGNORE_INDEX)

    # Se non ci sono minoritarie, elimina tutte le dominanti
    # if minority.sum() == 0:
    #     gt_new[dominant_union] = IGNORE_INDEX
    #     return gt_new

    # Processa ciascuna classe dominante separatamente
    for cls, k in KERNEL_BY_CLASS.items():
        cls_mask = (gt == cls)
        if not cls_mask.any():
            continue

        context2keep = dilate(minority, k).astype(bool)
        if context2keep.sum() > 0:
            keep = cls_mask & context2keep
        else:
            keep = cls_mask

        keep_ratio = KEEP_RATIO_BY_CLASS.get(cls, 1.0)
        keep_subset = keep_random_subset(keep, keep_ratio, rng)

        # tutto ciò che è della classe ma non nel subset -> ignore
        gt_new[cls_mask & ~keep_subset] = IGNORE_INDEX

    return gt_new


# ======================================================
# MAIN
# ======================================================

def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    rng = np.random.default_rng(RANDOM_SEED)

    ROIs_split = np.genfromtxt(os.path.join(DATASET_ROOT, 'splits', 'train_X.txt'),dtype='str')
    tiles = glob(os.path.join(DATASET_ROOT,'*'))

    for scene in tqdm(sorted(os.listdir(DATASET_ROOT)), desc="Scenes"):
        scene_path = os.path.join(DATASET_ROOT, scene)
        if not os.path.isdir(scene_path):
            continue

        for res in ["10", "20", "60"]:
            res_path = os.path.join(scene_path, res)
            if not os.path.isdir(res_path):
                continue

            out_res_path = os.path.join(OUTPUT_ROOT, scene, res)
            os.makedirs(out_res_path, exist_ok=True)

            for fname in os.listdir(res_path):
                in_path = os.path.join(res_path, fname)
                out_path = os.path.join(out_res_path, fname)

                # is fname in train?
                fname_split = fname.split('_')
                crop_name = '_'.join((fname_split[0], fname_split[1], fname_split[-1].split('.')[0]))

                if crop_name in ROIs_split:
                    # -------- FILE NON GT → COPIA --------
                    if GT_KEYWORD not in fname or not fname.endswith(GT_EXT):
                        if not os.path.exists(out_path):
                            shutil.copy2(in_path, out_path)
                        continue

                    # -------- FILE GT → UNDERSAMPLING --------
                    gt = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
                    if gt is None:
                        raise RuntimeError(f"Errore nel caricare {in_path}")
                    
                    # if 'Scene_4/10/Scene_4_L2R_cl_18.tif' in in_path:
                    #     import ipdb; ipdb.set_trace()

                    gt_new = undersample_gt(gt, rng).astype(np.uint8)
                    cv2.imwrite(out_path, gt_new)
                else:
                    if not os.path.exists(out_path):
                        shutil.copy2(in_path, out_path)

                        
    print("\n✔ Undersampling completato (k/ratio per classe applicati).")


if __name__ == "__main__":
    main()
