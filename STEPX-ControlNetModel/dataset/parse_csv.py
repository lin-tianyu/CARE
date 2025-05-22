import pandas as pd
import nibabel as nib
import numpy as np
import os, sys
from tqdm import tqdm

# Extract patient ID and phase information
def extract_info(entry):
    parts = entry.split("_")
    patient_id = parts[0] + "_" + parts[1]  # UCSF_344PEp5z1NU format
    phase_info = parts[-1]  # Last part should contain phase info (e.g., 'noncontrast', 'contrast')
    return patient_id, phase_info

if __name__ == "__main__":
    filename = "ct_contrast_translation.xlsx"
    BDMAP_root = "/mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/"

    """Step 1: analyze and extract info """
    # df = pd.read_excel(filename)
    # print(df.head())
    # # Extracting the relevant column (assuming it's the first column)
    # column_name = df.columns[0]
    # data = df[column_name]
    # # Apply extraction function
    # df_extracted = pd.DataFrame(data.apply(extract_info).tolist(), columns=["Patient_ID", "Phase"])
    # # Generate statistics
    # patient_count = df_extracted["Patient_ID"].nunique()
    # phase_distribution = df_extracted["Phase"].value_counts()
    # # Display the extracted data and statistics
    # print(df_extracted.head())
    # df_extracted.to_excel("step1.xlsx")
    # # patient_count, phase_distribution

    """Step 2: group by patient id """
    # df_transformed = pd.read_excel("step1.xlsx")
    # # df_transformed = df_extracted.copy()
    # df_transformed["BDMAP_ID"] = df[df.columns[1]]  # Assign the BDMAP_ID from the original sheet

    # # Pivot the data so that each row corresponds to a patient and each column is a phase
    # df_pivot = df_transformed.pivot_table(index="Patient_ID", columns="Phase", values="BDMAP_ID", aggfunc=lambda x: list(x))

    # # Reset the index for better readability
    # df_pivot.reset_index(inplace=True)

    # # Display the transformed dataset
    # print(df_pivot.head())
    # df_pivot.to_excel("step2.xlsx")

    """Step 3: check shapes for selection (and get BDMAP_ID)"""
    # # choose phase by the maximum resolution
    # df_shapes = pd.read_excel("step2.xlsx")
    # # df_shapes = df_transformed.copy()
    # df_final = df_shapes.copy()
    # for row_idx, row in df_shapes.iterrows():
    #     print(f"Patient-{row_idx}: {row['Patient_ID']}")  # Access patient ID
    #     for col_idx, phase in enumerate(df_shapes.columns[2:]):  # Skip the first and second column (Patient_ID)
    #         bdmap_ids = row[phase]  # Get BDMAP_ID(s) for the phase
    #         if not isinstance(bdmap_ids, float) or not pd.isna(bdmap_ids):  # Ensure it's not NaN
    #             bdmap_ids = bdmap_ids if isinstance(bdmap_ids, list) else [bdmap_ids]  # Ensure it's a list
    #             print(f"\tPhase: {phase}")
    #             print(f"\t\tBDMAP_ID: {bdmap_ids[0]}")
    #             bdmap_shapes = [nib.load(os.path.join(BDMAP_root, bdmap_id, "ct.nii.gz")).shape for bdmap_id in list(eval(bdmap_ids[0]))]
    #             bdmap_res = list(map(lambda x: sum(x), bdmap_shapes))
    #             max_res_index = bdmap_res.index(max(bdmap_res))
    #             print(f"\t\tBDMAP_shapes: {bdmap_shapes}  -- max res --> {bdmap_shapes[max_res_index]}")
    #             df_shapes.iloc[row_idx, col_idx + 2] = str(bdmap_shapes[max_res_index])
    #             df_final.iloc[row_idx, col_idx + 2] = list(eval(bdmap_ids[0]))[max_res_index]
    #         else:
    #             print("### Nan:", row[1], phase)
            
    # df_final["same_res"] = df_shapes[df_shapes.columns[2:5]].apply(lambda row: row.nunique() == 1, axis=1)   # add same resolution flag
    # df_final = df_final.drop(columns=["Unnamed: 0"], errors="ignore")  # Won't error if column isn't there
    # df_final.to_excel("step3-ids.xlsx")
    # df_shapes["same_res"] = df_shapes[df_shapes.columns[2:5]].apply(lambda row: row.nunique() == 1, axis=1)  # add same resolution flag
    # df_shapes = df_shapes.drop(columns=["Unnamed: 0"], errors="ignore")  # Won't error if column isn't there
    # df_shapes.to_excel("step3-shapes.xlsx")


    """Step 4: softlinking nii.gz file"""
    df_shapes = pd.read_excel("step3-shapes.xlsx")
    df_final = pd.read_excel("step3-ids.xlsx")
    for row_idx, row in df_final.iterrows():
        print(row_idx, row["Patient_ID"])
        for col_idx, phase in enumerate(df_final.columns[2:]):
            bdmap_id = row[phase]
            if not isinstance(bdmap_id, float) or not pd.isna(bdmap_id):
                BDMAP_niigz_path = os.path.join(BDMAP_root, bdmap_id, 'ct.nii.gz')
                output_niigz_root = os.path.join("data", row[1].split('_')[1]+"_"+phase)
                os.makedirs(output_niigz_root, exist_ok=True)
                output_niigz_path = os.path.join(output_niigz_root, "ct.nii.gz")
                print(f"\t{BDMAP_niigz_path} --> {output_niigz_path}")
                # os.symlink(BDMAP_niigz_path, output_niigz_path)
            else:
                print("### Nan:", row[1], phase)

    """Step 5: transform to h5 file for fast loading """
    # NOTE: Go to `/ccvl/net/ccvl15/tlin67/Dataset_raw/FELIXtemp/nii2h5.py`.
    