import os
import glob
import csv
import nibabel as nib
import numpy as np
from scipy.ndimage import sobel
from concurrent.futures import ProcessPoolExecutor, as_completed

############################################
# STEP 1: DEFINE THE METRIC FUNCTIONS
############################################

def load_nifti(file_path):
    """Load .nii or .nii.gz file and return a 3D numpy array of image data (float32)."""
    nii = nib.load(file_path)
    data = nii.get_fdata(dtype=np.float32)
    return data

def compute_noise_metric(volume):
    """
    Simple noise metric: standard deviation in a 'soft tissue' HU range.
    Adjust the HU range as needed for your data.
    """
    lower_bound = -100
    upper_bound = 200
    
    mask = (volume >= lower_bound) & (volume <= upper_bound)
    # If not enough voxels in that range, return NaN
    if np.sum(mask) < 1000:
        return np.nan
    
    noise_value = np.std(volume[mask])
    return noise_value

def compute_sharpness_metric(volume):
    """
    Compute sharpness (edge strength) via Sobel gradients, averaged across slices.
    """
    if volume.ndim != 3:
        return np.nan
    
    sharpness_scores = []
    num_slices = volume.shape[-1]
    for z in range(num_slices):
        slice_2d = volume[..., z]
        gx = sobel(slice_2d, axis=0)
        gy = sobel(slice_2d, axis=1)
        grad_mag = np.sqrt(gx**2 + gy**2)
        sharpness_scores.append(np.mean(grad_mag))
    
    return np.mean(sharpness_scores)

def compute_artifact_metric(volume):
    """
    A simplistic artifact measure: 1 - average correlation between consecutive slices.
    Lower correlation => higher artifact => bigger artifact metric.
    """
    num_slices = volume.shape[-1]
    if num_slices < 2:
        return np.nan

    correlations = []
    for z in range(num_slices - 1):
        slice1 = volume[..., z].ravel()
        slice2 = volume[..., z + 1].ravel()
        corr = np.corrcoef(slice1, slice2)[0, 1]
        correlations.append(corr)
    
    if len(correlations) == 0:
        return np.nan
    
    mean_corr = np.mean(correlations)
    return 1 - mean_corr

def compute_intensity_outliers_metric(volume):
    """
    For typical CT in range [-1000, 1000], measure fraction of voxels outside [-1200, 1200].
    Adjust as appropriate.
    """
    min_val, max_val = -1000, 1000
    total_voxels = volume.size
    outliers = np.sum((volume < min_val) | (volume > max_val))
    fraction_outliers = outliers / total_voxels
    return fraction_outliers

def compute_quality_score(file_path):
    """
    Load the volume, compute all metrics, combine them, and return a dict.
    """
    data = load_nifti(file_path)
    
    noise_val = compute_noise_metric(data)
    sharpness_val = compute_sharpness_metric(data)
    artifact_val = compute_artifact_metric(data)
    intensity_outliers_val = compute_intensity_outliers_metric(data)
    
    # Combine them with a simple weighting approach
    w_sharp = 1.0
    w_noise = 1.0
    w_artifact = 1.0
    w_outliers = 1.0
    
    quality_score = (
        w_sharp * sharpness_val
        - w_noise * noise_val
        - w_artifact * artifact_val
        - w_outliers * intensity_outliers_val
    )
    
    return {
        "filename": file_path.split("/")[-2],
        "noise": noise_val,
        "sharpness": sharpness_val,
        "artifact": artifact_val,
        "intensity_outliers": intensity_outliers_val,
        "quality_score": quality_score
    }

############################################
# STEP 2: MULTIPROCESSING + RESUME LOGIC
############################################

def get_nii_files(folder_path):
    """
    Return a list of .nii or .nii.gz files within the folder (recursively).
    """
    pattern_nii_gz = os.path.join(folder_path, "*", "ct.nii.gz")
    # pattern_nii = os.path.join(folder_path, "*", "*.nii")
    file_list_nii_gz = glob.glob(pattern_nii_gz, recursive=True)
    # file_list_nii = glob.glob(pattern_nii, recursive=True)
    return file_list_nii_gz 

def read_already_processed(csv_path):
    """
    Read the 'filename' column from CSV, return a set of processed filenames.
    Used to skip re-processing.
    """
    processed = set()
    if not os.path.exists(csv_path):
        return processed
    
    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 'filename' is the column name storing the file's basename
            filename = row["filename"]
            processed.add(filename)
    return processed

def process_ct_file(file_path):
    """
    Helper function for multiprocessing. 
    Returns the metrics dictionary for a single file, handles errors.
    """
    try:
        return compute_quality_score(file_path)
    except Exception as e:
        # Return partial info plus error for logging
        return {
            "filename": file_path.split("/")[-2],
            "noise": np.nan,
            "sharpness": np.nan,
            "artifact": np.nan,
            "intensity_outliers": np.nan,
            "quality_score": np.nan,
            "error": str(e)
        }

def batch_process_folder(
    folder_path,
    csv_path="quality_metrics.csv",
    num_workers=4
):
    """
    1) Gathers all NIfTI files from folder.
    2) Checks which have been processed already (via CSV).
    3) Uses multiprocessing for unprocessed scans.
    4) Writes to CSV **on the fly** so partial results are saved even if interrupted.
    """
    # 1) Gather files
    all_files = get_nii_files(folder_path)
    all_files = sorted(all_files)
    
    if len(all_files) == 0:
        print(f"No .nii or .nii.gz files found in '{folder_path}'")
        return
    
    # 2) Find already processed
    processed_filenames = read_already_processed(csv_path)
    unprocessed_files = [
        f for f in all_files if os.path.basename(f) not in processed_filenames
    ]
    
    if len(unprocessed_files) == 0:
        print("All files appear to be processed already. Nothing to do.")
        return
    
    print(f"Found {len(all_files)} total files. {len(processed_filenames)} already processed.")
    print(f"{len(unprocessed_files)} files remain to be processed.")
    
    # 3) Process unprocessed files with multiprocessing
    fieldnames = ["filename", "noise", "sharpness", "artifact", "intensity_outliers", "quality_score"]
    
    # Check if CSV exists, so we know whether we need to write a header
    csv_exists = os.path.exists(csv_path)
    
    with open(csv_path, mode=("a" if csv_exists else "w"), newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        # If file did not exist before, write header
        if not csv_exists:
            writer.writeheader()
            f.flush()
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_ct_file, fpath): fpath for fpath in unprocessed_files}
            
            # as_completed yields futures as they finish
            for future in as_completed(futures):
                result = future.result()
                # Write this result immediately to the CSV
                # If there's an 'error' key, it won't match the fieldnames exactly,
                # but we can handle it or ignore it. For simplicity, let's pop it out:
                if "error" in result:
                    # You could log the error somewhere else if you like
                    error_msg = result.pop("error")
                
                writer.writerow(result)
                # Flush after each write to ensure it's saved on disk
                f.flush()
    
    print("Processing complete (or interrupted). Partial results are in the CSV.")

############################################
# STEP 3: EXECUTION EXAMPLE
############################################
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch process CT scans for quality metrics, writing CSV on the fly.")
    parser.add_argument("--input_folder", type=str, required=True, help="Folder containing .nii or .nii.gz scans.")
    parser.add_argument("--csv_path", type=str, default="quality_metrics.csv", help="Output CSV file path.")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel processes.")
    args = parser.parse_args()
    
    batch_process_folder(
        folder_path=args.input_folder,
        csv_path=args.csv_path,
        num_workers=args.workers
    )