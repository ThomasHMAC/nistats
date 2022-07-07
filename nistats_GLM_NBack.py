#!/usr/bin/env python

import argparse
import logging
import os
import sys

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.plotting import plot_anat, plot_epi, plot_img, plot_stat_map, show
from nistats.design_matrix import (
    check_design_matrix,
    make_first_level_design_matrix,
    make_second_level_design_matrix,
)
from nistats.first_level_model import FirstLevelModel
from nistats.model import TContrastResults
from nistats.reporting import plot_contrast_matrix, plot_design_matrix
from nistats.second_level_model import SecondLevelModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")

file_handler = logging.FileHandler("nistat_first_lvl.log")
file_handler.setLevel(logging.CRITICAL)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def format_events(event_file):
    """
    Argument:
        event_file Full path to events.tsv file

    Output:
        event_df Newly formatted events dataframe
    """
    # Read in tsv file
    try:
        event_df = pd.read_csv(event_file, delimiter="\t")
    except FileNotFoundError:
        logger.critical("Could not load the task tsv file")
        sys.exit(1)
    # Filter out the one and threeback block from event_df
    event = event_df[["onset", "duration", "trial_type"]]
    block_type = event[
        event["trial_type"].str.match("onebackblock|threebackblock")
    ].copy()
    # block_type["trial_type"].replace(" ", "_", regex=True, inplace=True)

    # Extract hit, miss, and false alarm from dataframe
    mask_hit = (event_df["correct_response"] == 1) & (
        event_df["participant_response"] == 1
    )
    # mask_miss = (event_df["correct_response"] == 1) & (
    #    event_df["participant_response"] == 0
    # )
    mask_false = (event_df["correct_response"] == 0) & (
        event_df["participant_response"] == 1
    )
    event_df.loc[mask_hit, "trial_type"] = (
        event_df["trial_type"].astype(str) + "_hit"
    )
    # event_df.loc[mask_miss,"trial_type"] =  event_df['trial_type'].astype(str) + "_miss"
    event_df.loc[mask_false, "trial_type"] = (
        event_df["trial_type"].astype(str) + "_false"
    )
    event_df = event_df[["trial_type", "duration", "onset"]]
    events = event_df[event_df["trial_type"].str.endswith(("hit", "false"))]

    # Merge the timing of oneback and threeback together
    df_combined = pd.concat([block_type, events])
    df_combined = df_combined.reset_index(drop=True)

    return df_combined


def extract_confounds(confound_path):
    """
    Arguments:
        confound_path    Full path to confounds.tsv
        confound_vars    List of confound variables to extract

    Outputs:
        confound_df
    """
    # Load in data using pandas and extract the relevant columns
    try:
        confound_df = pd.read_csv(confound_path, delimiter="\t")
    except FileNotFoundError:
        logger.critical("Could not load the task tsv file")
        sys.exit(1)
    # Filter out desire regressors
    confound_vars = [
        col
        for col in confound_df.columns
        if col.startswith(
            ("WhiteMatter", "CSF", "GlobalSignal", "X", "Y", "Z", "Rot")
        )
    ]
    confound_df = confound_df[confound_vars]
    confound_df_new = confound_df.copy()

    for i in confound_vars:
        print(i + "_derivative1")
        confound_df_new.loc[:, i + "_derivative1"] = confound_df_new[i].diff()
        confound_df_new.loc[:, i + "_power2"] = confound_df_new[i].pow(2)
        confound_df_new.loc[:, i + "_derivative1_power2"] = (
            confound_df_new[i].diff().pow(2)
        )

    confound_df_new = confound_df_new[:].fillna(0)

    # During the initial stages of a functional scan there is a strong signal decay artifact
    # The first few TRs are very high intensity signals that don't reflect the rest of the scan
    # so they are dropped

    # tr_drop = 4
    # confound_df = confound_df.loc[tr_drop:].reset_index(drop=True)

    # demean all the confounds
    for col in confound_df_new.columns:
        confound_df_new[col] = confound_df_new[col].sub(
            confound_df_new[col].mean()
        )

    # Return confound matrix
    return confound_df_new


# Create a design matrix


def get_design_matrix(fmri_img, event_file, confound_file):
    """
    Arguments:
    fmri_img        full path to functional data
    event_file      full path to event type tsv file

    Output:

    dm              a full design matrix
    """

    event_df = format_events(event_file)
    confound_df = extract_confounds(confound_file)
    func_img = nib.load(fmri_img)
    n_scans = func_img.shape[-1]
    tr = 2
    frame_times = np.arange(n_scans) * tr
    dm = make_first_level_design_matrix(
        frame_times,
        event_df,
        drift_model=drift_model,
        drift_order=drift_order,
        add_regs=confound_df,
        add_reg_names=list(confound_df.columns),
        hrf_model=hrf_model,
    )
    return dm


def main():

    parser = argparse.ArgumentParser(
        description="Run FirstLevelModel from nistats"
    )
    parser.add_argument(
        "input_dir", type=str, help="path to subject preprocessed directory"
    )
    parser.add_argument(
        "output_dir", type=str, help="path to generate first level outputs"
    )
    parser.add_argument(
        "sub_id", type=str, help="a string of subject ID i.e sub-CMHHCT201"
    )

    args = parser.parse_args()
    in_path = args.input_dir
    out_path = args.output_dir
    sub_id = args.sub_id

    global tr
    global tr_drop
    global drift_model
    global drift_order
    global hrf_model
    global noise_model
    global period_cut
    global event_df
    global confound_df
    global frame_times

    # Make output directory
    out_dir = os.path.join(out_path, "first_lvl", sub_id)
    try:
        os.makedirs(out_dir)
        print("Directory ", out_dir, " Created ")
    except FileExistsError:
        print("Directory ", out_dir, " already exists")

    files = os.listdir(in_path)
    nifti_img = os.path.join(
        in_path,
        next(
            f
            for f in files
            if f.endswith("nbk_acq-CMH_run-01_bold_Atlas_s6.nii")
        ),
    )
    print(nifti_img)
    confound_file = os.path.join(
        in_path,
        next(
            f
            for f in files
            if f.endswith("nbk_acq-CMH_run-01_bold_confounds.tsv")
        ),
    )
    print(confound_file)
    event_file = os.path.join(
        in_path,
        next(f for f in files if f.endswith("NBACK.tsv")),
    )
    print(event_file)
    func_img = nib.load(nifti_img)
    n_scans = func_img.shape[-1]
    t_r = 2
    frame_times = np.arange(n_scans) * t_r
    # design matrix input
    drift_model = "polynomial"
    drift_order = 5
    hrf_model = "spm + derivative + dispersion"

    # first level model input
    noise_model = "ar1"

    dm = get_design_matrix(nifti_img, event_file, confound_file)
    dm_outpath = os.path.join(
        out_dir, "{}_ses-01_task-nbk_dm.tsv".format(sub_id)
    )
    # print(dm)
    dm.to_csv(dm_outpath, sep="\t")
    # define glm parameters
    first_lvl_glm = FirstLevelModel(
        t_r=t_r,
        hrf_model=hrf_model,
        drift_model=drift_model,
        drift_order=drift_order,
        noise_model=noise_model,
        standardize=False,
        minimize_memory=False,
        mask_img=False,
    )
    try:
        first_lvl_glm = first_lvl_glm.fit(nifti_img, design_matrices=dm)
    except FileNotFoundError:
        print("Error in fitting model for" + sub_id)

    # Compute contrast of interests
    contrast_matrix = np.eye(dm.shape[1])
    basic_contrasts = dict(
        [(column, contrast_matrix[i]) for i, column in enumerate(dm.columns)]
    )
    basic_contrasts["threeback_minus_oneback"] = (
        basic_contrasts["threebackblock"] - basic_contrasts["onebackblock"]
    )
    contrasts_id = [
        "threeback_minus_oneback",
        "onebackblock",
        "threebackblock",
    ]

    for i, val in enumerate(contrasts_id):
        t_map = first_lvl_glm.compute_contrast(
            basic_contrasts[contrasts_id[i]], stat_type="t", output_type="stat"
        )
        print(t_map)
        subject_tmap_path = os.path.join(
            out_dir,
            "{}_ses-01_task-nbk_{}_t_map.nii.gz".format(
                sub_id, contrasts_id[i]
            ),
        )
        t_map.to_filename(subject_tmap_path)


if __name__ == "__main__":
    main()
