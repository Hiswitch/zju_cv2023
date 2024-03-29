#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

echo -e "\nYou need to register at https://smpl.is.tue.mpg.de"
read -p "Username (SMPL):" username
read -p "Password (SMPL):" password
username=$(urle $username)
password=$(urle $password)

mkdir -p data/body_models
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.1.0.zip' -O './bedlam_data/body_models/smpl.zip' --no-check-certificate --continue
unzip bedlam_data/body_models/smpl.zip -d bedlam_data/body_models/
mv bedlam_data/body_models/SMPL_python_v.1.1.0/smpl/models/basicmodel_f_lbs_10_207_0_v1.1.0.pkl bedlam_data/body_models/SMPL_python_v.1.1.0/smpl/models/SMPL_FEMALE.pkl
mv bedlam_data/body_models/SMPL_python_v.1.1.0/smpl/models/basicmodel_m_lbs_10_207_0_v1.1.0.pkl bedlam_data/body_models/SMPL_python_v.1.1.0/smpl/models/SMPL_MALE.pkl
mv bedlam_data/body_models/SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl bedlam_data/body_models/SMPL_python_v.1.1.0/smpl/models/SMPL_NEUTRAL.pkl


echo -e "\nYou need to register at https://bedlam.is.tue.mpg.de/"
read -p "Username (BEDLAM):" username
read -p "Password (BEDLAM):" password
username=$(urle $username)
password=$(urle $password)

mkdir -p bedlam_data/smpl_gt
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=bedlam_labels/smpl_ground_truth.zip' -O './bedlam_data/smpl_ground_truth.zip' --no-check-certificate --continue
unzip bedlam_data/smpl_ground_truth.zip -d bedlam_data/smpl_gt