import os
from tqdm import tqdm

tags = ['Fix', 'Nav', 'Raw', 'ACC_UNCAL', 'GYR_UNCAL', 'MAG_UNCAL', 
        'PSR', 'ECG', 'PPG', 'ACC']

dataset_folder = "/mnt/c/Users/vmangr/OneDrive - TUNI.fi/Shared/GEOLOC/dataset/"
survey_list = os.listdir(dataset_folder)

exclude_survey = ['S1', 'S2', 'S3', 'S4', '.archive']
exclude_acquisition = ['A1', 'A2', 'A3', 'A5', 'A6', 'A7', 'A8' 'A9', 'A10', 'A11', 'A12', 'Novatel_reference_for_A9_A12']
exclude_device = ['A52', 'GP7', 'SW6']

for i in tqdm(range(len(survey_list))):
    survey = survey_list[i]
    survey_folder = f"{dataset_folder}{survey}/"
    if not os.path.isdir(f"{survey_folder}") or survey in exclude_survey:
        continue
    acq_list = os.listdir(survey_folder)
    for acq in acq_list:
        device_folder = f"{survey_folder}{acq}/"
        if not os.path.isdir(f"{device_folder}") or acq in exclude_acquisition:
            continue
        device_list = os.listdir(device_folder)
        for device in device_list:
            if device in exclude_device:
                continue
            log_folder = f"{device_folder}{device}/"
            file_list = os.listdir(log_folder)
            if not file_list:
                continue
            for file in file_list:
                if ".csv" in file:
                    os.remove(f"{log_folder}{file}")
            log = os.listdir(log_folder)
            if "log_mimir" not in log[0]:
                continue
            logfile = f"{log_folder}{log[0]}"
            for tag in tags:
                tagfile = f"{log_folder}{tag}.csv"
                #command = f"awk -F , '$1 == \"ACC_UNCAL\" {{print $2\",\"$4\",\"$5\",\"$6}}' {input_folder}{mfile} > {output_folder}{mfile[:-4]}_ACC_UNCAL.txt"
                #command = f"awk -F , '$1 == \"{tag}\" {{print}}' \"{logfile}\" > \"{tagfile}\""
                command = f"awk -F , '$1 == \"{tag}\" {{ print >> \"{tagfile}\"}}' \"{logfile}\""
                os.system(f'{command}')


# for mfile in input_files:

#     # Extract ACC
#     command = f"awk -F , '$1 == \"ACC_UNCAL\" {{print $2\",\"$4\",\"$5\",\"$6}}' {input_folder}{mfile} > {output_folder}{mfile[:-4]}_ACC_UNCAL.txt"
#     os.system(command)

#     # Extract bias
#     command = f"awk -F , '$1 == \"ACC_UNCAL\" {{print $2\",\"$7\",\"$8\",\"$9}}' {input_folder}{mfile} > {output_folder}{mfile[:-4]}_ACC_BIAS.txt"
#     os.system(command)