import os
import shutil
from tqdm import tqdm

rtkconv_path = "C:\\Users\\vmangr\\Desktop\\Softwares\\RTKLIB_bin-rtklib_2.4.3_b34\\bin\\convbin.exe"
dataset_folder = "C:\\Users\\vmangr\\OneDrive - TUNI.fi\\Shared\\GEOLOC\\dataset\\"
survey_list = os.listdir(dataset_folder)

exclude_survey = ['.archive'] # 'S2', 'S3', 'S4', 'S5',
exclude_acquisition = ['Novatel_reference_for_A7_to_A10'] # 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 
exclude_device = ['A52', 'GP7', 'SW6', 'GPW', 'AWINDA']

for i in tqdm(range(len(survey_list))):
    survey = survey_list[i]
    survey_folder = f"{dataset_folder}{survey}\\"
    if not os.path.isdir(f"{survey_folder}") or survey in exclude_survey:
        continue
    acq_list = os.listdir(survey_folder)
    for acq in acq_list:
        device_folder = f"{survey_folder}{acq}\\"
        if not os.path.isdir(f"{device_folder}") or acq in exclude_acquisition:
            continue
        device_list = os.listdir(device_folder)
        for device in device_list:
            if device in exclude_device:
                continue
            log_folder = f"{device_folder}{device}\\"
            file_list = os.listdir(log_folder)
            if not file_list:
                continue
            for file in file_list:
                if ".ubx" in file:
                    break
            logfile = f"{log_folder}{file}"

            # Create new folder
            if os.path.exists(f"{log_folder}RINEX"):
                shutil.rmtree(f"{log_folder}RINEX")
            #os.mkdir(f"{log_folder}RINEX")

            # Call rtkconv
            command = f"{rtkconv_path} \"{logfile}\" -od -os -o \"{log_folder}\gnss.rnx\""
            os.system(f'{command}')


# for mfile in input_files:

#     # Extract ACC
#     command = f"awk -F , '$1 == \"ACC_UNCAL\" {{print $2\",\"$4\",\"$5\",\"$6}}' {input_folder}{mfile} > {output_folder}{mfile[:-4]}_ACC_UNCAL.txt"
#     os.system(command)

#     # Extract bias
#     command = f"awk -F , '$1 == \"ACC_UNCAL\" {{print $2\",\"$7\",\"$8\",\"$9}}' {input_folder}{mfile} > {output_folder}{mfile[:-4]}_ACC_BIAS.txt"
#     os.system(command)