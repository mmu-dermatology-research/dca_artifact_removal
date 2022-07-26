import numpy as np
import pandas as pd
import os

def main():
    path_to_csv = "../Data/heatmap_csv_files/extracted_brightness_data.csv"
    csv = pd.read_csv(path_to_csv)

    models = list(set(csv["Model"]))
    sets = list(set(csv["Test_Set"]))
    sizes = list(set(csv["DCA_Size"]))

    for model in models:
        for s in sets:
            for size in sizes:
                print(model, s, size)

                temp_data = csv.copy()
                temp_data = temp_data[temp_data["Model"] == model]
                temp_data = temp_data[temp_data["Test_Set"] == s]
                temp_data = temp_data[temp_data["DCA_Size"] == size]

                internal_rms_mean = temp_data["Internal_Brightness_RMS"].mean()
                external_rms_mean = temp_data["External_Brightness_RMS"].mean()
                internal_rms_std = temp_data["Internal_Brightness_RMS"].std()
                external_rms_std = temp_data["External_Brightness_RMS"].std()

                internal_avg_mean = temp_data["Internal_Brightness_Mean"].mean()
                external_avg_mean = temp_data["External_Brightness_Mean"].mean()
                internal_avg_std = temp_data["Internal_Brightness_Mean"].std()
                external_avg_std = temp_data["External_Brightness_Mean"].std()

                print("RMS Internal")
                print("Mean: ", internal_rms_mean)
                print("Std: ", internal_rms_std)
                print("RMS External")
                print("Mean: ", external_rms_mean)
                print("Std: ", external_rms_std)

                print("Avg Brightness Internal")
                print("Mean: ", internal_avg_mean)
                print("Std: ", internal_avg_std)
                print("Avg Brightness External")
                print("Mean: ", external_avg_mean)
                print("Std: ", external_avg_std)

                print()




if __name__ == "__main__":
    main()
