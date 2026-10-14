import os
import numpy as np
import pandas as pd


def process_and_split_dataset(
    file_path, output_dir="samples", save_to_csv=True
):
    # 1. Read CSV (row 0 is the header)
    df = pd.read_csv(file_path)

    # 2. Extract column 4 (sin) and column 5 (cos)
    month_sin = df.iloc[:, 4].values
    month_cos = df.iloc[:, 5].values

    # 3. Decode month (1 to 12)
    angles = np.arctan2(month_sin, month_cos)
    angles = np.where(angles < 0, angles + 2 * np.pi, angles)
    months = np.round(angles * 12 / (2 * np.pi)).astype(int) + 1

    # 4. Detect H1 (Jan–Jun) or H2 (Jul–Dec)
    if np.median(months) <= 6:
        print(">> Detected: Jan–Jun dataset")

        # Drop July overflow (month > 6)
        dropped_count = np.sum(months > 6)
        if dropped_count > 0:
            print(f">> Dropped {dropped_count} overflow rows from July.")

        # Split: Jan to May and June
        df_main = df[months <= 5].copy()
        df_last = df[months == 6].copy()

    else:
        print(">> Detected: Jul–Dec dataset")

        # Drop January overflow (month < 7)
        dropped_count = np.sum(months < 7)
        if dropped_count > 0:
            print(
                f">> Dropped {dropped_count} overflow rows from January."
            )

        # Split: Jul to Nov and Dec
        df_main = df[(months >= 7) & (months <= 11)].copy()
        df_last = df[months == 12].copy()

    name_main, name_last = "train_trips.csv", "validation_trips.csv"

    # 5. Save with index=False (PREVENTS EXTRA COLUMN!)
    if save_to_csv:
        os.makedirs(output_dir, exist_ok=True)
        path_main = os.path.join(output_dir, name_main)
        path_last = os.path.join(output_dir, name_last)

        # -------------------------------------------------------------
        # CHANGED: index=False ensures NO extra column is added!
        # -------------------------------------------------------------
        df_main.to_csv(path_main, index=False)
        df_last.to_csv(path_last, index=False)

        print(
            f"Saved: '{path_main}' (Cols: {df_main.shape[1]}, Rows: {len(df_main)})"
        )
        print(
            f"Saved: '{path_last}' (Cols: {df_last.shape[1]}, Rows: {len(df_last)})"
        )

    return df_main, df_last


# ==========================================
# Run the script:
# ==========================================
if __name__ == "__main__":
    df_train, df_test = process_and_split_dataset(
        "trip_info_9_section_ver2_simplify_ultra_no_variance_2024_Jul_Dec.csv",
        output_dir="samples",
        save_to_csv=True,
    )