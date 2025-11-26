import os
import pandas as pd
from sodapy import Socrata
from utils import DOMAIN, OFFENSES_ID, SCHEMA, VICTIMS_ID


def fetch_data(limit=500000, save_path="lapd_offenses_victims_merged.parquet"):
    """
    Fetch offenses and victims from Socrata, merge on caseno, and save to parquet.
    """
    # If parquet already exists, read it and apply merged schema instead of querying
    if os.path.exists(save_path):
        print(f"Loading data from existing parquet: {save_path}")
        df = pd.read_parquet(save_path)
        return df
    
    client = Socrata(DOMAIN, None)
    offenses = client.get(OFFENSES_ID, limit=limit)
    df_off = pd.DataFrame.from_records(offenses)
    df_off = apply_schema(df_off, SCHEMA["offenses"])
    victims = client.get(VICTIMS_ID, limit=limit)
    df_vic = pd.DataFrame.from_records(victims)
    df_vic = apply_schema(df_vic, SCHEMA["victims"])

    if "caseno" not in df_off.columns or "caseno" not in df_vic.columns:
        raise ValueError("caseno not found in one of the datasets")
    # Remove duplicate columns from df_vic (keep only caseno and victim-specific columns)
    merge_cols = df_vic.columns.difference(df_off.columns)
    merge_cols = merge_cols.append(pd.Index(["caseno"]))  # Keep caseno for merge key
    df_vic = df_vic[merge_cols]

    # merge
    df = df_off.merge(df_vic, on="caseno", how="left")
    df = cast_types(df)
    # keep data with date_occ after 2000-01-01
    df['date_occ'] = pd.to_datetime(df['date_occ'], errors='coerce')
    df = df[df['date_occ'] >= pd.Timestamp('2024-01-01')]
    df.to_parquet(save_path, index=False)

    return df

def apply_schema(df, schema_dict):
    """Enforce schema types on a dataframe."""
    for col, dtype in schema_dict.items():
        if col not in df.columns:
            continue
        try:
            if dtype == "string":
                df[col] = df[col].astype("string")
            elif dtype == "int":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            elif dtype == "float":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif dtype == "datetime":
                df[col] = pd.to_datetime(df[col], errors="coerce")
        except Exception as e:
            print(f"Warning converting {col} to {dtype}: {e}")
    return df

def cast_types(df):
    """Apply complex type casting and transformations case by case."""
    
    if "time_occ" in df.columns:
        # Convert HHMM integer format (e.g., 855 -> 08:55, 2150 -> 21:50)
        df["time_occ"] = df["time_occ"].astype(str).str.zfill(4)  # Pad with zeros
        df["time_occ"] = pd.to_datetime(df["time_occ"], format="%H%M", errors="coerce").dt.time

    if "vict_age" in df.columns:
        # Convert victim age to numeric, coerce errors to NaN
        df["vict_age"] = pd.to_numeric(df["vict_age"], errors="coerce").astype("Int64")

    if "area_name" in df.columns:
        # Convert to categorical for memory efficiency
        df["area_name"] = df["area_name"].astype("category")
        df.drop(columns=["area"], inplace=True)

    if "nibr_description" in df.columns:
        # Convert to categorical for memory efficiency
        df["nibr_description"] = df["nibr_description"].astype("category")
        df.drop(columns=["nibr_code"], inplace=True)

    if "premis_desc" in df.columns:
        # Convert to categorical for memory efficiency
        df["premis_desc"] = df["premis_desc"].astype("category")
        df.drop(columns=["premis_cd"], inplace=True)

    if "weapon_desc" in df.columns:
        # Convert to categorical for memory efficiency
        df["weapon_desc"] = df["weapon_desc"].astype("category")
        df.drop(columns=["weapon_used_cd"], inplace=True)

    if "status_desc" in df.columns:
        # Convert to categorical for memory efficiency
        df["status_desc"] = df["status_desc"].astype("category")
        df.drop(columns=["status"], inplace=True)

    # Convert mixed-type string columns to string (handles str/float NaN issue)
    string_cols = [
        "rpt_dist_no",
        "group",
        "crime_against",
        "premis_desc",
        "weapon_desc",
        "uniquevictimno",
        "vict_descent",
        "vict_sex",
        "victim_type"
    ]
    
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype("object")

    # Convert Yes/No columns to boolean
    yes_no_cols = [
        "victim_shot",
        "domestic_violence_crime",
        "hate_crime",
        "gang_related_crime",
        "transit_related_crime",
        "homeless_victim_crime",
        "homeless_suspect_crime",
        "homeless_arrestee_crime"
    ]
    for col in yes_no_cols:
        if col in df.columns:
            df[col] = df[col].str.upper() == "YES"
    
    return df