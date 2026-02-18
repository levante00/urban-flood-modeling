import pandas as pd


def preprocess_dynamic_df(
    df: pd.DataFrame,
    fillna_value: float = 0.0,
    sort_columns: tuple[str, str] = ("node_id", "timestep"),
) -> pd.DataFrame:
    """Standardize dynamic node dataframe columns and ordering."""
    out = df.copy().fillna(fillna_value)

    rename_map: dict[str, str] = {}
    for col in out.columns:
        c = col.lower()
        if "node" in c and "idx" in c:
            rename_map[col] = "node_id"
        elif "time" in c:
            rename_map[col] = "timestep"
        elif "water" in c and "level" in c:
            rename_map[col] = "water_level"
        elif "rain" in c:
            rename_map[col] = "rainfall"

    out = out.rename(columns=rename_map)
    return out.sort_values(list(sort_columns)).reset_index(drop=True)
