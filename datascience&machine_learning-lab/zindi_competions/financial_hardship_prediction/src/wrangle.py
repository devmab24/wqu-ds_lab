import pandas as pd

def wrangle(df):
    """
    Clean a dataframe.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    df = df.copy()

    # Remove duplicates
    df = df.drop_duplicates()

    # Standardize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
    )

    return df


    #BEFORE
# def wrangle(filepath):
#     """
#     Clean a dataframe.
#     Parameters
#     ----------
#     df : pd.DataFrame

#     Returns
#     -------
#     pd.DataFrame
#     """

#     df = pd.read_csv(filepath)

#     # Remove duplicate rows
#     df = df.drop_duplicates()

#     # Standardize column names
#     df.columns = (
#         df.columns
#         .str.strip()
#         .str.replace(" ", "_")
#     )

#     return df