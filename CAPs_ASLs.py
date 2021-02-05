import pandas as pd

caps_asls_filename = 'ASL_ComuniISTAT_da_MRA_con_CAP_e_codISTAT.xlsx - ASL-Comuni con CAP e codISTAT.csv'


def csv_reader(csv_file: str) -> pd.DataFrame:
    """
    Reads and imports the csv file
    """
    df = pd.read_csv(
        csv_file
    )
    return df


def drop_unused_columns(df, col_list: list):
    """
    Just as the name suggests
    :return:
    """
    df = df[col_list]
    return df


def prepare_data(csv_file: str, col_list: list) -> pd.DataFrame:
    """

    """
    df = csv_reader(csv_file)
    df = drop_unused_columns(df, col_list)
    return df


if __name__ == "__main__":
    df = prepare_data(caps_asls_filename, ['COD_ASL', 'CAP'])
