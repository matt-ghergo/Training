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


def clean_CAP(df, column):
    """
    In some regions the CAPs are written without the initial zeros; we add the zeros so that all CAPs have 5 chars
    For cities with multiple CAPs (eg Ancona) they have substituted the variable part with x's (eg Ancona has CAPs from
    60121 to 60131 and you have 601xx in the data. I have replaced that with the old CAP (with two zeros instead of the
    x's, eg for Ancona is 60100) as they all belong to the same ASL
    """
    seq = range(len(df[column]))
    for x in seq:
        if 'x' in df[column][x]:
            df[column][x] = df[column][x].replace('x', '0')
        while len(df[column][x]) < 5:
            df[column][x] = '0' + df[column][x]
    return df


def four_digitizer(df, CAP_column):
    """
    Given a column of a DataFrame, returns a column with the first 4 characters of each element of that column
    """
    four_digit_CAP = [df[CAP_column][x][:4] for x in range(len(df[CAP_column]))]
    return four_digit_CAP


def asl_cap(df):
    """
    Creates a column of unique strings for all the couples of ASL and CAP_4 (a copy of the CAP column with only the
    first four digits for each element)
    """
    seq = range(len(df['CAP']))
    df['asl_cap'] = [str(df['REG_ASL'][x]) + ' ' + df['CAP_4'][x] for x in seq]
    asl_cap = list(df['asl_cap'].drop_duplicates())
    return asl_cap


def prepare_data(csv_file: str, col_list: list) -> pd.DataFrame:
    """
    Returns a nice and clean dataframe with only the columns we need
    """
    df = csv_reader(csv_file)
    df = drop_unused_columns(df, col_list)
    df = clean_CAP(df, 'CAP')
    df['CAP_4'] = four_digitizer(df, 'CAP')
    return df


def CAP_univocal(df, cap_col, list):
    """
    We have a list of all couples of ASL and CAP_4. If the last 4 characters (those of CAP_4) appear more than once, it
    means that  asking 4 digits of the CAP is not sufficient to return a unique ASL, otherwise it is. This function
    checks for this condition and if it applies it extract all the CAPs that start with those 4 digits.
    Eventually, returns the list of all the CAPs that satisfy that condition.
    """
    inst = [x[-4:] for x in list]
    list_cap = []
    for x in range(len(inst)):
        if inst.count(inst[x]) == 1:
            for y in df[cap_col].unique():
                if y[:4] == inst[x]:
                    list_cap.append(y)
    return list_cap


if __name__ == "__main__":
    df = prepare_data(caps_asls_filename, ['REG_ASL', 'CAP'])
    asl_cap = asl_cap(df)
    list_cap = CAP_univocal(df, 'CAP', asl_cap)
    no_cap = len(list_cap)
    print(no_cap)

# By printing list_cap you can read the list of caps for which it is sufficient to ask the first four digits to
# univocally determine the ASL they refer to.
# If you wanna know for how many CAPs asking the first four digits is sufficient, all one needs to do is to print no_cap
# and that will provide the total number of CAPs.
# NB: Big cities are considered as one CAP only, for a more precise asseessment one needs to adjust for the Multicap
# cities here: http://www.comuni-italiani.it/cap/multicap.html