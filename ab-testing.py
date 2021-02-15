import numpy as np
import pandas as pd
from bsp_query_builder import Query
from bsp_bq.pandas import read_gbq
from datetime import datetime
from uncertainties import unumpy
import plotly.graph_objects as go


def query_numerator(start_date: str, end_date: str, actions_end_date: str) -> Query:
    """
    Builds a query for data about free trials started between start_date and actions_end_date for people who downloaded
    the app between start_date and end_date
    """
    return (
        Query()
            .from_("lumen2-42.Layer1_Apps.com_path36_SpliceFree_Purchases")
            .select(("COUNT(*)", "free_trials"), "days_from_install", "experiment_paywall_close_button")
            .where(
            "install_timestamp > TIMESTAMP('" + start_date + "')",
            "install_timestamp < TIMESTAMP('" + end_date + "')",
            "data_acquisition_timestamp > TIMESTAMP('" + start_date + "')",
            "data_acquisition_timestamp < TIMESTAMP('" + actions_end_date + "')",
            "is_free=FALSE",
            "is_baseline = FALSE",
            "installed_before_pico = FALSE",
            "(experiment_paywall_close_button = 1 OR experiment_paywall_close_button = 2)",
            "type = 'free trial'"
        )
            .group_by("experiment_paywall_close_button", "days_from_install")
            .order_by("experiment_paywall_close_button", "days_from_install")
    )


def query_denominator(start_date: str, end_date: str) -> Query:
    """
    Builds a query for data about downloads between start_date and end_date
    """
    return (
        Query()
            .from_("lumen2-42.Layer1_Apps.com_path36_SpliceFree_Installs")
            .select(("COUNT(*)", "downloads"), ("DATE(install_timestamp)", "day"), "experiment_paywall_close_button")
            .where(
            "install_timestamp > TIMESTAMP('" + start_date + "')",
            "install_timestamp < TIMESTAMP('" + end_date + "')",
            "is_free=FALSE",
            "is_baseline = FALSE",
            "installed_before_pico = FALSE",
            "(experiment_paywall_close_button = 1 OR experiment_paywall_close_button = 2)",
        )
            .group_by("day", "experiment_paywall_close_button")
            .order_by("experiment_paywall_close_button", "day")
    )


def df_numerator(start_date: str, end_date: str, actions_end_date: str) -> pd.DataFrame:
    """
    Runs the query built by query_numerator(...) and returns a Dataframe
    """
    query = query_numerator(start_date, end_date, actions_end_date)
    df_numerator = read_gbq(query, project='lumen2-42')
    return df_numerator


def df_denominator(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Runs the query built by query_denominator(...) and returns a Dataframe
    """
    query = query_denominator(start_date, end_date)
    df_denominator = read_gbq(query, project='lumen2-42')
    return df_denominator


def days_between(d1: str, d2: str) -> int:
    """
    Returns the difference in days between two dates in the %Y-%m-%d format
    """
    d1 = datetime.strptime(d1, '%Y-%m-%d')
    d2 = datetime.strptime(d2, '%Y-%m-%d')
    return abs((d2 - d1).days)


def df_start() -> pd.DataFrame:
    """
    Helper function that creates a dataframe with a single column and one observation for each day in our observation
    period. This is needed to avoid jumps as not all days in our observation period have nonzero values
    """
    data = {'days_from_install': range(days_between(start_date, actions_end_date))}
    df = pd.DataFrame(data)
    return df


def df_num() -> pd.DataFrame:
    """
    Helper function that downloads, reshapes and cleans data about free trials (the numerator)
    """
    df_num = df_numerator(start_date, end_date, actions_end_date)
    df_num_seg_1 = df_num[df_num.experiment_paywall_close_button == 1]
    df_num_seg_2 = df_num[df_num.experiment_paywall_close_button == 2]
    df_num = pd.merge(df_num_seg_1, df_num_seg_2, how='outer', on='days_from_install', suffixes=('_1', '_2'))
    df_num = pd.merge(df_start(), df_num, how='inner', on='days_from_install')
    df_num['free_trials_1'] = df_num['free_trials_1'].fillna(0)
    df_num['free_trials_2'] = df_num['free_trials_2'].fillna(0)
    df_num = df_num.drop(['experiment_paywall_close_button_1', 'experiment_paywall_close_button_2'], axis=1)
    return df_num


def df_den() -> pd.DataFrame:
    """
    Helper function that downloads, reshapes and cleans data about downloads (the denominator)
    """
    df_den = df_denominator(start_date, end_date)
    df_den_seg_1 = df_den[df_den.experiment_paywall_close_button == 1]
    df_den_seg_2 = df_den[df_den.experiment_paywall_close_button == 2]
    df_den = pd.merge(df_den_seg_1, df_den_seg_2, how='outer', on='day', suffixes=('_1', '_2'))
    return df_den


def running_total(df: pd.DataFrame, column: str) -> list:
    """
    Helper function that takes a column of differentials and returns the accumulation of them
    """
    running_total = []
    total = 0
    for n in range(len(df[column])):
        total += df[column][n]
        running_total.append(total)
    return running_total


def running_total_inverse(df: pd.DataFrame, column: str) -> list:
    """
    Helper function that associates to each level of longevity (in days) the number of downloads for which such degree
    of longevity is observed
    """
    list = running_total(df, column)
    runningtotal_int = []
    for n in range(days_between(start_date, actions_end_date)):
        if n - days_between(end_date, actions_end_date) < 0:
            runningtotal_int.append(list[days_between(start_date, end_date) - 1])
        else:
            runningtotal_int.append(list[days_between(start_date, actions_end_date) - n - 1])
    return runningtotal_int


def dataframe() -> pd.DataFrame:
    """
    Return the dataframe with all the data needed for the estimation
    """
    df = df_num()
    df_downloads = df_den()
    df['denominator_1'] = running_total_inverse(df_downloads, 'downloads_1')
    df['denominator_2'] = running_total_inverse(df_downloads, 'downloads_2')
    return df


def mme_conversion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform a MM estimation for the conversion to free trials
    """
    df['conversion_ft_1_n'] = df['free_trials_1'] / df['denominator_1']
    df['conversion_ft_2_n'] = df['free_trials_2'] / df['denominator_2']
    df['var_1'] = df['conversion_ft_1_n'].apply(lambda x: x * (1 - x)) / df['denominator_1']
    df['se_1'] = df['var_1'].apply(np.sqrt)
    df['var_2'] = df['conversion_ft_2_n'].apply(lambda x: x * (1 - x)) / df['denominator_2']
    df['se_2'] = df['var_2'].apply(np.sqrt)
    df['conversion_ft_1'] = unumpy.uarray(df['conversion_ft_1_n'], df['se_1'])
    df['conversion_ft_2'] = unumpy.uarray(df['conversion_ft_2_n'], df['se_2'])
    df['conversion_ft_cumul_1'] = running_total(df, 'conversion_ft_1')
    df['conversion_ft_cumul_2'] = running_total(df, 'conversion_ft_2')
    return df


def plot_cumul(df: pd.DataFrame):
    """
    Plot the cumulative for the two segments of the experiment
    """
    fig = go.Figure(
        data=go.Scatter(
            x=df['days_from_install'],
            y=[x.n for x in df['conversion_ft_cumul_1']],
            error_y=dict(
                type='data',
                array=[1.96 * x.s for x in df['conversion_ft_cumul_1']],
                visible=True),
            name="Segment 1",
            mode="markers"
        ))
    fig.add_trace(go.Scatter(
        x=df['days_from_install'],
        y=[x.n for x in df['conversion_ft_cumul_2']],
        error_y=dict(
            type='data',
            array=[1.96 * x.s for x in df['conversion_ft_cumul_2']],
            visible=True),
        name="Segment 2",
        mode="markers"
    ))
    fig.update_layout(
        title="Conversion to Free Trial comparison for the two segments",
        xaxis_title='Days from Install',
        yaxis_title="Conversion to Free Trial",
        legend_title="Legend",
    )
    fig.show()


if __name__ == '__main__':
    start_date = '2019-07-19'
    end_date = '2019-08-19'
    actions_end_date = '2019-09-19'
    df = dataframe()
    df_with_estimates = mme_conversion(df)
    plot_cumul(df_with_estimates)