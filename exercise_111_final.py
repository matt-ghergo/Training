import math
import pandas as pd
import numpy as np
import statsmodels.api as sm
import uncertainties
from uncertainties import umath
import plotly.graph_objects as go

rank_filename = [
    '30df_rank.csv',
    'abmtv_rank.csv',
    'cvcc_rank.csv',
    'fcpte_rank.csv',
    'ffww_rank.csv',
    'opavl_rank.csv',
    'pcmp_rank.csv',
    'postmates_rank.csv',
    'readit_rank.csv',
    'samp_rank.csv',
    'sapl_rank.csv',
    'sfal_rank.csv',
    'smptv_rank.csv',
    'spcml_rank.csv',
    'splice_rank.csv',
    'steps_rank.csv',
    'tcve_rank.csv',
    'vault_rank.csv',
    'vmme_rank.csv',
    'yoga_rank.csv',
    'over_rank.csv',
    'st_rank.csv',
    'stash_rank.csv',
    'chase_rank.csv',
    'robin_rank.csv',
    'navy_rank.csv',
    'usbank_rank.csv',
    'regions_rank.csv',
    'capital_rank.csv',
]

download_filename = [
    'bendingspoons-download.csv',
]

real_data_filename = [
    'postmates_true.csv',
    'over_true.csv',
    'st_true.csv'
]


def csv_rank_reader(csv_file: str) -> pd.DataFrame:
    """
    Reads csv data about ranking from Sensor Tower
    """
    rank_data = pd.read_csv(
        csv_file,
        encoding="utf-16",
        sep="\t",
        skiprows=1,
        header=0
    )
    return rank_data


def csv_rank_cleaner(csv_file: str) -> pd.DataFrame:
    """
    Reads csv data about ranking from Sensor Tower and then (1) it creates the Category variable, (2) it removes rows
    in excess and (3) renames consistently wrongly-assigned columns
    """
    rank_data = csv_rank_reader(csv_file)
    if 'topfreeapplications.1' in rank_data.iloc[0]:
        rank_data.rename(
            columns={'topfreeapplications': 'Overall Rank', 'topfreeapplications.1': 'Category Rank'},
            inplace=True
        )
        rank_data['Category'] = rank_data.iloc[0][5]
    else:
        rank_data.rename(
            columns={'topfreeapplications': 'Category Rank'},
            inplace=True
        )
        rank_data['Category'] = rank_data.iloc[0][4]
    rank_data.drop(
        [0],
        inplace=True
    )
    if 'Overall Rank' in rank_data:
        rank_data['Overall Rank'] = pd.to_numeric(rank_data['Overall Rank'])
    rank_data['Category Rank'] = pd.to_numeric(rank_data['Category Rank'])
    return rank_data


def csv_rank_merger(rank_filename: list) -> pd.DataFrame:
    """
    Merges all data about ranking in a unique dataframe
    """
    rank_data = [csv_rank_cleaner(csv_file) for csv_file in rank_filename]
    rank_df = pd.concat(
        rank_data,
        ignore_index=True
    )
    return rank_df


def csv_download_reader(csv_file: str) -> pd.DataFrame:
    """
    Reads csv data about downloads from Sensor Tower
    """
    download_df = pd.read_csv(
        csv_file,
        encoding="utf-16",
        sep="\t",
        skiprows=range(0, 7)
    )
    return download_df


def csv_download_merger(download_filename: list) -> pd.DataFrame:
    """
    Merges all data about downloads in a unique dataframe
    """
    download_data = [csv_download_reader(csv_file) for csv_file in download_filename]
    download_df = pd.concat(
        download_data,
        ignore_index=True
    )
    return download_df


def add_lags(dataframe: pd.DataFrame, var: str, times: int) -> None:
    """
    For any given var in a given dataframe, it lags it for the given number of days based on the variable 'Date'
    and creates the corresponding column in the dataframe
    """
    dataframe['Date'] = pd.to_datetime(dataframe['Date'])
    lagged_variable = (
        dataframe.set_index('Date')
            .groupby('App ID')[var]
            .shift(times, freq=pd.DateOffset(days=1))
    )
    dataframe.set_index(
        ['App ID', 'Date'],
        inplace=True
    )
    dataframe[var + ' ' + str(times * (-1))] = lagged_variable
    dataframe.reset_index(inplace=True)


def add_weekday_dummy(dataframe: pd.DataFrame) -> None:
    """
    Creates the dummy variables for the days of the week and the variable 'Week Day' in the chosen dataframe
    """
    weekdays = [
        'Saturday',
        'Sunday',
        'Monday',
        'Tuesday',
        'Wednesday',
        'Thursday',
        'Friday'
    ]

    for x in range(len(weekdays)):
        dataframe[weekdays[x]] = [1 if int(dataframe['Date'][y][-2:]) % 7 == x else 0 for y in
                                  range(len(dataframe['Date']))]
    dataframe['Week Day'] = [weekdays[int(dataframe['Date'][x][-2:]) % 7] for x in range(len(dataframe['Date']))]


def add_category_dummy(dataframe):
    """
    Creates the dummy variables for the categories. It assumes the variable 'Category' has already been defined
    """
    categories = set(dataframe['Category'])
    for x in categories:
        dataframe[x + '_dummy'] = [1 if dataframe['Category'][y] == x else 0 for y in range(len(dataframe['Category']))]


def prepare_data(rank_filename: list, download_filename: list, add_lag: bool = True, lagged_var: str = None,
                 weekday_dummy: bool = True,
                 category_dummy: bool = True) -> pd.DataFrame:
    """
    Reads, cleans and merges the downloads and the ranking dataframes together, includes the lags and the dummy variable
    and then returns the final dataframe
    :param rank_filename: the list of the csv files' name with data about ranking
    :param download_filename: the list of the csv files'  name with data about downloads
    :param add_lag: whether to add a variable lagged by one period or not
    :param lagged_var: the variable to lag
    :param weekday_dummy: includes or not dummies for the days of the week and the variable 'Week Day'
    :param category_dummy: includes or not dummies for the categories of the app
    :return:
    """
    rank_df = csv_rank_merger(rank_filename)
    download_df = csv_download_merger(download_filename)
    dataframe = pd.merge(
        rank_df,
        download_df,
        how="left",
        on=["App ID", "Date"]
    )
    if weekday_dummy == True:
        add_weekday_dummy(dataframe)
    if category_dummy == True:
        add_category_dummy(dataframe)
    if add_lag == True:
        add_lags(dataframe, lagged_var, 1)
    return dataframe


def reg(regressand, regressors, addconstant=True, missing='drop', print_summary=True, cov_type=None, cluster=None,
        panel=False, print_eigen=False, print_mse=False):
    """
    Performs a regression.
    :param regressand: the explained variable
    :param regressors: the explanatory variables
    :param addconstant: includes or not a constant
    :param missing: what to do with missing data, it drops them by defaults
    :param print_summary: whether to wrint summary or not
    :param cov_type: the covariance matrix; by default it assumes spherical disturbances, other options includes:
                     'HC0' (heteroskedasticity robust), 'cluster' (clustered SE), 'hac-panel' (robust to
                     heteroskedasticity and autocorrelation);
    :param cluster: the groups, it must be of the same dimension of the available data and sets clustered SEs by default
    :param panel: if not False, uses hac robust SEs; requires cluster to be defined
    :param print_eigen: if true, prints the eigenvalues
    :param print_mse: if true, prints the MSE (total, model, resid) of the model
    :return:
    """
    X_var = regressors
    if addconstant == True:
        X_var = sm.add_constant(X_var)
    y_var = regressand
    ols_model = sm.OLS(y_var, X_var, missing=missing)
    ols_reg = ols_model.fit() if cov_type == None \
        else ols_model.fit(cov_type=cov_type) if cluster == None \
        else ols_model.fit(cov_type='cluster', cov_kwds={'groups': cluster}, df_correction=True) if panel == False \
        else ols_model.fit(cov_type='hac-panel', cov_kwds={'groups': cluster, 'time': df['Date'], 'maxlags': 1})
    if print_summary == True:
        print(ols_reg.summary())
    if print_eigen == True:
        print(ols_reg.eigenvals)
    if print_mse == True:
        print(ols_reg.mse_total, ols_reg.mse_model, ols_reg.mse_resid)
    return ols_reg


def def_var_reg1(include_dummy=True):
    if include_dummy == True:
        X_var1 = [[
            math.log(df['Overall Rank'][x]),
            df['Monday'][x],
            df['Tuesday'][x],
            df['Wednesday'][x],
            df['Thursday'][x],
            df['Friday'][x],
            df['Saturday'][x],
            df['Sunday'][x],  # all dummies as no constant included in the regression
        ]
            for x in range(len(df['Overall Rank']))]
    elif include_dummy == False:
        X_var1 = [[
            math.log(df['Overall Rank'][x]),
        ]
            for x in range(len(df['Overall Rank']))]
    y_var1 = [math.log(y) for y in df['iPhone Downloads -1']]
    cluster1 = [df['App Name_x'][i] for i in range(len(df['Overall Rank'])) if
                math.isnan(df['Overall Rank'][i]) == False and math.isnan(df['iPhone Downloads -1'][i]) == False]
    reg1_var = [y_var1, X_var1, cluster1]
    return reg1_var


def model1(include_dummy=True):
    reg1_var = def_var_reg1(include_dummy=include_dummy)
    if include_dummy == True:
        reg1 = reg(reg1_var[0], reg1_var[1], addconstant=False, cov_type='hac-panel', cluster=reg1_var[2], panel=True)
    elif include_dummy == False:
        reg1 = reg(reg1_var[0], reg1_var[1], addconstant=True, cov_type='hac-panel', cluster=reg1_var[2], panel=True)
    return reg1


def def_var_reg2():
    X_var2 = [[
        df['Monday'][x],
        df['Tuesday'][x],
        df['Wednesday'][x],
        df['Thursday'][x],
        df['Friday'][x],
        df['Saturday'][x],
        df['Sunday'][x],
        math.log(df['Category Rank'][x]),
        df['Photo & Video_dummy'][x] * math.log(df['Category Rank'][x]),
        df['Health & Fitness_dummy'][x] * math.log(df['Category Rank'][x]),
        df['Utilities_dummy'][x] * math.log(df['Category Rank'][x]),
        df['Books_dummy'][x] * math.log(df['Category Rank'][x]),
        df['Photo & Video_dummy'][x],
        df['Health & Fitness_dummy'][x],
        df['Utilities_dummy'][x],
        df['Books_dummy'][x],
    ]
        for x in range(len(df['Overall Rank']))]
    y_var2 = [math.log(y) for y in df['iPhone Downloads -1']]
    cluster2 = [df['App Name_x'][i] for i in range(len(df['Overall Rank'])) if
                math.isnan(df['Overall Rank'][i]) == False and math.isnan(df['iPhone Downloads -1'][i]) == False]
    reg2_var = [y_var2, X_var2, cluster2]
    return reg2_var


def model2():
    reg2_var = def_var_reg2()
    reg2 = reg(reg2_var[0], reg2_var[1], addconstant=False, cov_type='hac-panel', cluster=reg2_var[2], panel=True)
    return reg2


def def_var_reg3(Category):
    X_var3 = [[
        df['Category Rank'][x],
    ]
        for x in range(len(df['Overall Rank']))
        if df['Category'][x] == Category
    ]
    y_var3 = [df['Overall Rank'][y] for y in range(len(df['Overall Rank'])) if df['Category'][y] == Category]
    cluster3 = [df['App Name_x'][i] for i in range(len(df['Overall Rank'])) if
                math.isnan(df['Overall Rank'][i]) == False and math.isnan(df['Category Rank'][i]) == False and
                df['Category'][i] == Category]
    reg3_var = [y_var3, X_var3, cluster3]
    return reg3_var


def model3(Category):
    reg3_var = def_var_reg3(Category)
    reg3 = reg(reg3_var[0], reg3_var[1], addconstant=True, cov_type='hac-panel', cluster=reg3_var[2], panel=True)
    return reg3


def predict(predictors, model, loglinear=True):
    b_params = uncertainties.correlated_values(model.params, model.cov_params_default)
    predicted_y = np.matmul(predictors, b_params)
    if loglinear == True:
        predicted_y = [umath.pow(math.e, x) for x in predicted_y]
    return predicted_y


def predict_total(predictors, model):
    daily_downloads = predict(predictors, model)
    total_downloads = sum(daily_downloads) * (31 / 30)
    return total_downloads


def predictor_model1(App_name, include_dummy=True):
    if include_dummy == True:
        X1 = [
            [
                math.log(df['Overall Rank'][x]),
                df['Monday'][x],
                df['Tuesday'][x],
                df['Wednesday'][x],
                df['Thursday'][x],
                df['Friday'][x],
                df['Saturday'][x],
                df['Sunday'][x],
            ]
            for x in range(len(df['Overall Rank']))
            if df['App Name_x'][x] == App_name
        ]
    elif include_dummy == False:
        X1 = sm.add_constant([
            [
                math.log(df['Overall Rank'][x]),
            ]
            for x in range(len(df['Overall Rank']))
            if df['App Name_x'][x] == App_name
        ]
        )
    return X1


def predictor_model2(App_name):
    X2 = [
        [
            df['Monday'][x],
            df['Tuesday'][x],
            df['Wednesday'][x],
            df['Thursday'][x],
            df['Friday'][x],
            df['Saturday'][x],
            df['Sunday'][x],
            math.log(df['Category Rank'][x]),
            df['Photo & Video_dummy'][x] * math.log(df['Category Rank'][x]),
            df['Health & Fitness_dummy'][x] * math.log(df['Category Rank'][x]),
            df['Utilities_dummy'][x] * math.log(df['Category Rank'][x]),
            df['Books_dummy'][x] * math.log(df['Category Rank'][x]),
            df['Photo & Video_dummy'][x],
            df['Health & Fitness_dummy'][x],
            df['Utilities_dummy'][x],
            df['Books_dummy'][x],
        ]
        for x in range(len(df['Overall Rank']))
        if df['App Name_x'][x] == App_name
    ]
    return X2


def predictor_model3(App_name):
    X3 = sm.add_constant([[
        df['Category Rank'][x],
    ]
        for x in range(len(df['Overall Rank']))
        if df['App Name_x'][x] == App_name
    ]
    )
    return X3


def predict_postmates(include_dummy=True):
    predictors = predictor_model1('Postmates - Food Delivery', include_dummy=include_dummy)
    if include_dummy == True:
        prediction = predict(predictors, model1_dummy)
    elif include_dummy == False:
        prediction = predict(predictors, model1_nodummy)
    return prediction


def predict_over_direct():
    predictors = predictor_model2('Over: Graphic Design Maker')
    prediction = predict(predictors, model2)
    return prediction


def predict_overall_rank(App_Name, model):
    predictors = predictor_model3(App_Name)
    prediction = predict(predictors, model, loglinear=False)
    return prediction


def add_predicted_rank(dataframe, App_name, model):
    predicted_rank = predict_overall_rank(App_name, model)
    for x in range(len(dataframe['Overall Rank'])):
        if dataframe['App Name_x'][x] == App_name:
            dataframe['Overall Rank'][x] = predicted_rank[x % 31].n


def predict_over_indirect(model):
    add_predicted_rank(df, 'Over: Graphic Design Maker', model)
    predictors = predictor_model1('Over: Graphic Design Maker')
    prediction = predict(predictors, model1_dummy)
    return prediction


def predict_st(model):
    add_predicted_rank(df, 'Spending Tracker', model)
    predictors = predictor_model1('Spending Tracker')
    prediction = predict(predictors, model1_dummy)
    return prediction


def real_data():
    real_data = csv_download_merger(real_data_filename)
    return real_data


def plot_prediction_vs_real(real_data, predicted_data1, predicted_data2=None, App_Name=None, name1=None, name2=None,
                            name3=None):
    real_data_app = [real_data['iPhone Downloads'][x] for x in range(len(real_data['iPhone Downloads'])) if
                     real_data['App Name'][x] == App_Name]
    fig = go.Figure(
        data=go.Scatter(
            x=df['Date'][:31],
            y=[predicted_data1[x].n for x in range(len(predicted_data1))],
            error_y=dict(
                type='data',  # value of error bar given in data coordinates
                array=[predicted_data1[x].s for x in range(len(predicted_data1))],
                visible=True),
            name=name1
        ),
        layout_yaxis_range=[0, 1.25 * max(real_data_app)])

    if predicted_data2 != None:
        fig.add_trace(go.Scatter(x=df['Date'][:31],
                                 y=[predicted_data2[x].n for x in range(len(predicted_data2))],
                                 error_y=dict(
                                     type='data',  # value of error bar given in data coordinates
                                     array=[predicted_data2[x].s for x in range(len(predicted_data2))],
                                     visible=True),
                                 mode='lines',
                                 name=name2))

    fig.add_trace(go.Scatter(x=df['Date'][:31],
                             y=real_data_app,
                             mode='lines',
                             name=name3))

    fig.update_layout(
        title="Actual Downloads vs Predicted Downloads for " + str(App_Name) + ' for October 2017',
        yaxis_title="Daily Downloads",
        legend_title="Legend",
    )

    fig.show()


if __name__ == "__main__":
    df = prepare_data(rank_filename, download_filename, add_lag=True, lagged_var='iPhone Downloads')
    model1_dummy = model1()
    model1_nodummy = model1(include_dummy=False)
    model2 = model2()
    model3_finance = model3('Finance')
    model3_photovideo = model3('Photo & Video')
    predict_postmates_download_dummy = predict_postmates()
    predict_postmates_download_nodummy = predict_postmates(include_dummy=False)
    predict_over_download_direct = predict_over_direct()
    predict_over_download_indirect = predict_over_indirect(model3_photovideo)
    predict_st_download = predict_st(model3_finance)
    real_data = real_data()
    plot_prediction_vs_real(real_data, predict_postmates_download_dummy, predict_postmates_download_nodummy,
                            'Postmates - Food Delivery', 'Prediction considering weekly seasonality',
                            'Prediction ignoring weekly seasonality', 'Actual Downloads')
    plot_prediction_vs_real(real_data, predict_over_download_direct, predict_over_download_indirect,
                            'Over: Graphic Design Maker', 'Prediction using the direct approach',
                            'Prediction using the indirect approach', 'Actual downloads')
    plot_prediction_vs_real(real_data, predict_st_download, None, 'Spending Tracker', 'Predicted downloads', None,
                            'Actual downloads')

print(
    'The real values for the downloads of Postmates, Over and Spending Tracker are 237,179, 35,653 and 29,641 \nrespect'
    'ively.\nOur estimate for the downloads of Postmates is ' + str(
        int(sum(predict_postmates_download_dummy).n * 31 / 30)) + '.\n'
                                                                  'Our estimate for the downloads of Over is ' + str(
        int(sum(predict_over_download_direct).n * 31 / 30)) + ' using the direct approach and '
    + str(int(sum(
        predict_over_download_indirect).n * 31 / 30)) + ' with the indirect one.\nOur estimation for the downloads of Spending Tracker'
                                                        ' is ' + str(int(sum(predict_st_download).n * 31 / 30)) + '.'
)
