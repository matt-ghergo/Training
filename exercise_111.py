import math
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm

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


# This function imports and cleans the data about ranking history for an app from a SensorTower CSV file
# It also creates the 'Category' variable in the dataframe and renames ambiguous variables where appropriate

def read_sensortower_rank_data(csv_file):
    rank_data = pd.read_csv(
        csv_file,
        encoding="utf-16",
        sep="\t",
        skiprows=1,
        header=0
    )

    if 'topfreeapplications.1' in rank_data.iloc[0]:
        rank_data.rename(
            columns={'topfreeapplications': 'Overall Rank', 'topfreeapplications.1': 'Category Rank'},
            inplace=True
        )
        rank_data['Category'] = rank_data.iloc[0][5]
        rank_data.drop(
            [0],
            inplace=True
        )
        rank_data['Overall Rank'] = pd.to_numeric(rank_data['Overall Rank'])
        rank_data['Category Rank'] = pd.to_numeric(rank_data['Category Rank'])

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
        rank_data['Category Rank'] = pd.to_numeric(rank_data['Category Rank'])

    return rank_data


# The files about BS apps' rank and about those we need an estimate for are exported and merged in a unique dataframe

rank_data = [read_sensortower_rank_data(filename) for filename in rank_filename]

rank_df = pd.concat(
    rank_data,
    ignore_index=True
)

# Now the files about downloads for our apps are imported and merged with the previous dataframe

download_df = pd.read_csv(
    'bendingspoons-download.csv',
    encoding="utf-16",
    sep="\t",
    skiprows=range(0, 7)
)

df = pd.merge(
    rank_df,
    download_df,
    how="left",
    on=["App ID", "Date"]
)

# A variable for the days of the week is created and accordingly 7 dummy variables one for each day

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
    df[weekdays[x]] = [1 if int(df['Date'][y][-2:]) % 7 == x else 0 for y in range(len(df['Date']))]  # the 7 dummies

df['Week Day'] = [weekdays[int(df['Date'][x][-2:]) % 7] for x in range(len(df['Date']))]

# Dummies for different categories are created as well

categories = set(df['Category'])

for x in categories:
    df[x + '_dummy'] = [1 if df['Category'][y] == x else 0 for y in range(len(df['Category']))]

# The following function creates lagged variable from the variables in our dataframe
# It might be useful as theoretically the rank variable is influenced by downloads in the past hours or days

df['Date'] = pd.to_datetime(df['Date'])


def lag(dataframe, var, times):
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


# As an example, the following variable has data for the downloads in the previous day
lag(df, 'iPhone Downloads', 1)


# This function draws a scatter plot to illustrate the co-movements of two variables on a Cartesian plane
# It also allows for the possibility to transform a variable using a mathematical function

def plot_function(dataframe, ind_var, dep_var, col_var=None, function_ind=None, function_dep=None, param_ind=None,
                  param_dep=None, trendline=None, x_axis='Independent Variable', y_axis='Dependent Variable'):
    def plot_var(dataframe, ind_var, dep_var, col_var):
        fig = px.scatter(
            dataframe,
            x=ind_var,
            y=dep_var,
            color=col_var,
            trendline=trendline
        )
        fig.show()

    dataframe[x_axis] = dataframe[ind_var] if function_ind == None else [
        function_ind(x, param_ind) if param_ind != None else function_ind(x) for x in
        dataframe[ind_var]]
    dataframe[y_axis] = dataframe[dep_var] if function_dep == None else [
        function_dep(y, param_dep) if param_dep != None else function_dep(y) for y in
        dataframe[dep_var]]

    plot_var(dataframe, dataframe[x_axis], dataframe[y_axis], dataframe[col_var] if col_var != None else None)


# This function simply performs a linear regression based on some given data

def reg(regressand, regressors, addconstant=True, missing='drop', print_summary=True, cov_type=None, cluster=None,
        print_eigen=True, print_mse=False):
    X_var = regressors
    if addconstant == True:
        X_var = sm.add_constant(X_var)
    y_var = regressand
    ols_model = sm.OLS(y_var, X_var, missing=missing)
    ols_reg = ols_model.fit() if cov_type == None else ols_model.fit(
        cov_type=cov_type) if cluster == None else ols_model.fit(cov_type='cluster', cov_kwds={'groups': cluster},
                                                                 df_correction=True)
    if print_summary == True:
        print(ols_reg.summary())
    if print_summary == True:
        print(ols_reg.eigenvals)
    if print_mse == True:
        print(ols_reg.mse_total, ols_reg.mse_model, ols_reg.mse_resid)
    return ols_reg


# plot_function(df, 'iPhone Downloads', 'Overall Rank', 'Week Day')
# plot_function(df, 'iPhone Downloads', 'Category Rank', 'Category')

# plot_function(df, 'iPhone Downloads', 'Category Rank', 'Category', function_ind=math.log, function_dep=math.log)
# plot_function(df, 'iPhone Downloads', 'Overall Rank', 'Week Day', function_ind=math.log, function_dep=math.log, trendline='ols')


# The following graph shows how the correlation between 'Overall Rank' and each category 'Category Rank' is almost
# perfect.

# plot_function(df, 'Overall Rank', 'Category Rank', 'Category', trendline='ols')


# If both 'Overall Rank' and the various ('Category Rank' multiplied by the appropriate category dummy) were included as
# regressors, our X'X would have some eigenvalues really close to zero which implies an extremely high variance in all
# our coefficient estimators. In other words, the model would not be able to distinguish between the variation driven
# by one variable and another. For this reason only regressions including one variable or another are performed, under
# the assumption that the uncommon element (which depends most likely on the structure of its market) is not correlated
# with the other regressors (the days of the week)


# For the first part of the exercise, it is required to provide an estimate of the Downloads for Postmates, for which
# both the category rank and the overall rank are known but for which BS has no apps in its category (Food & Drink). So
# the relationship between Downloads and Overall Rank is explored first and then the Downloads for Postmates are
# estimated exploiting the previous findings.

# The X matrix is defined

X_var1 = [[
    math.log(df['Overall Rank'][x]),
    # math.pow(math.log(df['Overall Rank'][x]), 2),
    df['Monday'][x],
    df['Tuesday'][x],
    df['Wednesday'][x],
    df['Thursday'][x],
    df['Friday'][x],
    df['Saturday'][x],
    df['Sunday'][x],
]
    for x in range(len(df['Overall Rank']))]

# The y vector is defined

y_var1 = [math.log(y) for y in df['iPhone Downloads']]

# The clustering dimension is defined (of the same length of the SE vectors)

cluster1 = [df['Week Day'][i] for i in range(len(df['Overall Rank'])) if
            math.isnan(df['Overall Rank'][i]) == False and math.isnan(df['iPhone Downloads'][i]) == False]

# The regression is performed

reg1 = reg(y_var1, X_var1, addconstant=False, cov_type='cluster', cluster=cluster1)

# The data matrix for the values to predict is defined

X_postmates = [
    [
        math.log(df['Overall Rank'][x]),
        # math.pow(math.log(df['Overall Rank'][x]), 2),
        df['Monday'][x],
        df['Tuesday'][x],
        df['Wednesday'][x],
        df['Thursday'][x],
        df['Friday'][x],
        df['Saturday'][x],
        df['Sunday'][x],
    ]
    for x in range(len(df['Overall Rank']))
    if df['App Name_x'][x] == 'Postmates - Food Delivery'
]

# The no. of download for Postmates are estimated and stored

postmates_daily_downloads = [math.pow(math.exp(1), x) for x in reg1.predict(X_postmates)]
postmates_est_downloads = sum(postmates_daily_downloads)

# For the second part of the exercise, it is required to provide an estimate of the downloads for Over, for which only
# the categorical rank is known but for which BS possesses apps in its category ('Photo and Video'). Two approaches are
# hereby explored: (1) a prediction based on the direct relationship between categorical rank and downloads; (2) first,
# the overall rank is predicted from the categorical one and then the relationship between overall rank and downloads is
# used to predict downloads based on the estimated overall rank for Over.


# Again, all the relevant variables for the regression are defined

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

y_var2 = [math.log(y) for y in df['iPhone Downloads']]

cluster21 = [df['Week Day'][i] for i in range(len(df['Overall Rank'])) if
             math.isnan(df['iPhone Downloads'][i]) == False]
cluster22 = [df['Category'][i] for i in range(len(df['Overall Rank'])) if
             math.isnan(df['iPhone Downloads'][i]) == False]

reg2 = reg(y_var2, X_var2, addconstant=False, cov_type='cluster', cluster=[cluster21, cluster22])

# The data matrix for the values to predict is defined

X_over1 = [
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
    if df['App Name_x'][x] == 'Over: Graphic Design Maker'
]

# The no. of download for Over are estimated and stored

over1_daily_downloads = [math.pow(math.exp(1), x) for x in reg2.predict(X_over1)]
over1_est_downloads = sum(over1_daily_downloads)

# It now follows an estimation of the downloads of Over using the second approach


# First, we define the relevant variable for the regression of the overall rank on the Photo & Video rank and perform it

X_var3 = [[
    df['Category Rank'][x],
]
    for x in range(len(df['Overall Rank']))
    if df['Category'][x] == 'Photo & Video'
]

y_var3 = [df['Overall Rank'][y] for y in range(len(df['Overall Rank'])) if df['Category'][y] == 'Photo & Video']

reg3 = reg(y_var3, X_var3, addconstant=True)

# The data matrix to estimate the overall rank for Over is defined

X_over_c_rank = [[
    df['Category Rank'][x],
]
    for x in range(len(df['Overall Rank']))
    if df['App Name_x'][x] == 'Over: Graphic Design Maker'
]

X_over_c_rank = sm.add_constant(X_over_c_rank)

# The predicted values for the Over overall rank are now put in the general dataframe

X_over_rank = [x for x in reg3.predict(X_over_c_rank)]

for x in range(len(df['Overall Rank'])):
    if df['App Name_x'][x] == 'Over: Graphic Design Maker':
        df['Overall Rank'][x] = X_over_rank[x % 31]

# Finally, an estimate for the Over downloads is provided and stored

X_over2 = [
    [
        math.log(df['Overall Rank'][x]),
        # math.pow(math.log(df['Overall Rank'][x]), 2),
        df['Monday'][x],
        df['Tuesday'][x],
        df['Wednesday'][x],
        df['Thursday'][x],
        df['Friday'][x],
        df['Saturday'][x],
        df['Sunday'][x],
    ]
    for x in range(len(df['Overall Rank']))
    if df['App Name_x'][x] == 'Over: Graphic Design Maker'
]

over2_daily_downloads = [math.pow(math.exp(1), x) for x in reg1.predict(X_over2)]
over2_est_downloads = sum(over2_daily_downloads)

# For the third part of the exercise, it is required to provide an estimate for the downloads of Spending tracker, for
# which only the categorical rank is known and for which BS possesses no apps in its category ('Finance'). An
# estimation approach identical to the second one used for Over is preferred, using data about the categorical and
# overall rank of non-BS Finance apps. The steps as well as the definition of the variables are identical to those in
# the second estimation of Over.


X_var3 = [[
    df['Category Rank'][x],
]
    for x in range(len(df['Overall Rank']))
    if df['Category'][x] == 'Finance'
]

y_var3 = [df['Overall Rank'][y] for y in range(len(df['Overall Rank'])) if df['Category'][y] == 'Finance']

reg3 = reg(y_var3, X_var3, addconstant=True)

X_st_c_rank = [[
    df['Category Rank'][x],
]
    for x in range(len(df['Overall Rank']))
    if df['App Name_x'][x] == 'Spending Tracker'
]

X_st_c_rank = sm.add_constant(X_st_c_rank)

X_st_rank = [x for x in reg3.predict(X_st_c_rank)]

for x in range(len(df['Overall Rank'])):
    if df['App Name_x'][x] == 'Spending Tracker':
        df['Overall Rank'][x] = X_st_rank[x % 31]

X_st = [
    [
        math.log(df['Overall Rank'][x]),
        # math.pow(math.log(df['Overall Rank'][x]), 2),
        df['Monday'][x],
        df['Tuesday'][x],
        df['Wednesday'][x],
        df['Thursday'][x],
        df['Friday'][x],
        df['Saturday'][x],
        df['Sunday'][x],
    ]
    for x in range(len(df['Overall Rank']))
    if df['App Name_x'][x] == 'Spending Tracker'
]

st_daily_downloads = [math.pow(math.exp(1), x) for x in reg1.predict(X_st)]
st_est_downloads = sum(st_daily_downloads)

print(
    'The real values for the downloads of Postmates, Over and Spending Tracker are 237,179, 35,653 and 29,641 \nrespect'
    'ively.\nOur estimation for the downloads of Postmates is ' + str(int(postmates_est_downloads)) + '.\n'
    'Our estimation for the downloads of Over is ' + str(int(over1_est_downloads)) + ' using the direct approach and '
    + str(int(over2_est_downloads)) + ' with the indirect one.\nOur estimation for the downloads of Spending Tracker'
    ' is ' + str(int(st_est_downloads)) + '.'
)
