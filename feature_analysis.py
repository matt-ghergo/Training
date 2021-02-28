import pandas as pd
import numpy as np
from bsp_query_builder import Query
from bsp_query_builder.dialects.big_query.common import case_cond, Case
from bsp_bq.pandas import read_gbq
import plotly.express as px


def get_features_info(features_list: list) -> pd.DataFrame:
    """
    Helper function that returns an auxiliary dataframe with the relevant information about the selected features needed
    for the analysis
    :param features_list: this is the list of features
    :return: features properties
    """
    features_info = pd.DataFrame(
        np.array(
            [
                ["type", False, None, None, None, True, False],
                ["frames", False, None, None, None, False, False],
                ["aspectratio", True, 1, 0.5, 2, False, False],
                ["outermargin", True, 0.2, 0, 1, False, False],
                ["innermargin", True, 0.1, 0, 1, False, False],
                ["cornerradius", True, 0, 0, 1, False, False],
                ["shadow", True, 0, 0, 1, False, False],
                ["background", True, "backgroundcolor = #FFFFFFFF", "NaN", "NaN", False, True],
                ["stickers", True, "", None, None, False, False],
                ["texts", True, "", None, None, False, False],
            ]
        ),
        columns=["feature", "has_default", "default", "min", "max", "is_type", "is_background"]
    )
    features_info = features_info.set_index("feature", drop=False)
    features_info = features_info[features_info["feature"].apply(lambda x: True if x in features_list else False)]
    return features_info


def enjoyment(start_date: str, end_date: str, actions_end_date: str) -> Query:
    """
    Helper function that defines a query that returns data about the number of collages and the number of sessions each
    users did on Pic Jointer in a given period
    :param start_date: this is the date after which the users we consider have installed the app
    :param end_date: this is the date before which the users we consider have installed the app
    :param actions_end_date: this is the date until which we have observed the users
    :return: query for information about the enjoyment
    """
    return (
        Query("max_session")
            .from_("lumen2-42.Layer1_Apps.com_lileping_photoframepro_Timeline")
            .select(
            "uid",
            ("COUNT(uid)", "no_collage"),
            ("MAX(session_progressive_number)", "max_session")
        )
            .where(
            "type = 'photo_shared_with_full_info'",
            "is_free = FALSE",
            "installed_before_pico = FALSE",
            f"DATE(install_timestamp) BETWEEN '{start_date}' AND '{end_date}'",
            f"DATE(action_timestamp) BETWEEN '{start_date}' AND '{actions_end_date}'",
            f"days_from_install <= DATE_DIFF('{end_date}', '{start_date}', day)"
        )
            .group_by("uid")
    )


def retention(start_date: str, end_date: str, actions_end_date: str) -> Query:
    """
    Helper function that defines a query that returns data about the number of renewal each users did on Pic Jointer in a
    given period
    :param start_date: this is the date after which the users we consider have installed the app
    :param end_date: this is the date before which the users we consider have installed the app
    :param actions_end_date: this is the date until which we have observed the users
    :return: query for information about the retention
    """
    return (
        Query("max_renewal")
            .from_("lumen2-42.Layer1_Apps.com_lileping_photoframepro_Purchases")
            .select(
            "uid",
            ("MAX(renewal_progressive_number)", "max_renewal")
        )
            .where(
            "is_free = FALSE",
            "installed_before_pico = FALSE",
            f"DATE(install_timestamp) BETWEEN '{start_date}' AND '{end_date}'",
            f"DATE(purchase_timestamp) BETWEEN '{start_date}' AND '{actions_end_date}'",
            f"days_from_install <= DATE_DIFF('{end_date}', '{start_date}', day)"
        )
            .group_by("uid")
    )


def main_query(start_date: str, end_date: str, actions_end_date: str) -> Query:
    """
    Helper function that defines our main query, the one which returns all the relevant data about the collages made by
    all users in the observation period
    :param start_date: this is the date after which the users we consider have installed the app
    :param end_date: this is the date before which the users we consider have installed the app
    :param actions_end_date: this is the date until which we have observed the users
    """
    enj = enjoyment(start_date, end_date, actions_end_date)
    ret = retention(start_date, end_date, actions_end_date)
    return (
        Query()
            .from_("lumen2-42.Layer1_Apps.com_lileping_photoframepro_Timeline", "main")
            .select(
            "main.uid",
            "main.install_timestamp",
            "idfv",
            "days_from_install",
            "action_timestamp",
            "props__collage_type",
            "props__collage_frames",
            "props__collage_aspectratio",
            "props__collage_outermargin",
            "props__collage_innermargin",
            "props__collage_cornerradius",
            "props__collage_shadow",
            "props__collage_backgroundcolor",
            "props__collage_backgroundgradient",
            "props__collage_backgroundpattern",
            "props__collage_stickers",
            "props__collage_texts",
            "enj.no_collage",
            "enj.max_session",
            "ret.max_renewal",
            "PERCENT_RANK() OVER(ORDER BY enj.no_collage) AS percentile_collage",
            "PERCENT_RANK() OVER(ORDER BY enj.max_session) AS percentile_session",
            "PERCENT_RANK() OVER(ORDER BY ret.max_renewal) AS percentile_renewal",
            "session_progressive_number",
            "event_progressive_number"
        )
            .where(
            "max_renewal IS NOT NULL",
            "type = 'photo_shared_with_full_info'",
            "is_free = FALSE",
            "installed_before_pico = FALSE",
            f"DATE(install_timestamp) BETWEEN '{start_date}' AND '{end_date}'",
            f"DATE(action_timestamp) BETWEEN '{start_date}' AND '{actions_end_date}'",
            f"days_from_install <= DATE_DIFF('{end_date}', '{start_date}', day)",
            "props__collage_type <> ''"
        )
            .join(enj, "enj", join_type="LEFT", on=("enj.uid = main.uid"))
            .join(ret, "ret", join_type="LEFT", on=("ret.uid = main.uid"))
            .order_by(
            "main.uid",
            "session_progressive_number",
            "event_progressive_number"
        )
    )


def total_users_query(start_date: str, end_date: str, actions_end_date: str, percentile_session: float = 0,
                      percentile_renewal: float = 0, percentile_collage: float = 0) -> Query:
    """
    Helper function that defines a query that returns the total number of users in the observation period
    :param start_date: this is the date after which the users we consider have installed the app
    :param end_date: this is the date before which the users we consider have installed the app
    :param actions_end_date: this is the date until which we have observed the users
    :param percentile_session: this is the top percentile for number of sessions we want to consider
    :param percentile_renewal: this is the top percentile for number of renewals we want to consider
    :param percentile_collage: this is the top percentile for number of collages we want to consider
    :return: query for the total number of users
    """
    mq = main_query(start_date, end_date, actions_end_date)
    total_users = (
        Query()
            .select("COUNT(DISTINCT uid)")
            .from_(mq)
            .where(
            f"percentile_session >= {percentile_session}",
            f"percentile_renewal >= {percentile_renewal}",
            f"percentile_collage >= {percentile_collage}"
        )
    )
    return total_users


def total_collage_query(start_date: str, end_date: str, actions_end_date: str, percentile_session: float = 0,
                        percentile_renewal: float = 0, percentile_collage: float = 0) -> Query:
    """
    Helper function that defines a query that returns the total number of collages in the observation period
    :param start_date: this is the date after which the users we consider have installed the app
    :param end_date: this is the date before which the users we consider have installed the app
    :param actions_end_date: this is the date until which we have observed the users
    :param percentile_session: this is the top percentile for number of sessions we want to consider
    :param percentile_renewal: this is the top percentile for number of renewals we want to consider
    :param percentile_collage: this is the top percentile for number of collages we want to consider
    :return: query for the total number of collages
    """
    mq = main_query(start_date, end_date, actions_end_date)
    total_collage = (
        Query()
            .select(f"COUNT(props__collage_type)")
            .from_(mq)
            .where(
            f"percentile_session >= {percentile_session}",
            f"percentile_renewal >= {percentile_renewal}",
            f"percentile_collage >= {percentile_collage}"
        )
    )
    return total_collage


def no_default_query(feature: str, start_date: str, end_date: str, actions_end_date: str, percentile_session: float = 0,
                     percentile_renewal: float = 0, percentile_collage: float = 0) -> list:
    """
    Define a query that returns the distribution across collages (absolute and by users) of the values for a certain
    feature which has no default value
    :param feature: this is the feature to consider
    :param start_date: this is the date after which the users we consider have installed the app
    :param end_date: this is the date before which the users we consider have installed the app
    :param actions_end_date: this is the date until which we have observed the users
    :param percentile_session: this is the top percentile for number of sessions we want to consider
    :param percentile_renewal: this is the top percentile for number of renewals we want to consider
    :param percentile_collage: this is the top percentile for number of collages we want to consider
    :return: a list with the query
    """
    mq = main_query(start_date, end_date, actions_end_date)
    total_users = total_users_query(start_date, end_date, actions_end_date, percentile_session, percentile_renewal,
                                    percentile_collage)
    total_collage = total_collage_query(start_date, end_date, actions_end_date, percentile_session,
                                        percentile_renewal,
                                        percentile_collage)
    return [
        (
            Query()
                .from_(mq)
                .select(
                f"props__collage_{feature}",
                f"COUNT(props__collage_{feature}) / ({total_collage}) AS frequency_collage",
                f"COUNT(DISTINCT uid) / ({total_users}) AS frequency_users"
            )
                .where(
                f"percentile_session >= {percentile_session}",
                f"percentile_renewal >= {percentile_renewal}",
                f"percentile_collage >= {percentile_collage}"
            )
                .group_by(
                f"props__collage_{feature}"
            )
                .order_by(
                "frequency_collage"
            )
        )
    ]


def default_zero_query(feature: str, start_date: str, end_date: str, actions_end_date: str,
                       percentile_session: float = 0,
                       percentile_renewal: float = 0, percentile_collage: float = 0) -> list:
    """
    Define a query that returns the distribution across collages (absolute and by users) of the values for a certain
    feature which has 0 as the default value
    :param feature: this is the feature to consider
    :param start_date: this is the date after which the users we consider have installed the app
    :param end_date: this is the date before which the users we consider have installed the app
    :param actions_end_date: this is the date until which we have observed the users
    :param percentile_session: this is the top percentile for number of sessions we want to consider
    :param percentile_renewal: this is the top percentile for number of renewals we want to consider
    :param percentile_collage: this is the top percentile for number of collages we want to consider
    :return: a list with the query
    """
    mq = main_query(start_date, end_date, actions_end_date)
    total_users = total_users_query(start_date, end_date, actions_end_date, percentile_session, percentile_renewal,
                                    percentile_collage)
    total_collage = total_collage_query(start_date, end_date, actions_end_date, percentile_session,
                                        percentile_renewal,
                                        percentile_collage)
    int_query = (
        Query()
            .from_(mq)
            .select("uid",
                    f"props__collage_{feature}",
                    (
                        case_cond(
                            Case(when=f"props__collage_{feature} = 0", then="'Default (0)'"),
                            else_clause="'Deviates'",
                        ), f"{feature}"
                    ),
                    ("COUNT(uid) OVER()", "total")
                    )
            .where(
            f"percentile_session >= {percentile_session}",
            f"percentile_renewal >= {percentile_renewal}",
            f"percentile_collage >= {percentile_collage}"
        )
    )
    return [
        (
            Query()
                .from_(int_query)
                .select(
                f"{feature}",
                f"COUNT({feature}) / ({total_collage}) AS frequency_collage",
                f"COUNT(DISTINCT uid) / ({total_users}) AS frequency_users"
            )
                .group_by(
                f"{feature}"
            )
                .order_by(
                "frequency_collage"
            )
        )
    ]


def default_null_query(feature: str, start_date: str, end_date: str, actions_end_date: str,
                       percentile_session: float = 0,
                       percentile_renewal: float = 0, percentile_collage: float = 0) -> list:
    """
    Define a query that returns the distribution across collages (absolute and by users) of the values for a certain
    feature which has '' as the default value
    :param feature: this is the feature to consider
    :param start_date: this is the date after which the users we consider have installed the app
    :param end_date: this is the date before which the users we consider have installed the app
    :param actions_end_date: this is the date until which we have observed the users
    :param percentile_session: this is the top percentile for number of sessions we want to consider
    :param percentile_renewal: this is the top percentile for number of renewals we want to consider
    :param percentile_collage: this is the top percentile for number of collages we want to consider
    :return: a list with the query
    """
    mq = main_query(start_date, end_date, actions_end_date)
    total_users = total_users_query(start_date, end_date, actions_end_date, percentile_session, percentile_renewal,
                                    percentile_collage)
    total_collage = total_collage_query(start_date, end_date, actions_end_date, percentile_session,
                                        percentile_renewal,
                                        percentile_collage)
    int_query = (
        Query()
            .from_(mq)
            .select("uid",
                    f"props__collage_{feature}",
                    (
                        case_cond(
                            Case(when=f"props__collage_{feature} = ''", then=f"'No {feature}'"),
                            else_clause=f"'{feature}'",
                        ), f"{feature}"
                    ),
                    ("COUNT(uid) OVER()", "total")
                    )
            .where(
            f"percentile_session >= {percentile_session}",
            f"percentile_renewal >= {percentile_renewal}",
            f"percentile_collage >= {percentile_collage}"
        )
    )
    return [
        (
            Query()
                .from_(int_query)
                .select(
                f"{feature}",
                f"COUNT({feature}) / ({total_collage}) AS frequency_collage",
                f"COUNT(DISTINCT uid) / ({total_users}) AS frequency_users"
            )
                .group_by(
                f"{feature}"
            )
                .order_by(
                "frequency_collage"
            )
        )
    ]


def default_query(feature: str, default: float, min: float, max: float, start_date: str, end_date: str,
                  actions_end_date: str, percentile_session: float = 0,
                  percentile_renewal: float = 0, percentile_collage: float = 0) -> list:
    """
    Define a query that returns the distribution across collages (absolute and by users) of the values for a certain
    feature which has a default value different from 0 or ''
    :param feature: this is the feature to consider
    :param default: this is the default value
    :param min: this is the minimum admissible value
    :param max: this is the maximum admissible value
    :param start_date: this is the date after which the users we consider have installed the app
    :param end_date: this is the date before which the users we consider have installed the app
    :param actions_end_date: this is the date until which we have observed the users
    :param percentile_session: this is the top percentile for number of sessions we want to consider
    :param percentile_renewal: this is the top percentile for number of renewals we want to consider
    :param percentile_collage: this is the top percentile for number of collages we want to consider
    :return: a list with the query
    """
    mq = main_query(start_date, end_date, actions_end_date)
    total_users = total_users_query(start_date, end_date, actions_end_date, percentile_session, percentile_renewal,
                                    percentile_collage)
    total_collage = total_collage_query(start_date, end_date, actions_end_date, percentile_session,
                                        percentile_renewal,
                                        percentile_collage)
    int_query = (
        Query()
            .from_(mq)
            .select("uid",
                    f"props__collage_{feature}",
                    (
                        case_cond(
                            Case(when=f"props__collage_{feature} = {default}", then=f"'Default ({default})'"),
                            Case(when=f"props__collage_{feature} = {max}", then=f"'Max ({max})'"),
                            Case(when=f"props__collage_{feature} = {min}", then=f"'Min ({min})'"),
                            else_clause="'Other'",
                        ), f"{feature}"
                    ),
                    ("COUNT(uid) OVER()", "total")
                    )
            .where(
            f"percentile_session >= {percentile_session}",
            f"percentile_renewal >= {percentile_renewal}",
            f"percentile_collage >= {percentile_collage}"
        )
    )
    return [
        (
            Query()
                .from_(int_query)
                .select(
                f"{feature}",
                f"COUNT({feature}) / ({total_collage}) AS frequency_collage",
                f"COUNT(DISTINCT uid) / ({total_users}) AS frequency_users"
            )
                .group_by(
                f"{feature}"
            )
                .order_by(
                "frequency_collage"
            )
        )
    ]


def type_query(start_date: str, end_date: str, actions_end_date: str, percentile_session: float = 0,
               percentile_renewal: float = 0, percentile_collage: float = 0) -> list:
    """
    Define some queries that returns the distribution across collages (absolute and by users) of different type of
    values for the type feature
    :param start_date: this is the date after which the users we consider have installed the app
    :param end_date: this is the date before which the users we consider have installed the app
    :param actions_end_date: this is the date until which we have observed the users
    :param percentile_session: this is the top percentile for number of sessions we want to consider
    :param percentile_renewal: this is the top percentile for number of renewals we want to consider
    :param percentile_collage: this is the top percentile for number of collages we want to consider
    :return: a list with the queries
    """
    mq = main_query(start_date, end_date, actions_end_date)

    total_users = total_users_query(start_date, end_date, actions_end_date, percentile_session, percentile_renewal,
                                    percentile_collage)

    total_collage = total_collage_query(start_date, end_date, actions_end_date, percentile_session,
                                        percentile_renewal,
                                        percentile_collage)

    int_query_1 = (
        Query()
            .from_(mq)
            .select(
            "uid",
            "props__collage_type",
            (
                case_cond(
                    Case(when="props__collage_type LIKE 'Class%'", then="'Classic'"),
                    Case(when="props__collage_type LIKE 'Styl%'", then="'Stylish'"),
                    else_clause="'Other'",
                ),
                "collage_type"
            ),
            ("COUNT(uid) OVER()", "total"),
            ("COUNT(DISTINCT uid) OVER()", "total_users"),
        )
            .where(
            f"percentile_session >= {percentile_session}",
            f"percentile_renewal >= {percentile_renewal}",
            f"percentile_collage >= {percentile_collage}"
        )
    )

    int_query_2 = (
        Query()
            .from_(mq)
            .select(
            "props__collage_type",
            f"COUNT(DISTINCT uid) / ({total_users}) AS frequency_collage_byusers",
            f"COUNT(props__collage_type) / ({total_collage}) AS frequency_collage",
            (f"RANK() OVER(ORDER BY COUNT(DISTINCT uid) / ({total_users}) DESC)", "rank_users"),
            (f"RANK() OVER(ORDER BY COUNT(props__collage_type) / ({total_collage}) DESC)", "rank_collage")
        )
            .where(
            f"percentile_session >= {percentile_session}",
            f"percentile_renewal >= {percentile_renewal}",
            f"percentile_collage >= {percentile_collage}"
        )
            .group_by(
            f"props__collage_type"
        )
            .order_by(
            "rank_users"
        )
    )

    type_query_1 = (
        Query()
            .from_(int_query_1)
            .select(
            "collage_type",
            "COUNT(collage_type) / SUM(COUNT(collage_type)) OVER()",
        )
            .group_by("collage_type")
    )

    type_query_2 = (
        Query()
            .from_(int_query_2)
            .select(
            (
                case_cond(
                    Case(when="rank_users < 15", then="props__collage_type"),
                    Case(when="rank_users >= 15", then="'Other'"),
                ),
                "props__collage_type_users_top10"
            ),
            "SUM(frequency_collage_byusers)"
        )
            .group_by("props__collage_type_users_top10")
    )

    type_query_3 = (
        Query()
            .from_(int_query_2)
            .select(
            (
                case_cond(
                    Case(when="rank_collage < 10", then="props__collage_type"),
                    Case(when="rank_collage >= 10", then="'Other'"),
                ),
                "props__collage_type_top10"
            ),
            "SUM(frequency_collage)"
        )
            .group_by("props__collage_type_top10")
    )

    return [
        type_query_1,
        type_query_2,
        type_query_3
    ]


def background_query(start_date: str, end_date: str, actions_end_date: str, percentile_session: float = 0,
                     percentile_renewal: float = 0, percentile_collage: float = 0) -> list:
    """
    Define some queries that returns the distribution across collages (absolute and by users) of different type of
    values for the background feature
    :param start_date: this is the date after which the users we consider have installed the app
    :param end_date: this is the date before which the users we consider have installed the app
    :param actions_end_date: this is the date until which we have observed the users
    :param percentile_session: this is the top percentile for number of sessions we want to consider
    :param percentile_renewal: this is the top percentile for number of renewals we want to consider
    :param percentile_collage: this is the top percentile for number of collages we want to consider
    :return: a list with the query
    """
    mq = main_query(start_date, end_date, actions_end_date)
    total_users = total_users_query(start_date, end_date, actions_end_date, percentile_session, percentile_renewal,
                                    percentile_collage)
    total_collage = total_collage_query(start_date, end_date, actions_end_date, percentile_session,
                                        percentile_renewal,
                                        percentile_collage)
    int_query = (
        Query()
            .from_(mq)
            .select("uid",
                    (
                        case_cond(
                            Case(
                                when="(props__collage_backgroundcolor = '#FFFFFFFF' AND props__collage_backgroundgradient = '')",
                                then="'color (def)'"),
                            Case(
                                when="(props__collage_backgroundcolor <> '' AND props__collage_backgroundgradient = '')",
                                then="'color'"),
                            Case(
                                when="(props__collage_backgroundcolor = '' AND props__collage_backgroundgradient <> '')",
                                then="'gradient'"),
                            Case(
                                when="(props__collage_backgroundcolor = '' AND props__collage_backgroundgradient = '')",
                                then="'pattern'"),
                            else_clause="'unknown'",
                        ), "background"
                    ),
                    ("COUNT(uid) OVER()", "total")
                    )
            .where(
            f"percentile_session >= {percentile_session}",
            f"percentile_renewal >= {percentile_renewal}",
            f"percentile_collage >= {percentile_collage}"
        )
    )
    return [
        (
            Query()
                .from_(int_query)
                .select(
                f"background",
                f"COUNT(background) / ({total_collage}) AS frequency_collage",
                f"COUNT(DISTINCT uid) / ({total_users}) AS frequency_users"
            )
                .group_by(
                "background"
            )
                .order_by(
                "frequency_collage"
            )
        )
    ]


def feature_query(features_info: pd.DataFrame, feature: str, start_date: str, end_date: str, actions_end_date: str,
                  percentile_session: float = 0, percentile_renewal: float = 0, percentile_collage: float = 0) -> list:
    """
    Auxiliary function that allocates a feature to the right query/ies based on its properties
    :param features_info: this is the dataframe with the information about the features
    :param feature: this is the feature to consider
    :param start_date: this is the date after which the users we consider have installed the app
    :param end_date: this is the date before which the users we consider have installed the app
    :param actions_end_date: this is the date until which we have observed the users
    :param percentile_session: this is the top percentile for number of sessions we want to consider
    :param percentile_renewal: this is the top percentile for number of renewals we want to consider
    :param percentile_collage: this is the top percentile for number of collages we want to consider
    :return: a list of queries
    """
    if features_info.loc[feature, 'is_type'] == False:
        if features_info.loc[feature, 'is_background'] == False:
            if features_info.loc[feature, 'has_default'] == False:
                return no_default_query(feature, start_date, end_date, actions_end_date, percentile_session,
                                        percentile_renewal, percentile_collage)
            elif features_info.loc[feature, 'has_default'] == True:
                if features_info.loc[feature, 'default'] == 0:
                    return default_zero_query(feature, start_date, end_date, actions_end_date, percentile_session,
                                              percentile_renewal, percentile_collage)
                elif features_info.loc[feature, 'default'] == "":
                    return default_null_query(feature, start_date, end_date, actions_end_date, percentile_session,
                                              percentile_renewal, percentile_collage)
                else:
                    return default_query(feature, features_info.loc[feature, 'default'],
                                         features_info.loc[feature, 'min'], features_info.loc[feature, 'max'],
                                         start_date, end_date, actions_end_date, percentile_session,
                                         percentile_renewal, percentile_collage)
        elif features_info.loc[feature, 'is_background'] == True:
            return background_query(start_date, end_date, actions_end_date, percentile_session,
                                    percentile_renewal, percentile_collage)
    elif features_info.loc[feature, 'is_type'] == True:
        return type_query(start_date, end_date, actions_end_date, percentile_session,
                          percentile_renewal, percentile_collage)


def feature_plot(df: pd.DataFrame, feature: str) -> None:
    """
    Plot the information we have about a certain feature
    :param df: where to get the data to plot
    :param feature: the feature we are considering in the data
    """
    fig1 = px.pie(names=df.iloc[:, 0], values=df.iloc[:, 1], title=f"{feature} distribution")
    fig1.update_traces(textinfo='percent+label')
    fig1.show()
    try:
        fig2 = px.bar(x=df.iloc[:, 0], y=df.iloc[:, 2], title=f"{feature} distribution by users")
        fig2.show()
    except:
        pass


def feature_analysis(features_list, start_date: str, end_date: str, actions_end_date: str,
                  percentile_session: float = 0, percentile_renewal: float = 0, percentile_collage: float = 0) -> None:
    """
    Auxiliary function that defines the queries for the features we want to analyze, runs it and download the data into
    a dataframe, and then plots it
    :param features_list:
    :param start_date: this is the date after which the users we consider have installed the app
    :param end_date: this is the date before which the users we consider have installed the app
    :param actions_end_date: this is the date until which we have observed the users
    :param percentile_session: this is the top percentile for number of sessions we want to consider
    :param percentile_renewal: this is the top percentile for number of renewals we want to consider
    :param percentile_collage: this is the top percentile for number of collages we want to consider
    """
    features_info = get_features_info(features_list)
    for feature in features_list:
        query_list = feature_query(features_info, feature, start_date, end_date, actions_end_date, percentile_session,
                                   percentile_renewal, percentile_collage)
        for query in query_list:
            df = read_gbq(query, project='lumen2-42')
            feature_plot(df, feature)


if __name__ == '__main__':
    features_list = [
        "type",
        "frames",
        "aspectratio",
        "outermargin",
        "innermargin",
        "cornerradius",
        "shadow",
        "background",
        "stickers",
        "texts"
    ]
    start_date: str = '2020-01-01'
    end_date: str = '2020-06-30'
    actions_end_date: str = '2020-12-31'
    percentile_session: float = 0
    percentile_renewal: float = 0
    percentile_collage: float = 0
    feature_analysis(features_list, start_date, end_date, actions_end_date, percentile_session, percentile_renewal,
                     percentile_collage)
