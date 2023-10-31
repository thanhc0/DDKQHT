import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st
import joblib
import re


def get_year(student_id):
    year_str = ""
    for char in student_id:
        if char.isdigit():
            year_str += char
            if len(year_str) == 2:
                break
    return int(year_str)


@st.cache_data()
def process_data(raw_data):

    raw_data = raw_data[
        ~raw_data["TenMH"].str.contains("IE|Intensive English|IE2|IE1|IE3|IE0")
    ]

    pivot_df = pd.pivot_table(
        raw_data, values="DiemHP", index="MaSV", columns="TenMH", aggfunc="first"
    )
    pivot_df = pivot_df.reset_index().rename_axis(None, axis=1)
    pivot_df.columns.name = None
    pivot_df = pivot_df.dropna(thresh=50, axis=1)
    pivot_df = pivot_df.rename(columns=lambda x: x.strip())

    df = pd.merge(pivot_df, raw_data[["MaSV"]], on="MaSV")
    df.drop_duplicates(subset="MaSV", keep="last", inplace=True)
    dfid = df["MaSV"]
    df.drop(["MaSV"], axis=1, inplace=True)
    df.replace(["WH", "VT", "I"], np.nan, inplace=True)
    df.iloc[:, :-1] = df.iloc[:, :-1].apply(pd.to_numeric)
    df = pd.merge(dfid, df, left_index=True, right_index=True)
    df["MaSV_school"] = df["MaSV"].str.slice(2, 4)
    df["Major"] = df["MaSV"].str.slice(0, 2)
    df["Year"] = 2000 + df["MaSV"].apply(get_year)
    df["Year"] = df["Year"].astype(str)
    df = pd.merge(df, raw_data[["MaSV", "DTBTK"]].drop_duplicates(), on="MaSV")
    df = df.drop(columns="MaSV")

    return df


def process_data_per(raw_data):

    raw_data = raw_data[
        ~raw_data["TenMH"].str.contains("IE|Intensive English|IE2|IE1|IE3|IE0")
    ]
    pivot_df = pd.pivot_table(
        raw_data, values="DiemHP", index="MaSV", columns="TenMH", aggfunc="first"
    )
    pivot_df = pivot_df.reset_index().rename_axis(None, axis=1)
    pivot_df.columns.name = None
    pivot_df = pivot_df.dropna(thresh=50, axis=1)
    pivot_df = pivot_df.rename(columns=lambda x: x.strip())

    pivot_df.replace(["WH", "VT", "I"], np.nan, inplace=True)
    pivot_df.iloc[:, 1:] = pivot_df.iloc[:, 1:].apply(pd.to_numeric)

    return pivot_df


def process_predict_data(raw_data):
    dtk = raw_data[["MaSV", "DTBTKH4"]].copy()
    dtk.drop_duplicates(subset="MaSV", keep="last", inplace=True)

    count_duplicates = (
        raw_data.groupby(["MaSV", "MaMH"]).size().reset_index(name="Times")
    )
    courses = raw_data[
        raw_data["MaMH"].str.startswith(
            ("IT", "BA", "BM", "BT", "MA", "CE", "EE", "EL", "ENEE", "IS", "MAFE", "PH")
        )
    ]

    courses_list = courses["MaMH"].unique().tolist()

    count_duplicates["fail_courses_list"] = (
        (count_duplicates["MaMH"].isin(courses_list)) & (count_duplicates["Times"] >= 2)
    ).astype(int)

    count_duplicates["fail_not_courses_list"] = (
        (~count_duplicates["MaMH"].isin(courses_list))
        & (count_duplicates["Times"] >= 2)
    ).astype(int)

    count_duplicates["pass_courses"] = (
        (~count_duplicates["MaMH"].isin(courses_list))
        & (count_duplicates["Times"] == 1)
    ).astype(int)

    fail = (
        count_duplicates.groupby("MaSV")[["fail_courses_list", "fail_not_courses_list"]]
        .sum()
        .reset_index()
    )

    fail.columns = ["MaSV", "fail_courses_list_count", "fail_not_courses_list_count"]

    df = pd.merge(dtk, fail, on="MaSV")
    df = df.rename(columns={"DTBTKH4": "GPA"})

    data = raw_data[["MaSV", "NHHK", "SoTCDat"]]
    data = (
        data.groupby(["MaSV"])["SoTCDat"].mean().reset_index(name="Mean_Cre").round(2)
    )

    df = pd.merge(df, data, on="MaSV")
    df1 = raw_data[["MaSV", "MaMH", "NHHK"]]
    courses_list = raw_data[
        (raw_data["MaMH"].str.startswith("EN"))
        & ~(raw_data["MaMH"].str.contains("EN007|EN008|EN011|EN012"))
    ].MaMH.tolist()
    filtered_df = df1[df1["MaMH"].isin(courses_list)]
    nhhk_counts = (
        filtered_df.groupby("MaSV")["NHHK"].nunique().reset_index(name="EPeriod")
    )
    df = pd.merge(df, nhhk_counts, on="MaSV", how="left").fillna(0)
    df = df[
        [
            "MaSV",
            "GPA",
            "Mean_Cre",
            "fail_courses_list_count",
            "fail_not_courses_list_count",
            "EPeriod",
        ]
    ]
    return df


def predict_late_student(test_df):

    model = joblib.load("model/Time/Late.joblib")
    model1 = joblib.load("model/Time/Sem.joblib")

    test_dfed = process_predict_data(test_df)

    std_id = test_dfed.iloc[:, 0]

    test_dfed = test_dfed.drop(test_dfed.columns[0], axis=1)

    prediction = model.predict(test_dfed)

    prediction1 = model1.predict(test_dfed)

    test_dfed["Semeters"] = prediction1
    test_dfed["Progress"] = ["late" if p == 1 else "not late" for p in prediction]

    test_dfed.insert(0, "MaSV", std_id)

    for index, row in test_dfed.iterrows():
        if row["Semeters"] <= 9 and row["Progress"] == "late":
            test_dfed.loc[index, "Semeters"] = row["Semeters"] / 2
            test_dfed.loc[index, "Progress"] = "may late"
        else:
            test_dfed.loc[index, "Semeters"] = row["Semeters"] / 2

    return test_dfed


def get_major(raw_data):
    major_mapping = {
        "BA": "BA",
        "BE": "BM",
        "BT": "BT",
        "CE": "CE",
        "EE": "EE",
        "EN": "EL",
        "EV": "ENEE",
        "IE": "IS",
        "IT": "IT",
        "MA": "MAFE",
        "SE": "PH",
    }
    for major, ma_mh in major_mapping.items():
        if raw_data["MaSV"].str[:2].str.contains(major).any():
            return major, ma_mh
    return None, None


def create_pivot_table(raw_data):
    pivot_df = pd.pivot_table(
        raw_data, values="DiemHP", index="MaSV", columns="MaMH", aggfunc="first"
    )
    pivot_df = pivot_df.reset_index().rename_axis(None, axis=1)
    pivot_df.columns.name = None
    return pivot_df


def drop_nan_columns(pivot_df):
    pivot_df = pivot_df.rename(columns=lambda x: x.strip())
    pivot_df.replace(["WH", "VT", "I", "P", "F"], np.nan, inplace=True)
    pivot_df.iloc[:, 1:] = pivot_df.iloc[:, 1:].apply(pd.to_numeric)
    return pivot_df


def merge_with_xeploainh(pivot_df, raw_data):
    df = pd.merge(pivot_df, raw_data[["MaSV", "DTBTK"]], on="MaSV")
    df.drop_duplicates(subset="MaSV", keep="last", inplace=True)
    return df


def fill_missing_values(df):
    col = df.drop(["MaSV", "DTBTK"], axis=1)
    columns_data = get_column_data(df)
    dup = pd.DataFrame(columns=columns_data)
    df = pd.merge(dup, df, on=col.columns.tolist(), how="outer")
    for col in df.columns:
        if df[col].isnull().values.any():
            df[col].fillna(value=df["DTBTK"], inplace=True)
    return df


def get_column_data(df):
    major = df["MaSV"].str[:2].unique()[0]
    column_file = f"Columns/column_{major}.txt"
    columns_data = []
    with open(column_file, "r") as f:
        for line in f:
            columns_data.append(str(line.strip()))
    return columns_data


def prepare_data(df):
    std_id = df["MaSV"].copy()
    df = df.drop(["MaSV", "DTBTK"], axis=1)
    df.sort_index(axis=1, inplace=True)
    return df


def predict_rank(raw_data):
    major, ma_mh = get_major(raw_data)
    if major:
        raw_data["MaMH"] = raw_data["MaMH"].str[:-2]
        raw_data = raw_data[raw_data["MaMH"].str.startswith(ma_mh)]

        pivot_df = create_pivot_table(raw_data)
        pivot_df = drop_nan_columns(pivot_df)

        df = merge_with_xeploainh(pivot_df, raw_data)
        df = fill_missing_values(df)

        std_id = df["MaSV"].copy()
        df = prepare_data(df)

        model = joblib.load(f"model/{major}_rank.joblib")
        prediction = model.predict(df)

        new_columns = pd.concat(
            [pd.Series(std_id, name="MaSV"), pd.Series(prediction, name="Pred Rank")],
            axis=1,
        )
        df = pd.concat([new_columns, df], axis=1)
        newframe = df.copy()

        df = newframe[["MaSV", "Pred Rank"]]
        return df
    else:
        return None


def predict_one_student(raw_data, student_id):

    student = process_data_per(raw_data)
    filtered_df = student[student["MaSV"] == student_id]
    if len(filtered_df) > 0:
        selected_row = filtered_df.iloc[0, 1:].dropna()
        values = selected_row.values.tolist()
        course_data_filtered = [x for x in selected_row if not np.isnan(x)]
        counts, bins = np.histogram(course_data_filtered, bins=np.arange(0, 110, 10))
        grade_bins = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]
        total_count = len(selected_row)
        frequencies_percentage = (counts / total_count) * 100

        fig1 = go.Figure()

        fig1.add_trace(
            go.Scatter(
                x=bins[:-1], y=frequencies_percentage, mode="lines", name="Frequency"
            )
        )

        fig1.update_layout(
            title="Frequency Range for",
            xaxis_title="Score",
            yaxis_title="Percentage",
            height=400,
            width=400,
        )

        data = raw_data[["MaSV", "NHHK", "TenMH", "DiemHP"]]
        data["TenMH"] = data["TenMH"].str.lstrip()
        data["NHHK"] = data["NHHK"].apply(lambda x: str(x)[:4] + " S " + str(x)[4:])
        rows_to_drop = []
        with open("rows_to_drop.txt", "r") as f:
            for line in f:
                rows_to_drop.append(str(line.strip()))
        data = data[~data["TenMH"].isin(rows_to_drop)]
        student_data = data[data["MaSV"] == student_id][["NHHK", "TenMH", "DiemHP"]]
        student_data["DiemHP"] = pd.to_numeric(student_data["DiemHP"], errors="coerce")

        fig2 = px.bar(
            student_data,
            x="TenMH",
            y="DiemHP",
            color="NHHK",
            title="Student Score vs. Course",
        )
        fig2.update_layout(
            title="Student Score vs. Course",
            xaxis_title=None,
            yaxis_title="Score",
        )
        fig2.add_shape(
            type="line",
            x0=0,
            y0=50,
            x1=len(student_data["TenMH"]) - 1,
            y1=50,
            line=dict(color="red", width=3),
        )

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.write("No data found for student {}".format(student_id))


def show_boxplot1(
    new1_df, new1_dfa, major, school, year, additional_selection="", year_a=""
):
    if additional_selection != " ":
        show_boxplot = st.checkbox(
            "Show Boxplot for student's performance", key="checkbox2"
        )

        if show_boxplot:
            fig = px.box(new1_df)
            fig1 = px.box(new1_dfa)
            fig.update_layout(
                title="Boxplot of " + major + school + " student in " + year
            )
            fig1.update_layout(
                title="Boxplot of "
                + major
                + additional_selection
                + " student in "
                + year_a
            )
            col1, col2 = st.columns(2)

            with col1:
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.plotly_chart(fig1, use_container_width=True)

    elif additional_selection == " " and year_a != " ":
        show_boxplot = st.checkbox(
            "Show Boxplot for student's performance", key="checkbox2"
        )

        if show_boxplot:
            fig = px.box(new1_df)
            fig1 = px.box(new1_dfa)
            fig.update_layout(
                title="Boxplot of " + major + school + " student in " + year
            )
            fig1.update_layout(
                title="Boxplot of " + major + school + " student in " + year_a
            )
            col1, col2 = st.columns(2)

            with col1:
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.plotly_chart(fig1, use_container_width=True)

    elif additional_selection == " ":
        show_boxplot = st.checkbox(
            "Show Boxplot for student's performance", key="checkbox2"
        )

        if show_boxplot:
            fig = px.box(new1_df)
            fig.update_layout(title="Boxplot of " + major + " student in " + year)
            st.plotly_chart(fig, use_container_width=True)
