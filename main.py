import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
import plotly.graph_objs as go
from function import (
    process_data,
    predict_late_student,
    predict_rank,
    predict_one_student,
    show_boxplot1,
)
from datetime import datetime
from PIL import Image
import base64
import re
import sqlite3

df = pd.DataFrame()


def color_cell(val):
    if val == "not late":
        color = "green"
    elif val == "may late":
        color = "yellow"
    elif val == "late":
        color = "red"
    else:
        color = "black"

    return f"color: {color};"


def clear_resources():
    """Clears all resources from the st.session_state."""
    for key in list(st.session_state.keys()):
        if key.startswith("resource"):
            del st.session_state[key]


def get_year(student_id):
    year_str = ""
    for char in student_id:
        if char.isdigit():
            year_str += char
            if len(year_str) == 2:
                break
    return int(year_str)


def generate_comment(median):
    if median < 30:
        comment = f"The median score for {course} is quite low at {median}. Students may need to work harder to improve their performance."
    elif median < 50:
        comment = f"The median score for {course} is below average at {median}. Students should work on improving their understanding of the material."
    elif median < 80:
        comment = f"The median score for {course} is solid at {median}. Students are making good progress but could still work on improving their skills."
    else:
        comment = f"The median score for {course} is outstanding at {median}. Students are doing an excellent job in this course."
    return comment


favicon = "R.png"
hcm = "HCM.png"
intera = "Logo-iuoss-trans.png"
st.set_page_config(
    page_title="Student System",
    page_icon=favicon,
    layout="wide",
)
st.markdown(
    '<div style="text-align: center; margin-top: 50px; color: #808080;">'
    '© Copyright by Truong Quoc An'
    '</div>',
    unsafe_allow_html=True
)

currentYear = datetime.now().year
im1 = Image.open("R.png")
im2 = Image.open("HCM.png")
im3 = Image.open("Logo-iuoss-trans.png")


col1, col2, col3 = st.columns([1, 3, 1])


with col1:
    st.image(im1, width=150)


with col2:
    st.markdown(
        "<h1 style='text-align: center;'>Student Performance Evaluation</h1>",
        unsafe_allow_html=True,
    )


with col3:
    st.image(im2, width=250)


@st.cache_data()
def score_table():
    # Establish a connection to the database
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    # Fetch data from the tables
    cursor.execute('''SELECT Students.MaSV, Enrollment.MaMH, Courses.TenMH, Enrollment.NHHK, Enrollment.DiemHP, Students.DTBTK
                      FROM Students
                      INNER JOIN Enrollment ON Students.MaSV = Enrollment.MaSV
                      INNER JOIN Courses ON Enrollment.MaMH = Courses.MaMH''')
    data = cursor.fetchall()

    # Create a DataFrame
    df = pd.DataFrame(data, columns=['MaSV', 'MaMH', 'TenMH', 'NHHK', 'DiemHP', 'DTBTK'])
    df = df.drop_duplicates()

    # Close the database connection
    conn.close()

    return df

@st.cache_data()
def score_table_for_student():
    with sqlite3.connect("database.db") as conn:
        cursor = conn.cursor()

        cursor.execute('''
            SELECT DISTINCT Students.MaSV, Enrollment.MaMH, Courses.TenMH, Enrollment.NHHK, Enrollment.DiemHP, Students.DTBTK
            FROM Students
            INNER JOIN Enrollment ON Students.MaSV = Enrollment.MaSV
            INNER JOIN Courses ON Enrollment.MaMH = Courses.MaMH
        ''')
        results = cursor.fetchall()
        df = pd.DataFrame(results, columns=['MaSV', 'MaMH', 'TenMH', 'NHHK', 'DiemHP', 'DTBTK'])

        cursor.execute('''
            SELECT MaSV, NHHK, SoTCDat
            FROM Students
        ''')
        results = cursor.fetchall()
        df1 = pd.DataFrame(results, columns=['MaSV', 'NHHK', 'SoTCDat'])
        merged_df = pd.merge(df, df1, on=['MaSV', 'NHHK'])

    return merged_df




st.sidebar.image(im3)
st.sidebar.title("Student Performance Prediction System")
option = ["Prediction Performance","Dashboard",  "Grade Distribution Tables"]

tabs = st.sidebar.selectbox("Select an option", option)


def filter_dataframe(df, column, value):
    if value == "All":
        return df
    else:
        return df[df[column] == value]


if tabs == "Dashboard":
    clear_resources()
    raw_data = score_table()
    df = process_data(raw_data)
    additional_selection = " "
    unique_values_major = df["Major"].unique()
    unique_values_major = [
        "BA",
        "BE",
        "BT",
        "CE",
        "EE",
        "EN",
        "EV",
        "IE",
        "MA",
        "SE",
        "IT",
    ]
    unique_values_major = sorted(unique_values_major, key=lambda s: s)
    major = st.selectbox("Select a school:", unique_values_major)
    df = filter_dataframe(df, "Major", major)
    dfa = filter_dataframe(df, "Major", major)

    unique_values_school = df["MaSV_school"].unique()
    all_values_school = np.concatenate([["All"], unique_values_school])
    no_numbers = [x for x in all_values_school if not re.search(r"\d", str(x))]

    if len(no_numbers) == 2:
        school = no_numbers[1]
    else:
        col1, col2 = st.columns(2)

        with col1:
            school = st.selectbox("Select a major:", no_numbers)

        if school != "All":
            values = [x for x in no_numbers if x != "All" and x != school]
            values = np.concatenate([[" "], values])

            with col2:
                additional_selection = st.selectbox(
                    "Select another major for comparisons:", values
                )
                if additional_selection != " ":
                    dfa = filter_dataframe(dfa, "MaSV_school", additional_selection)

    df = filter_dataframe(df, "MaSV_school", school)

    unique_values_year = df["Year"].unique()
    all_values_year = np.concatenate([["All"], unique_values_year])

    col1, col2 = st.columns(2)

    with col1:
        year = st.selectbox("Select a year:", all_values_year)

    with col2:
        if year != "All" and additional_selection == " ":
            year_list = [x for x in all_values_year if x != "All" and x != year]
            year_list = np.concatenate([[" "], year_list])
            year_a = st.selectbox("Select another year for comparisons:", year_list)
        elif year == "All":
            year_a = " "
        elif year != "All" and additional_selection != " ":
            year_a = year
            if year_a != " ":
                dfa = filter_dataframe(dfa, "Year", year_a)
                dfa.dropna(axis=1, thresh=1, inplace=True)
            else:
                year_a = " "

    df = filter_dataframe(df, "Year", year)
    new1_df = df.DTBTK
    new1_dfa = dfa.DTBTK
    show_boxplot1(
        new1_df, new1_dfa, major, school, year, additional_selection="", year_a=""
    )

    df.dropna(axis=1, thresh=1, inplace=True)

    new_df = df.iloc[:, :-4].dropna(axis=1, thresh=10).apply(pd.to_numeric)
    new_dfa = dfa.iloc[:, :-4].dropna(axis=1, thresh=10).apply(pd.to_numeric)
    list1 = new_df.columns.tolist()
    list2 = new_dfa.columns.tolist()
    if (year != "All" and year_a != " ") or (
        school != "All" and additional_selection != " "
    ):
        dfac = new_dfa.columns[:-4].tolist()
        common_elements = np.intersect1d(list1, list2)

        merged_array = np.concatenate((list1, list2), axis=None)

        list3 = np.intersect1d(merged_array, common_elements)
        new_df = new_df[list3]
        new_dfa = new_dfa[list3]
    if additional_selection != " ":
        show_boxplot = st.checkbox("Show Boxplot for All Course", key="checkbox1")

        if show_boxplot:
            fig = px.box(new_df)
            fig1 = px.box(new_dfa)
            fig.update_layout(
                title="Boxplot of " + major + school + " student in " + year
            )
            fig1.update_layout(
                title="Boxplot of "
                + major
                + additional_selection
                + " student in "
                + year
            )

            st.plotly_chart(fig, use_container_width=True)
            st.plotly_chart(fig1, use_container_width=True)

    elif additional_selection == " " and year_a != " ":
        show_boxplot = st.checkbox("Show Boxplot for All Course", key="checkbox1")

        if show_boxplot:
            fig = px.box(new_df)
            fig1 = px.box(new_dfa)
            fig.update_layout(
                title="Boxplot of " + major + school + " student in " + year
            )
            fig1.update_layout(
                title="Boxplot of " + major + school + " student in " + year_a
            )

            st.plotly_chart(fig, use_container_width=True)
            st.plotly_chart(fig1, use_container_width=True)

    elif additional_selection == " ":
        show_boxplot = st.checkbox("Show Boxplot for All Course", key="checkbox1")

        if show_boxplot:
            fig = px.box(new_df)
            fig.update_layout(title="Boxplot of " + major + " student in " + year)

            st.plotly_chart(fig, use_container_width=True)

    options = df.columns[:-4]

    course_data_dict = {course: df[course].dropna() for course in options}
    valid_courses = [
        course for course, data in course_data_dict.items() if len(data) > 1
    ]

    if (year != "All" and year_a != " ") or (
        school != "All" and additional_selection != " "
    ):
        dfac = new_dfa.columns[:-4].tolist()
        common_elements = np.intersect1d(valid_courses, dfac)

        merged_array = np.concatenate((valid_courses, dfac), axis=None)

        valid_courses = np.intersect1d(merged_array, common_elements)

    if len(valid_courses) > 5:
        course = st.selectbox("Select a course:", valid_courses)
    elif len(valid_courses) == 1:
        course = valid_courses[0]
    else:
        st.write("No valid course data found!")
        st.stop()

    course_data = course_data_dict[course]

    if len(course_data) > 1:
        if school == "All":
            st.write("Course:", course, " of ", major, " student")
        else:
            st.write("Course:", course, " of ", major + school, " student")
        st.write(generate_comment(course_data.median()))
    else:
        st.write("No data available for the selected course.")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        counts, bins = np.histogram(course_data, bins=np.arange(0, 110, 10))
        total_count = len(course_data)
        frequencies_percentage = (counts / total_count) * 100
        grade_bins = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]

        df = pd.DataFrame(
            {"Grade": grade_bins, "Grading percentage": frequencies_percentage}
        )
        df["Grading percentage"] = df["Grading percentage"].map(
            lambda x: "{:.2f}".format(x)
        )

        st.table(df)

    with col2:

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=bins[:-1], y=frequencies_percentage, mode="lines", name="Frequency"
            )
        )

        fig.update_layout(
            title="Histogram of {}".format(course),
            xaxis_title="Score",
            yaxis_title="Percentage",
            height=400,
            width=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        fig = go.Figure()
        fig.add_trace(go.Box(y=course_data, name="Box plot"))
        fig.update_layout(
            title="Box plot of Scores for {}".format(course),
            yaxis_title="Score",
            height=400,
            width=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        raw_data1 = raw_data.copy()
        raw_data1["major"] = raw_data1["MaSV"].str.slice(0, 2)
        raw_data1.replace(["WH", "VT", "I"], np.nan, inplace=True)
        raw_data1 = raw_data1[~raw_data1["DiemHP"].isin(["P", "F", "PC"])]
        if major != "All":
            raw_data1 = raw_data1[raw_data1["major"] == major]

        raw_data1["MaSV_school"] = raw_data1["MaSV"].str.slice(2, 4)
        if school != "All":
            raw_data1 = raw_data1[raw_data1["MaSV_school"] == school]

        df1 = raw_data1[["TenMH", "NHHK", "DiemHP"]].copy()
        
        df1["DiemHP"] = df1["DiemHP"].replace('', pd.NA).dropna().astype(float)
        df1["NHHK"] = df1["NHHK"].apply(lambda x: str(x)[:4] + " S " + str(x)[4:])

        selected_TenMH = " " + course
        filtered_df1 = df1[df1["TenMH"] == selected_TenMH]

        mean_DiemHP = (
            filtered_df1.groupby("NHHK")["DiemHP"]
            .mean()
            .round(1)
            .reset_index(name="Mean")
        )

        if year != "All":
            st.write("")
        else:
            fig = px.line(
                mean_DiemHP,
                x="NHHK",
                y="Mean",
                title=f"Mean Course Score for{selected_TenMH} through Semeters",
            )
            fig.update_layout(xaxis_title="Semeters",height=400, width=400)
            st.plotly_chart(fig, use_container_width=True)

    if (year != "All" and year_a != " ") or (
        school != "All" and additional_selection != " "
    ):
        course_data_dict = {course: new_dfa[course]}
        course_data = course_data_dict[course]

        st.write(
            "Course:",
            course,
            " of ",
            major + additional_selection,
            " student in ",
            year_a,
        )
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            course_data_filtered = [x for x in course_data if not np.isnan(x)]
            counts, bins = np.histogram(
                course_data_filtered, bins=np.arange(0, 110, 10)
            )
            total_count = len(course_data_filtered)
            frequencies_percentage = (counts / total_count) * 100
            grade_bins = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]

            df1 = pd.DataFrame(
                {"Grade": grade_bins, "Grading percentage": frequencies_percentage}
            )
            df1["Grading percentage"] = df1["Grading percentage"].map(
                lambda x: "{:.2f}".format(x)
            )

            st.table(df1)

        with col2:

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=bins[:-1],
                    y=frequencies_percentage,
                    mode="lines",
                    name="Frequency",
                )
            )

            fig.update_layout(
                title="Histogram of {}".format(course),
                xaxis_title="Score",
                yaxis_title="Percentage",
                height=400,
                width=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            fig = go.Figure()
            fig.add_trace(go.Box(y=course_data, name="Box plot"))
            fig.update_layout(
                title="Box plot of Scores for {}".format(course),
                yaxis_title="Score",
                height=400,
                width=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            raw_data["major"] = raw_data["MaSV"].str.slice(0, 2)
            raw_data.replace(["WH", "VT", "I"], np.nan, inplace=True)
            raw_data = raw_data[~raw_data["DiemHP"].isin(["P", "F", "PC"])]
            if major != "All":
                raw_data = raw_data[raw_data["major"] == major]

            raw_data["MaSV_school"] = raw_data["MaSV"].str.slice(2, 4)
            raw_data = raw_data[raw_data["MaSV_school"] == additional_selection]

            df1 = raw_data[["TenMH", "NHHK", "DiemHP"]].copy()
            df1["DiemHP"] = df1["DiemHP"].replace('', pd.NA).dropna().astype(float)
            df1["NHHK"] = df1["NHHK"].apply(lambda x: str(x)[:4] + " S " + str(x)[4:])

            selected_TenMH = " " + course
            filtered_df1 = df1[df1["TenMH"] == selected_TenMH]

            mean_DiemHP = (
                filtered_df1.groupby("NHHK")["DiemHP"]
                .mean()
                .round(1)
                .reset_index(name="Mean")
            )

            if year != "All":
                st.write("")
            else:
                fig = px.line(
                    mean_DiemHP,
                    x="NHHK",
                    y="Mean",
                    title=f"Mean Course Score for{selected_TenMH} through Semeters",
                )
                fig.update_layout(xaxis_title="Semeters",height=400, width=400)
                st.plotly_chart(fig, use_container_width=True)
    variables_to_delete = [
        'raw_data1', 'df1', 'filtered_df1', 'mean_DiemHP', 'counts', 'bins',
        'total_count', 'frequencies_percentage', 'grade_bins', 'fig1',
        'common_elements', 'merged_array', 'list3', 'dfac', 'fig', 'new_df',
        'new_dfa', 'new1_df', 'new1_dfa', 'course_data',
        'options', 'valid_courses', 'list2', 'list1'
    ]

    for variable in variables_to_delete:
        if variable in locals():
            del locals()[variable]



elif tabs == "Prediction Performance":

    clear_resources()

    raw_data = score_table_for_student()
    raw_data["DTBTKH4"] = raw_data["DTBTK"]/25
    df=raw_data.copy()
    df["MaSV_school"] = df["MaSV"].str.slice(2, 4)
    df["Major"] = df["MaSV"].str.slice(0, 2)
    unique_values_major = df["Major"].unique()
    unique_values_major = [
        "BA",
        "BE",
        "BT",
        "CE",
        "EE",
        "EN",
        "EV",
        "IE",
        "MA",
        "SE",
        "IT",
    ]
    unique_values_major = sorted(unique_values_major, key=lambda s: s)
    col1, col2 = st.columns(2)
    with col1:
        major = st.selectbox("Select a school:", unique_values_major)
        df = filter_dataframe(df, "Major", major)

        unique_values_school = df["MaSV_school"].unique()
        all_values_school = np.concatenate([["All"], unique_values_school])
        no_numbers = [x for x in all_values_school if not re.search(r"\d", str(x))]

        if len(no_numbers) == 2:
            school = no_numbers[1]
    with col2:
        school = st.selectbox("Select a major:", no_numbers)

    df = filter_dataframe(df, "MaSV_school", school)
    predict = predict_late_student(df)
    rank = predict_rank(df)
    predict = pd.merge(predict, rank, on="MaSV")
    predict.rename(columns={"Mean_Cre": "Mean Credit"}, inplace=True)

    rank_mapping = {
        "Khá": "Good",
        "Trung Bình Khá": "Average good",
        "Giỏi": "Very good",
        "Kém": "Very weak",
        "Trung Bình": "Ordinary",
        "Yếu": "Weak",
        "Xuất Sắc": "Excellent",
    }
    predict["Pred Rank"].replace(rank_mapping, inplace=True)

    df_late = predict

    MaSV = st.text_input("Enter Student ID:", key="MaSV")

    def clear_form():
        st.session_state["MaSV"] = ""

    if st.button("Clear", on_click=clear_form):
        MaSV = ""

    if MaSV:
        df_filtered = predict[predict["MaSV"] == MaSV]
        styled_table = (
            df_filtered[
                ["MaSV", "GPA", "Mean Credit", "Pred Rank", "Progress", "Semeters"]
            ]
            .style.applymap(color_cell)
            .format({"GPA": "{:.2f}", "Mean Credit": "{:.1f}", "Semeters": "{:.1f}"})
        )

        with st.container():
            st.table(styled_table)
            predict_one_student(df, MaSV)
    else:
        df_late = predict

        df_late["Year"] = 2000 + df_late["MaSV"].apply(get_year)
        df_late = df_late[
            (df_late["Year"] != currentYear - 1) & (df_late["Year"] != currentYear - 2)
        ]
        year = st.selectbox("Select Year", options=df_late["Year"].unique())
        df_filtered = df_late[df_late["Year"] == year]
        styled_table = (
            df_filtered[
                ["MaSV", "GPA", "Mean Credit", "Pred Rank", "Progress", "Semeters"]
            ]
            .style.applymap(color_cell)
            .format({"GPA": "{:.2f}", "Mean Credit": "{:.2f}", "Semeters": "{:.2f}"})
        )
        csv = df_filtered.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="Preidct data.csv">Download CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

        legend_order = [
            "Excellent",
            "Very good",
            "Good",
            "Average good",
            "Ordinary",
            "Weak",
            "Very weak",
        ]

        fig1 = px.pie(
            df_filtered,
            names="Pred Rank",
            title="Pred Rank",
            color_discrete_sequence=px.colors.sequential.Mint,
            height=400,
            width=400,
            labels=legend_order,
        )

        fig2 = px.pie(
            df_filtered,
            names="Progress",
            title="Progress",
            color_discrete_sequence=px.colors.sequential.Peach,
            height=400,
            width=400,
        )

        fig1.update_layout(
            title={
                "text": "Pred Rank",
                "y": 0.95,
                "x": 0.35,
                "xanchor": "center",
                "yanchor": "top",
            }
        )
        fig2.update_layout(
            title={
                "text": "Progress",
                "y": 0.95,
                "x": 0.35,
                "xanchor": "center",
                "yanchor": "top",
            }
        )

        col3, col1, col2 = st.columns([2, 1, 1])
        with col3:
            st.dataframe(styled_table,use_container_width=True)
        with col1:
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            st.plotly_chart(fig2, use_container_width=True)
    variables_to_delete = [
    "raw_data",
    "df",
    "df_late",
    "MaSV",
    "predict",
    "rank",
    "rank_mapping",
    "styled_table",
    "df_filtered",
    "csv",
    "b64",
    "href",
    "legend_order",
    "fig1",
    "fig2",
    "col1",
    "col2",
    "col3"
    ]

    # Delete the variables after running the code
    for variable_name in variables_to_delete:
        if variable_name in locals():
            del locals()[variable_name]

elif tabs == "Grade Distribution Tables":
    clear_resources()
    raw_data = score_table()
    df = process_data(raw_data)
    additional_selection = " "

    unique_values_major = df["Major"].unique()
    unique_values_major = [
        "BA",
        "BE",
        "BT",
        "CE",
        "EE",
        "EN",
        "EV",
        "IE",
        "MA",
        "SE",
        "IT",
    ]
    unique_values_major = sorted(unique_values_major, key=lambda s: s)
    col1, col2 = st.columns(2)
    with col1:
        major = st.selectbox("Select a school:", unique_values_major)
        df = filter_dataframe(df, "Major", major)

        unique_values_school = df["MaSV_school"].unique()
        all_values_school = np.concatenate([["All"], unique_values_school])
        no_numbers = [x for x in all_values_school if not re.search(r"\d", str(x))]

        if len(no_numbers) == 2:
            school = no_numbers[1]
    with col2:
        school = st.selectbox("Select a major:", no_numbers)

    df = filter_dataframe(df, "MaSV_school", school)

    unique_values_year = df["Year"].unique()
    all_values_year = np.concatenate([["All"], unique_values_year])

    year = st.selectbox("Select a year:", all_values_year)

    options = df.columns[:-4]
    

    course_data_dict = {course: df[course].dropna() for course in options}
    
    valid_courses = [
        course for course, data in course_data_dict.items() if len(data) > 1
    ]

    course = "All"

    if st.button("Generate Chart"):
        courses_per_row = 4
        num_courses = len(valid_courses)
        num_rows = (num_courses + courses_per_row - 1) // courses_per_row

        for row in range(num_rows):
            start_index = row * courses_per_row
            end_index = min((row + 1) * courses_per_row, num_courses)
            courses_in_row = valid_courses[start_index:end_index]

            for course in courses_in_row:
                course_data = course_data_dict[course]
                course_data = course_data.astype(float)
                st.markdown(f"Course:  **{course}**")
                st.write("Number of examinations: ", str(len(course_data)))
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    counts, bins = np.histogram(course_data, bins=np.arange(0, 110, 10))
                    total_count = len(course_data)
                    frequencies_percentage = (counts / total_count) * 100
                    grade_bins = [
                        f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)
                    ]
                    result_array = []
                    cumulative_sum = 0

                    for element in frequencies_percentage:
                        cumulative_sum += element
                        result_array.append(cumulative_sum)

                    df = pd.DataFrame(
                        {
                            "Grade": grade_bins,
                            "Grading percentage": frequencies_percentage,
                            "Cumulative percentage": result_array
                        }
                    )
                    df["Grading percentage"] = df["Grading percentage"].map(
                        lambda x: "{:.2f}".format(x)
                    )
                    df["Cumulative percentage"] = df["Cumulative percentage"].map(
                        lambda x: "{:.2f}".format(x)
                    )

                    st.table(df)

                with col2:
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=bins[:-1],
                            y=frequencies_percentage,
                            mode="lines",
                            name="Frequency",
                        )
                    )

                    fig.update_layout(
                        title="Histogram of {}".format(course),
                        xaxis_title="Score",
                        yaxis_title="Percentage",
                        height=400,
                        width=400,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col3:
                    fig = go.Figure()
                    fig.add_trace(go.Box(y=course_data, name="Box plot"))
                    fig.update_layout(
                        title="Box plot",
                        yaxis_title="Score",
                        height=400,
                        width=400,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col4:
                    raw_data1 = raw_data.copy()
                    raw_data1["major"] = raw_data1["MaSV"].str.slice(0, 2)
                    raw_data1.replace(["WH", "VT", "I"], np.nan, inplace=True)
                    raw_data1 = raw_data1[~raw_data1["DiemHP"].isin(["P", "F", "PC"])]
                    if major != "All":
                        raw_data1 = raw_data1[raw_data1["major"] == major]

                    raw_data1["MaSV_school"] = raw_data1["MaSV"].str.slice(2, 4)
                    if school != "All":
                        raw_data1 = raw_data1[raw_data1["MaSV_school"] == school]

                    df1 = raw_data1[["TenMH", "NHHK", "DiemHP"]].copy()
                    df1["DiemHP"] = df1["DiemHP"].astype(float)
                    df1["NHHK"] = df1["NHHK"].apply(
                        lambda x: str(x)[:4] + " S " + str(x)[4:]
                    )

                    selected_TenMH = " " + course
                    filtered_df1 = df1[df1["TenMH"] == selected_TenMH]

                    mean_DiemHP = (
                        filtered_df1.groupby("NHHK")["DiemHP"]
                        .mean()
                        .round(1)
                        .reset_index(name="Mean")
                    )

                    if year != "All":
                        st.write("")
                    else:
                        fig = px.line(
                            mean_DiemHP,
                            x="NHHK",
                            y="Mean",
                            title=f"Mean DiemHP through Semesters",
                        )
                        fig.update_layout(height=400, width=400)
                        st.plotly_chart(fig, use_container_width=True)
                        del raw_data1, df1, filtered_df1, mean_DiemHP, counts, bins, total_count, frequencies_percentage, grade_bins, fig
        del course_data, course_data_dict,  valid_courses
    st.stop()
