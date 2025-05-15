import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# df = pd.read_csv("trial.csv", index_col=-1)
# df = pd.read_csv("trial.csv", index_col="name")
# df = pd.read_csv("trial.csv")
# df1 = pd.DataFrame(data=df, columns=["mpg"])
# df2 = pd.read_csv("divorce.csv")
df3 = pd.read_csv("data3.csv")
df4 = pd.read_csv("data4.csv")
df5 = pd.read_csv("data5.csv")

# print(df4.shape)

# print(df1.head())

# print(df.index.get_loc("ford torino"))

# print(df.loc["ford torino"])
# print(df.loc[4])
# print(df.iloc[4])
# print(df.loc[[5, 9, 3], ["name", "mpg"]])
# print(df.iloc[[5, 9, 3], [-1, 0]])
# print(df.loc[3:10, "mpg":"origin"])
# print(df.iloc[3:10, 0:8])
# print(df.loc[:, "mpg":"origin"])
# print(df.iloc[:, 0:8])
# print(df.iloc[:, [0, 5, 2]])

# print(df[df.index == "ford torino"])
# print(df[df.index.isin(["ford torino"])])
# print(df[df["origin"] == "europe"])

# print(df.loc(axis=0)["ford torino"])
# print(df.loc(axis=1)["name"])

# print(df.loc(axis=1)["mpg"])
# print(df["mpg"])

# lst = [1.73, 1.68, 1.71, 1.89]
# for i, x in enumerate(lst):
#     print(f"index {i} : {x}")

# height = np.array([1.73, 1.68, 1.71, 1.89, 1.79])
# weight = np.array([65.4, 59.2, 63.6, 88.4, 68.7])
# mix = np.array([height, weight])
# for i in np.nditer(mix):
#     print(i)

# for i, x in df.iterrows():
#     print(i)
#     print(x)
#     print("-----------------")

# for i, x in df.iterrows():
#     print(f"index {i} : {x['name']}")

# for i, x in df.iterrows():
#     df.loc[i, "length of name"] = len(x["name"])

# df["length of name"] = df["name"].apply(len)
# print(df)

# height = np.array([1.73, 1.68, 1.71, 1.89, 1.79])
# weight = np.array([65.4, 59.2, 63.6, 88.4, 68.7])
# mix = height * weight
# print(mix[np.logical_and(mix > 110, mix < 120)])
# print(mix[np.logical_or(mix > 110, mix < 120)])
# print(mix[np.logical_xor(mix > 110, mix < 120)])
# print(mix[np.logical_not(mix > 110, mix < 120)])

# print(df[df["mpg"] > 15]["name"])

# np.random.seed(123)
# print(np.random.rand(5, 2))
# print(np.random.randint(0,5, 10))
# print(np.random.random(5))
# print(np.random.randn(5))
# print(np.random.random_sample(5))
# print(np.random.ranf(5))
# print(np.random.rayleigh(5, 10))

# df1 = np.random.rand(1000) # Uniform distribution
# df2 = np.random.randint(0, 100, 1000) # Discrete uniform distribution
# df3 = np.random.random(1000) # Uniform distribution
# df5 = np.random.random_sample(1000) # Uniform distribution (this method is identical to np.random.random completely)
# df6 = np.random.ranf(1000) # Uniform distribution (this method is identical to np.random.random and np.random.random_sample completely)
# df4 = np.random.randn(10000) # Standard normal (Gaussian) distribution with mean 0 and standard deviation 1
# df7 = np.random.rayleigh(100, 10000) # Rayleigh distribution, a special case of the chi-squared distribution
# plt.hist(df7)
# plt.show()

# print(df.sample(10))
# print(df.shape)
# print(df.info())
# print(df.describe())
# print(list(df.columns))
# print(df.isnull().sum())
# print(df.values)
# print(df.value_counts())
# print(df.index)
# print(df.sort_values("mpg", ignore_index=True))
# print(df.sort_values(["mpg", "horsepower"], ignore_index=True))
# print(df.sort_values(["mpg", "acceleration"], ascending=[False, True]))
# print(df.sort_index(axis=1))

# print(df[df.isna().any(axis=1)].index) # to get on the indices for any row with any null value in any column
# print(df[df.isnull().any(axis=1)].index) # to get on the indices for any row with any null value in any column
# print(df.isna().any(axis=1).sum()) # to get on the sum of the rows have null values
# print(df.isnull().any(axis=1).sum()) # to get on the sum of the rows have null values
# print(df.iloc[32])

# print(df[["mpg", "horsepower"]].agg(["mean", "median", "std"]))
# print(df[["mpg", "horsepower"]].quantile([0.25, 0.5, 0.75]))
# print(df[["mpg", "horsepower"]].apply([np.mean, np.median]))
# print(df[["mpg", "horsepower"]].add(10))
# print(df[["mpg", "horsepower"]].cumsum()) # Cumulative sum
# print(df[["mpg", "horsepower"]].cummin()) # Cumulative
# print(df[["mpg", "horsepower"]].cummax()) # Cumulative
# print(df[["mpg", "horsepower"]].cumprod()) # Cumulative

# print(df["mpg"].agg(["mean", "median", "std", "var"]))
# print(df["mpg"].quantile([0.25, 0.5, 0.75]))
# print(df[["mpg", "cylinders", "horsepower"]].agg(["mean", "median", "std", "var"]))

# fig, ax = plt.subplots(4, 2)
# ax[0, 0].hist(df["mpg"].cumsum())
# ax[0, 1].hist(df["mpg"])

# ax[1, 0].hist(df["mpg"].cumprod())
# ax[1, 1].hist(df["mpg"])

# ax[2, 0].hist(df["mpg"].cummin())
# ax[2, 1].hist(df["mpg"])

# ax[3, 0].hist(df["mpg"].cummax())
# ax[3, 1].hist(df["mpg"])
# plt.show()

# ------------------------------------------------------------------------------

# df2 = pd.read_csv("clean_jobs.csv")

# df2.dropna(axis=1, inplace=True)
# print(df2.isnull().sum())

# df2.drop(["work_type", "employment_type"], axis=1, inplace=True)
# print(df2.isnull().sum())

# ------------------------------------------------------------------------------

# print(df3.isna().sum())
# print(len(list(df3.columns)))
# print(df3["parental_education_level"].value_counts())
# sns.countplot(data=df3, x="parental_education_level")
# plt.show()

# df3.dropna(inplace=True)
# print(df3.isna().sum())
# print(df3.shape)

# print(df3.info())
# print(df3.info(verbose=False))

# print(df3[["gender", "part_time_job", "parental_education_level"]].value_counts(sort=True))
# print(df3["part_time_job"].value_counts(sort=True))
# print(df3["gender"].value_counts(sort=True, normalize=True))
# print(df3["age"].value_counts(sort=True, bins=4))

# print(list(df3.columns))

# print(df3.groupby("student_id")["mental_health_rating"].mean())
# print(df3.groupby(["student_id", "diet_quality"])[["mental_health_rating", "sleep_hours"]].agg(["mean", "median"]))
# print(df3.groupby("student_id")["age"].mean(False))

# print(df3.groupby("student_id")["exam_score"].mean())
# print(df3.pivot_table("exam_score", index="student_id", columns="gender", fill_value=0, aggfunc=["mean", "median"], margins=True))

# print(df3.loc[(5, 15), ("age", "exam_score")])
# print(df3.loc[10:25, "study_hours_per_day":"parental_education_level"])

# ------------------------------------------------------------------------------

# df3 = pd.read_csv("student_habits_performance.csv")
# df3 = pd.read_csv("student_habits_performance.csv", index_col="student_id")

# df3.set_index(["student_id", "diet_quality"], inplace=True)

# df3.dropna(inplace=True)

# df3.reset_index(inplace=True, drop=True)
# df3.reset_index(inplace=True)

# print(df3[df3["age"].isin([18, 29])])

# print(df3.loc[["S1022", "S1975"], ["age", "exam_score"]])
# print(df3.loc[("S1022", "S1975"), ("age", "exam_score")])
# print(df3.loc[[("S1022", "High School"), ("S1004", "Master")], ("age", "exam_score")])
# print(df3.loc[("S1004", "Master") : ("S1022", "High School"), ("age", "exam_score")])

# df3.sort_index(level=["student_id", "parental_education_level"], ascending=[True, False],inplace=True)
# print(df3.head())

# df3["gender"].hist()
# plt.bar(data=df3, height="social_media_hours", x="age")
# df3.plot(x="age", y="social_media_hours", kind="scatter")

# df3[df3["gender"] == "Male"]["sleep_hours"].hist()
# df3[df3["gender"] == "Female"]["sleep_hours"].hist()
# plt.legend(["Male", "Female"])
# plt.show()

# print(df3.isna().sum())
# print(df3.isna().any())
# print(df3.isna().all())

# print(df3.value_counts("gender"))
# print(df3["gender"].value_counts())study_hours_per_day
# print(df3["gender"].count())

# df3["gender"].hist()
# sns.countplot(data=df3, x="gender")
# sns.histplot(data=df3, x="gender")
# sns.boxplot(data=df3, x="gender", y="social_media_hours")
# plt.show()

# ------------------------------------------------------------------------------

# print(df3.pivot_table("age", "student_id"))
# print(sum([df3["age"], df3["social_media_hours"]]))
# print(df3.groupby())
# print(df3[["age", "social_media_hours"]].std(axis=1).head())
# print(pd.DataFrame())

# ------------------------------------------------------------------------------

# print(df3.merge(df4, left_on="age", right_on="Age").shape)
# print(df3.merge(df4, left_on="age", right_on="Age"))

# print(df3.merge(df4, left_on="age", right_on="Age", how="outer").shape)
# print(df3.merge(df4, left_on="age", right_on="Age", how="outer"))

# print(df3.merge(df4, left_on="age", right_on="Age", how="left").shape)
# print(df3.merge(df4, left_on="age", right_on="Age", how="left"))

# print(df3.merge(df4, left_on="age", right_on="Age").shape)
# print(df3.merge(df4, left_on="age", right_on="Age", how="right"))

# print(df3.merge(df4, left_on="age", right_on="Age").merge(df5, on="Gender").shape)
# print(df3.merge(df4, left_on="age", right_on="Age").merge(df5, on="Gender"))

# print(df3.merge(df4, left_on="age", right_on="Age").shape)
# print(df3.merge(df4, left_on=["age", "gender"], right_on=["Age", "Gender"]))

# data = (df3.merge(df4, left_on=["age", "gender"], right_on=["Age", "Gender"]))
# data.groupby("gender")["social_media_hours"].sum().plot(kind="bar", y="Most_Used_Platform")
# plt.show()

# print(df3.merge(df3, on="age"))
# print(df3.merge(df3, on="age", how="left"))

# print(df3.merge(df4, left_on="age", right_on="Age", left_index=True, right_index=True))

# ------------------------------------------------------------------------------

# ( Semi-join )
semi = df3.merge(df4, on="gender")
top = df3[df3["gender"].isin(semi["gender"])]
print(top)

# ( anti join )
anti = df3.merge(df4, on="gender", how="left", indicator=True)
lst = anti.loc[anti["_merge"] == "left_only", "gender"]
non_top = df3[df3["gender"].isin(lst)]
print(non_top)