import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from scipy.stats import iqr
from PIL import Image

# df = pd.read_csv("trial.csv", index_col=-1)
# df = pd.read_csv("trial.csv", index_col="name")
# df = pd.read_csv("trial.csv")
# df1 = pd.DataFrame(data=df, columns=["mpg"])
# df2 = pd.read_csv("divorce.csv")
# df3 = pd.read_csv("data3.csv")
# df4 = pd.read_csv("data4.csv")
# df5 = pd.read_csv("data5.csv")

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

# df4 = pd.read_csv("student_habits_performance.csv")
# df4 = pd.read_csv("student_habits_performance.csv", index_col="student_id")

# df4.set_index(["student_id", "diet_quality"], inplace=True)

# df4.dropna(inplace=True)

# df4.reset_index(inplace=True, drop=True)
# df4.reset_index(inplace=True)

# print(df4[df4["age"].isin([18, 29])])

# print(df4.loc[["S1022", "S1975"], ["age", "exam_score"]])
# print(df4.loc[("S1022", "S1975"), ("age", "exam_score")])
# print(df4.loc[[("S1022", "High School"), ("S1004", "Master")], ("age", "exam_score")])
# print(df4.loc[("S1004", "Master") : ("S1022", "High School"), ("age", "exam_score")])

# df4.sort_index(level=["student_id", "parental_education_level"], ascending=[True, False],inplace=True)
# print(df4.head())

# df4["gender"].hist()
# plt.bar(data=df4, height="social_media_hours", x="age")
# df4.plot(x="age", y="social_media_hours", kind="scatter")

# df4[df4["gender"] == "Male"]["sleep_hours"].hist()
# df4[df4["gender"] == "Female"]["sleep_hours"].hist()
# plt.legend(["Male", "Female"])
# plt.show()

# print(df4.isna().sum())
# print(df4.isna().any())
# print(df4.isna().all())

# print(df4.value_counts("gender"))
# print(df4["gender"].value_counts())
# print(df4["gender"].count())

# df4["gender"].hist()
# sns.countplot(data=df4, x="gender")
# sns.histplot(data=df4, x="gender")
# sns.boxplot(data=df4, x="gender", y="social_media_hours")
# plt.show()

# ------------------------------------------------------------------------------

# print(df3.pivot_table("age", "student_id"))
# print(sum([df3["age"], df3["social_media_hours"]]))
# print(df3.groupby())
# print(df3[["age", "social_media_hours"]].std(axis=1).head())
# print(pd.DataFrame())

# ------------------------------------------------------------------------------

# ( Joining Data by Merge() method )
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

# print(df3.merge(df4, left_on="age", right_on="Age"))
# print(df3.merge(df4, left_index=True, right_index=True))
# print(df3.merge(df4))

# ------------------------------------------------------------------------------

# ( Semi-join )
# semi = df3.merge(df4, on="gender")
# top = df3[df3["gender"].isin(semi["gender"])]
# print(top)

# # ( anti join )
# anti = df3.merge(df4, on="gender", how="left", indicator=True)
# lst = anti.loc[anti["_merge"] == "left_only", "gender"]
# non_top = df3[df3["gender"].isin(lst)]
# print(non_top)

# ------------------------------------------------------------------------------

# ( Joining Data by Concat() method )
# print(pd.concat([df3, df4, df5], ignore_index=True))
# print(pd.concat([df3, df4, df5], keys=["Table1", "Table2", "Table3"]))
# print(pd.concat([df3, df4, df5], keys=["Table1", "Table2", "Table3"], sort=True)) # the sorting is applied on the columns names

# ------------------------------------------------------------------------------

# ( Joining Data by Merge_order() method )
# print(pd.merge_ordered(df3, df4))
# print(df3.merge(df4, how="outer"))

# print(pd.merge_ordered(df3, df4, how="inner"))
# print(df3.merge(df4))

# print(pd.merge_ordered(df3, df4, how="left"))
# print(df3.merge(df4, how="left"))

# print(pd.merge_ordered(df3, df4, how="right"))
# print(df3.merge(df4, how="right"))

# ------------------------------------------------------------------------------

# ( Joining Data by Merge_order() method )

# df3.sort_values("age", inplace=True)
# df4.sort_values("Age", inplace=True)
# print(pd.merge_asof(df3, df4, left_on="age", right_on="Age"))

# ------------------------------------------------------------------------------

# ( query() method )
# print(df3.query("age > 20 and age < 23 and gender == 'Male'"))
# print(df3[(df3["age"] > 20) & (df3["age"] < 23) & (df3["gender"] == "Male")])

# ------------------------------------------------------------------------------

# ( melt() method )
# print(df3.melt(id_vars=["student_id", "age"]))
# print(df3.melt(id_vars=["student_id", "age"], value_vars="gender"))
# print(df3.melt(id_vars=["student_id", "age"], value_vars="gender", var_name="Sex", value_name="M | F"))

# ------------------------------------------------------------------------------

# df = pd.read_csv("student_habits_performance.csv")

# print(list(df.columns))
# print(df.info())
# print(df.describe())
# print(df.head())

# (socil media hours)

# female = df[df["gender"] == "Female"]
# female.reset_index(inplace=True)

# print(female["social_media_hours"].index.max())
# print(female["social_media_hours"].max())
# print(female[female["social_media_hours"] == female["social_media_hours"].max()])

# print(female.loc[female["social_media_hours"].idxmax()])

# ------------------------------------------------------------------------------

# data = pd.read_csv("student_habits_performance.csv")

# data.dropna(inplace=True)

# ( Mode )
# print(data["age"].mode())
# print(statistics.mode(data["age"]))

# ( Variance )
# print(data["age"].var())
# print(np.var(data["age"], ddof=1))
# print(statistics.variance(data["age"]))

# ( Standard deviation )
# print(data["age"].std())
# print(np.std(data["age"], ddof=1))
# print(statistics.stdev(data["age"]))

# ( Quantiles )
# print(data["age"].quantile([0.25, 0.5, 0.75]))
# print(np.quantile(data["age"], [0.25, 0.5, 0.75]))
# print(statistics.quantiles(data["age"]))

# print(data["age"].quantile(np.linspace(0, 1, 5)))
# print(np.quantile(data["age"], np.linspace(0, 1, 5)))

# ( Inquartile range )
# print(data["age"].quantile(0.75) - data["age"].quantile(0.25))
# print(np.quantile(data["age"], 0.75) - np.quantile(data["age"], 0.25))
# print(iqr(data["age"]))

# ( How to find the min and max data point & get on the outliers )
# iqr = iqr(data["age"])
# lower_point = data["age"].quantile(0.25) - 1.5 * iqr
# upper_point = data["age"].quantile(0.75) + 1.5 * iqr
# print(data[(data["age"] < lower_point) | (data["age"] > upper_point)])

# plt.boxplot(data=data, x="age")
# plt.show()

# ------------------------------------------------------------------------------

# ( Types of distributions )

# data = pd.read_csv("student_habits_performance.csv")

# data.dropna(inplace=True)

# np.random.seed(123)
# print(data.sample(5))
# print(data.sample(5, replace=True))

# ( cdf --> Cumulative Distribution Function )
# ( rvs --> random variates samples )
# ( pmf --> Probability Mass Function )
# ( ppf --> Percent Point Function )

# ( Uniform Distribution --> Continuous )
# from scipy.stats import uniform
# print(uniform.cdf(7, 1, 12)) # p(1 <= x <= 7) = (7 - 1) / 12
# print(uniform.rvs(0 , 5, size=10)) # 10 random numbers in range 0 and 5 to uniform distribution

# plt.hist(uniform.rvs(0 , 5, size=10000))
# sns.kdeplot(uniform.rvs(0 , 5, size=10000))
# plt.show()

# ( Binomial Distribution --> Discrete )
# from scipy.stats import binom
# print(binom.rvs(2, 0.5, size=8)) # Flip 2 coins with 50% chance of success 1 time
# print(binom.pmf(7, 10, 0.5)) # the probability of 7 successes from 10 trials with 50% chance --> P(x = 7)
# print(binom.cdf(7, 10, 0.5)) # the probability of 7 successes or fewer from 10 trials with 50% chance --> P(x <= 7)

# plt.hist(binom.rvs(10, 0.5, size=10000))
# sns.kdeplot(binom.rvs(10, 0.5, size=100000))
# plt.show()

# ( Normal Distribution --> Continuous )
# from scipy.stats import norm
# print(norm.cdf(10, 20, 3)) # cdf(value, mean, std)
# print(norm.ppf(0.25, 20, 3)) # ppf(percent, mean, std)
# print(norm.rvs(20, 3, 10))

# plt.hist(norm.rvs(20, 3, 100000))
# sns.kdeplot(norm.rvs(20, 3, 100000))
# plt.show()

# ( Poisson Distribution --> Discrete )
# from scipy.stats import poisson

# print(poisson.pmf(2, 5)) # If the average number of adoptions per week is 5, what is P (adoptions in a week = 2)
# print(poisson.cdf(2, 5)) # If the average number of adoptions per week is 5, what is P (adoptions in a week <= 2)
# print(poisson.rvs(8, size=5)) # rvs(λ, size)

# plt.hist(poisson.rvs(8, size=100000))
# sns.kdeplot(poisson.rvs(8, size=100000))
# plt.show()

# ( Exponential Distribution --> Continuous )
# from scipy.stats import expon

# print(expon.cdf(5, scale=20)) # cdf(x, λ) --> p(something < x)
# print(expon.rvs(5, scale=20, size=10))

# plt.hist(expon.rvs(5, scale=20, size=100000))
# sns.kdeplot(expon.rvs(5, scale=20, size=10000))
# plt.show()

# ------------------------------------------------------------------------------

# ( Correlation )

# data = pd.read_csv("student_habits_performance.csv")

# data.dropna(inplace=True)

# int_columns = data.select_dtypes("number")

# print(int_columns["study_hours_per_day"].corr(int_columns["exam_score"]))

# sns.heatmap(int_columns.corr(), annot=True)
# sns.scatterplot(data=int_columns,  x="exam_score", y="study_hours_per_day")
# sns.scatterplot(data=int_columns, x="exam_score", y="netflix_hours")

# Note on lmplot() : the line is called the regression line, you can with the slope of this line to know the relationship between the two continuous variables
# sns.lmplot(data=int_columns,  x="exam_score", y="study_hours_per_day", ci=None)
# sns.lmplot(data=int_columns,  x="exam_score", y="netflix_hours", ci=None)
# sns.lmplot(data=data,  x="exam_score", y="study_hours_per_day", ci=None, col="gender", hue="gender")
# sns.lmplot(data=data,  x="exam_score", y="netflix_hours", ci=None, col="gender", hue="gender")
# plt.show()

# ------------------------------------------------------------------------------

# ( Introduction to Data Visualization with Matplotlib )

# data = pd.read_csv("student_habits_performance.csv")

# df1 = np.array(["Jan", "Feb", "Mar", "Apr", "May", "Jun"])
# df2_1 = np.array([10, 5, 20, 18, 40, 20])
# df2_2 = np.array([20, 15, 26, 18, 50, 40])
# df2_3 = np.array([15, 55, 60, 38, 20, 10])
# df2_4 = np.array([15, 25, 60, 5, 10, 20])

# fig, ax = plt.subplots()
# ax.plot(df1, df2_1, marker="o", linestyle="--", color="orange")
# ax.set_xlabel("Months")
# ax.set_ylabel("Profit")
# ax.set_title("Profits througout The Months")
# plt.show()

# fig, ax = plt.subplots(2, 2, sharey=True, sharex=True)
# ax[0, 0].plot(df1, df2_1, marker="o", linestyle="--", color="red")
# ax[0, 1].plot(df1, df2_2, marker="o", linestyle="--", color="blue")
# ax[1, 0].plot(df1, df2_3, marker="o", linestyle="--", color="green")
# ax[1, 1].plot(df1, df2_4, marker="o", linestyle="--", color="orange")
# plt.show()

# ( Plotting Tiem Series )
# data = pd.read_csv("stock_data.csv")
# data2 = pd.read_csv("student_habits_performance.csv")

# data.drop("Unnamed: 0", inplace=True, axis=1)
# data["Date"] = pd.to_datetime(data["Date"])
# print(data["Date"].dt.year.nunique())

# fig, ax = plt.subplots()
# ax.plot(data["Date"], data["Open"])

# ( To customize the style )
# plt.style.use("bmh")
# fig.set_size_inches(10, 8)

# date_index = data.set_index("Date")
# decade = date_index["2010-01-01" : "2019-12-31"]
# ax.plot(decade.index, decade["Open"])

# date_index = data.set_index("Date")
# year = date_index["2010-01-01" : "2010-12-31"]
# ax.plot(year.index, year["Open"])

# date_index = data.set_index("Date")
# year1 = date_index["2011-01-01":"2011-12-31"]
# year2 = date_index["2014-01-01":"2014-12-31"]

# ax.plot(year1.index, year1["High"], color="blue")
# ax.set_xlabel("Year 2011", color="blue")
# ax.tick_params("x", colors="blue")

# ax2 = ax.twiny()
# ax2.plot(year2.index, year1["Open"], color="red")
# ax2.set_xlabel("Year 2014", color="red")
# ax2.tick_params("x", colors="red")
# ax2.annotate(
#     "Highest Point in 2014",
#     xy=(pd.Timestamp("2014-06-01"), 1),
#     xytext=(pd.Timestamp("2014-06-01"), -0.2),
#     arrowprops={"arrowstyle": "->", "color": "gray"},
# )

# date_index = data.set_index("Date")
# decade = date_index["2010-01-01" : "2019-12-31"]
# ax.plot(decade.index, decade["Open"])
# ax.set_xticklabels(decade.index, rotation = 30) # This method is not suitable with time series
# plt.xticks(rotation=45) # This is useful for anything
# plt.show()

# data["year"] = data["Date"].dt.year
# decade = data[(data["year"] <= 2019) & (data["year"] >= 2010)]
# ax.bar(decade["year"], decade["Open"], label="Open")
# ax.bar(decade["year"], decade["Close"], bottom=decade["Open"], label="Close")
# ax.bar(decade["year"], decade["High"], bottom = decade["Close"] + decade["Open"], label="High")
# ax.bar(decade["year"], decade["Low"], bottom = decade["High"] + decade["Close"] + decade["Open"], label="Low")
# plt.legend()

# ax.bar("Open", data["Open"].mean())
# ax.bar("Close", data["Close"].mean())
# ax.bar("High", data["High"].mean())
# ax.bar("Low", data["Low"].mean())
# plt.show()

# ax.hist(data["Open"], label="Open", bins=[9, 17, 25, 33, 41, 49, 57, 65, 73], histtype="step")
# ax.hist(data["Close"], label="Close", bins=8, histtype="step")
# plt.legend()
# plt.show()

# ax.bar("Open", data["Open"].mean(), yerr=data["Open"].std())
# ax.set_ylabel("Height")
# plt.show()

# gender_group = data2.groupby("gender")["sleep_hours"].agg(["mean", "std"])
# ax.errorbar(gender_group.index, gender_group["mean"], yerr=gender_group["std"], marker="o", color="r")
# ax.bar(x=gender_group.index, height=gender_group["mean"], yerr=gender_group["std"], color=["red", "blue", "green"])
# plt.show()

# ax.scatter(data=data2, x="study_hours_per_day", y="exam_score", c=data2.index, cmap="coolwarm")
# plt.show()

# ( To share visualization with others )
# plt.savefig("scatter.png")

# ------------------------------------------------------------------------------

# ( Introduction to Data Visualization with Seaborn )

# tips = sns.load_dataset("tips")  # (DataFrame)
# tips.to_csv("tips.csv", index=False)
# data = pd.read_csv("tips.csv")  # (CSV file)

# sns.set_context("notebook")
# sns.set_context("paper")
# sns.set_context("poster")
# sns.set_context("talk")

# sns.scatterplot(data=data, x="total_bill", y="tip", hue="smoker")
# sns.heatmap(data.corr(numeric_only=True), annot=True)
# sns.lmplot(data=data, x="total_bill", y="tip", ci=None, hue="smoker")
# sns.scatterplot(data=data, x="total_bill", y="tip", hue="smoker", palette={"Yes" : "red", "No" : "blue"})
# sns.countplot(data=data, x="sex", hue="smoker")

# sns.relplot(data=data, x="total_bill", y="tip", hue="smoker", col="smoker", row="sex", kind="scatter")
# sns.relplot(data=data, x="total_bill", y="tip", hue="smoker", col="smoker", row="sex", kind="line")
# sns.relplot(data=data, x="total_bill", y="tip", hue="smoker", col="day", kind="scatter", col_wrap=2)
# sns.relplot(data=data, x="total_bill", y="tip", kind="scatter", size="size", hue="size", style="smoker")
# sns.relplot(data=data, x="total_bill", y="tip", kind="line", size="size", hue="size", style="smoker")
# sns.relplot(data=data, x="total_bill", y="tip", kind="scatter", size="size", hue="size", style="smoker", alpha=0.5)
# sns.relplot(data=data, x="total_bill", y="tip", kind="line", style="smoker", hue="sex", col="sex", row="smoker", markers=True, dashes=False)

# sns.catplot(data=data, x="sex", y="total_bill", kind="box")
# sns.catplot(data, x="sex", y="total_bill", kind="bar", hue="sex")
# sns.catplot(data=data, x="sex", y="total_bill", kind="point", estimator=np.median, linestyle="none", capsize=0.2)
# sns.catplot(data=data, x="sex", y="total_bill", kind="point", estimator=np.median, linestyle="none", errorbar=None)
# sns.catplot(data=data, x="sex", y="total_bill", kind="point")
# sns.catplot(data=data, x="sex", y="total_bill", kind="point", estimator=np.median)
# sns.catplot(data=data, x="sex", y="total_bill", kind="point", estimator=np.std)
# sns.catplot(data=data, x="sex", y="total_bill", kind="point", errorbar="ci")
# sns.catplot(data=data, x="sex", y="total_bill", kind="point", errorbar="pi")
# sns.catplot(data=data, x="sex", y="total_bill", kind="point", errorbar="se")
# sns.catplot(data=data, x="sex", y="total_bill", kind="point", errorbar="sd")

# draw = sns.catplot(data=data, x="sex", y="total_bill", kind="point")
# draw.figure.suptitle("A visualization to a Pointplot", x=0.35, y=1)

# draw = sns.catplot(data, x="sex", y="total_bill", kind="bar", hue="sex", col="smoker")
# draw.figure.suptitle("Any Sex is Smoker")
# draw.set_titles("Is he\she Smoker : {col_name}")
# draw.set(xlabel="Gender", ylabel="The Bill")

# plt.show()

# ------------------------------------------------------------------------------

# ( Introduction to Functions in Python )

# value = 10
# def add():
#     # to convert from local to global
#     global value
#     value = value ** 2
#     return value
# print(add())

# def outer(x1, x2, x3):
#     def inner(x):
#         return x**2
#     return (inner(x1), inner(x2), inner(x3))
# print(outer(3, 5, 8))

# def outer(n1, s1):
#     def inner(n2, x2):
#         new = n2 * n1
#         word = s1 * x2
#         return(new, word)
#     return(inner(5, "word"))
# print(outer(10, 3))

# def outer(x1, x2, x3):
#     def inner(x):
#         new = x * (x1 + x2 + x3)
#         return new
#     return inner
# nums = outer(4, 5, 2)
# print(nums(10))

# def outer():
#     n = 1
#     def inner():
#         nonlocal n
#         n = 2
#         print(n)
#     inner()
#     return n
# print(outer())

# def add_all(*args):
#     sum_all = 0
#     for num in args:
#         sum_all += num
#     return sum_all
# print(add_all(5, 10, 20))

# def print_all(**kwargs):
#     for key, value in kwargs.items():
#         print(f"{key} : {value}")
# print_all(Name="Abdallah", Work_as="AI & ML Engineer")

# lambda_func = lambda x, y : x ** y
# print(lambda_func(10, 3))

# lst = [2, 4, 10, 5, 9]
# map_func = map(lambda x : x ** 2, lst)
# print(*map_func)

# def error_handling(num):
#     n = int(input("Enter the number: "))
#     if num < 0:
#         raise ValueError("X must be positive number")
#     try:
#         print(num / n)
#     except:
#         print("Invalid number, Try again")
#     finally:
#         print(f"your number was {n}")
# error_handling(10)
# error_handling(-4)

# ------------------------------------------------------------------------------

# ( Python Toolbox )

# word = "DataCamp"
# it = iter(word)
# print(next(it))
# print(next(it))
# print(next(it))
# print(next(it))
# print(*it)
# print(*word)

# with open("test.txt", "r") as file:
#     it = iter(file)
#     print(next(it))
#     print(next(it))
#     print(*it)
#     print(*file)

# lst = ["Abdallah", "Ahmad", "Mohammed", "Tolba"]
# print(list(enumerate(lst)))

# lst1 = [1, 2, 3, 4]
# lst2 = ["Abdallah", "Ahmad", "Mohammed", "Tolba"]
# print(list(zip(lst1, lst2)))

# even_nums = {num + 1: num for num in range(10) if num % 2 == 0}
# print(even_nums)

# def num_sequence(n):
#     i = 0
#     while i < n:
#         yield i
#         i += 1
# result = num_sequence(5)
# for item in result:
#     print(item)

# ------------------------------------------------------------------------------

# ( Exploratory Data Analysis in Python )

