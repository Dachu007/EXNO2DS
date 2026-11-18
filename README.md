# EXNO2DS
# AIM:
      To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT
```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

dt = pd.read_csv("/titanic_dataset.csv")
dt
```
<img width="920" height="317" alt="image" src="https://github.com/user-attachments/assets/5dd820bb-0088-4aa6-9689-33257be47593" />

```
dt.info()
```

<img width="485" height="258" alt="image" src="https://github.com/user-attachments/assets/d2e704cd-da74-4aae-b16d-a4f29ffb0132" />

```
dt.head()
```
<img width="901" height="196" alt="image" src="https://github.com/user-attachments/assets/42e97ee3-6db9-498e-89d9-38359bb35798" />

```
dt.tail()
```
<img width="830" height="213" alt="image" src="https://github.com/user-attachments/assets/f6eac944-8ab6-445d-8e67-8f65904fee3d" />

```
dt.shape
```
<img width="160" height="33" alt="image" src="https://github.com/user-attachments/assets/67a89adb-0f19-4b64-b913-eb34992113dd" />

```
dt.set_index("PassengerId",inplace=True)
dt
```
<img width="872" height="333" alt="image" src="https://github.com/user-attachments/assets/c0f5fde4-b1ee-4b15-9a4e-9112bfb99505" />

```
print("Data types:")
display(dt.dtypes.to_frame("Data Type"))
```
<img width="217" height="306" alt="image" src="https://github.com/user-attachments/assets/cf7c6991-31d5-4819-acba-14c3a090ae6e" />

```
print("Missing values:")
missing=pd.DataFrame({
    "Missing count":dt.isnull().sum(),
    "Missing percentage":(dt.isna().mean()*100).round(2)
})
display(missing.sort_values("Missing percentage",ascending=False))
```
<img width="351" height="312" alt="image" src="https://github.com/user-attachments/assets/3052a5a6-83a7-4cf2-8cc9-777de21f1f30" />

```
print("Exact duplicate rows:",dt.duplicated().sum())
print("Numeric summary")
display(dt.describe())
```
<img width="510" height="250" alt="image" src="https://github.com/user-attachments/assets/314c6335-3e9d-420d-ae41-0c37abd5251e" />

```
dt.nunique()
```
<img width="132" height="307" alt="image" src="https://github.com/user-attachments/assets/c9cb64f7-451e-4b6c-9b42-90c4ad37a35a" />

```
dt["Survived"].value_counts()
```
<img width="173" height="133" alt="image" src="https://github.com/user-attachments/assets/9def84b4-5ffa-4cca-903c-5d429a58869c" />

```
per=(dt["Survived"].value_counts()/dt.shape[0]*100).round(2)
per
```
<img width="161" height="135" alt="image" src="https://github.com/user-attachments/assets/24a501e8-71bf-412f-aeea-6172e9e508f3" />

```
sns.countplot(data=dt, x="Survived")
```
<img width="530" height="352" alt="image" src="https://github.com/user-attachments/assets/e6be9e08-183e-497c-965f-e79d6e708796" />

```
dt.Pclass.unique()
```
<img width="137" height="33" alt="image" src="https://github.com/user-attachments/assets/0d8ca6d7-dc3e-4deb-8ddf-4fef9ab8823b" />

```
dt.rename(columns={'Sex':'Gender'},inplace=True) 
dt
```
<img width="956" height="341" alt="image" src="https://github.com/user-attachments/assets/915d1ad0-35c8-4fe9-8cb3-751f218cba31" />

```
sns.catplot(x='Gender',col='Survived',kind='count',data=dt,height=5,aspect=.7)
```
<img width="553" height="388" alt="image" src="https://github.com/user-attachments/assets/24e80b20-e0fa-4032-bb11-2818a70f3ede" />

```
sns.catplot(x='Survived', hue="Gender", data=dt, kind = "count")
```
<img width="472" height="387" alt="image" src="https://github.com/user-attachments/assets/9d6f9614-b16d-489b-9f6e-16ca8fac18d9" />

```
dt.boxplot(column='Age',by='Survived')
```
<img width="452" height="367" alt="image" src="https://github.com/user-attachments/assets/b81da3ca-750d-4436-a331-8e4c045a9c27" />

```
sns.scatterplot(x=dt['Age'],y=dt['Fare'])
```
<img width="457" height="346" alt="image" src="https://github.com/user-attachments/assets/ed55196f-59a0-4f27-9ecd-a216fcf30b23" />

```
sns.jointplot(x="Age",y="Fare", data=dt)
```
<img width="468" height="451" alt="image" src="https://github.com/user-attachments/assets/9956751b-8dec-402f-acf0-931c2a75549e" />

```
fig, ax1= plt.subplots(figsize=(8,5))
pt=sns.boxplot(ax=ax1,x="Pclass",y='Age', hue='Gender', data=dt)
```
<img width="552" height="326" alt="image" src="https://github.com/user-attachments/assets/3d6b15ea-3324-4e2a-9be8-200ae6c22c50" />

```
sns.catplot(data=dt, col ="Survived", x = "Gender", hue="Pclass", kind = "count")
```
<img width="855" height="397" alt="image" src="https://github.com/user-attachments/assets/8a6397fb-6707-4bf0-970c-2a51a24928d8" />

```
corr=dt.corr(numeric_only=True)

sns.heatmap(corr,annot=True,cmap='coolwarm')
```
<img width="447" height="350" alt="image" src="https://github.com/user-attachments/assets/02596b72-91f9-4bc3-aa3d-ef2f2bc2422a" />

```
sns.pairplot(dt)
```
<img width="1017" height="662" alt="image" src="https://github.com/user-attachments/assets/bfc7977f-1d2e-47d8-9c06-03e5ece85adc" />



### Summary

In this experiment, Exploratory Data Analysis (EDA) was performed on the Titanic dataset using Python libraries such as Pandas, NumPy, Seaborn, and Matplotlib.First, the dataset was loaded using `read_csv()`. Methods such as `info()`, `shape()`, `head()`, and `tail()` were used to understand the structure and contents of the dataset.Next, data types, missing values, duplicate records, unique values, and statistical summaries were analyzed using Pandas functions.Different visualization techniques such as count plots, cat plots, boxplots, scatterplots, joint plots, and heatmaps were used to understand passenger survival, gender distribution, age, fare, passenger class, and correlations between numerical variables.These methods were selected because they help identify patterns, distributions, relationships, and outliers in the dataset.Other methods such as Pie Chart, Line Plot, Violin Plot, KDE Plot and Regression Plot were not selected because they were either less suitable for the analysis objectives or provided alternative ways of examining similar information.Finally, EDA helped in understanding the Titanic dataset and identifying important factors related to passenger survival.


# RESULT
Thus, the performance of EDA on the given data set is Verified  successfully along with the Visualization .
