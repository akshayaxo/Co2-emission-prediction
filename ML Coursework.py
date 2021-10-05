import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['figure.figsize'] = (9, 9)
from plotly import express as px, graph_objects as go
from plotly.subplots import make_subplots
from IPython.core.pylabtools import figsize
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import BayesianRidge, LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler

# Reading the csv file
co2_vehicles = pd.read_csv('CO2_Emissions_Canada.csv')
data_copy = co2_vehicles.copy()
data_copy.head(5)

f, ax = plt.subplots(figsize=(25, 7))
sns.set_theme(style="darkgrid")

x = co2_vehicles.Make.value_counts().sort_values()

ax = sns.barplot(data=co2_vehicles, x='Make', y='CO2 Emissions(g/km)', hue='Cylinders')
plt.title('Number of cylinders in respective vehicles')
plt.xticks(rotation=35)
plt.show()

# Corelation maps
corr = data_copy.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True)

data_copy.info()
data_copy.describe().T
data_copy['Vehicle Class'].unique()
data_copy['Model'].unique()

# Split Gears from Transmission Column
data_copy["Gears"] = data_copy['Transmission'].str.extract('(\d+)')
data_copy["Gears"] = data_copy["Gears"].fillna(0)
data_copy["Gears"] = pd.to_numeric(data_copy["Gears"])
data_copy['Transmission'] = data_copy['Transmission'].str.replace('\d+', '')
data_copy.head()

# Count of categorical data and Mean of CO2 for each category
cat_cols = data_copy.select_dtypes(include=object).columns.tolist()
for col in cat_cols:
    print(col)
    print(data_copy.groupby(col).agg({'CO2 Emissions(g/km)': [np.mean, 'count']}))

# Find index of fuel type = Natural Gas
data_copy_N = data_copy[data_copy["Fuel Type"] == "N"]
indexes = data_copy_N.index
data_copy_N

# Remove fuel Type N(Only One data)
for i in indexes:
    data_copy.drop(i, axis=0, inplace=True)

sns.heatmap(data_copy.corr(), annot=True)
plt.show()

# Distribution of CO2 Emissions in relation to number of Cylinders
plt.figure(figsize=(16, 7))
order = data_copy.groupby("Cylinders")["CO2 Emissions(g/km)"].median().sort_values(ascending=True).index
sns.boxplot(x="Cylinders", y="CO2 Emissions(g/km)", data=data_copy, order=order, width=0.5)
plt.title("Distribution of CO2 Emissions in relation to number of Cylinders", fontsize=15)
plt.xlabel("Cylinders", fontsize=12)
plt.ylabel("CO2 Emissions(g/km)", fontsize=12)
plt.axhline(data_copy["CO2 Emissions(g/km)"].median(), color='r', linestyle='dashed', linewidth=2)
plt.tight_layout()
plt.show()

# CO2 Emissions with Fuel Consumption Comb (L/100 km)
CO2_comb = data_copy.groupby(['Fuel Consumption Comb (L/100 km)'])['CO2 Emissions(g/km)'].mean().reset_index()
plt.figure(figsize=(25, 8))
sns.barplot(x="Fuel Consumption Comb (L/100 km)", y="CO2 Emissions(g/km)", data=CO2_comb,
            edgecolor=sns.color_palette("dark", 3))
plt.title('CO2 Emissions with Fuel Consumption Comb (L/100 km)', fontsize=15)
plt.xlabel('Fuel Consumption Comb (L/100 km)', fontsize=12)
plt.xticks(rotation=90, horizontalalignment='center', fontweight='light', fontsize='7')
plt.ylabel('CO2 Emissions(g/km)', fontsize=12)
plt.show()

# CO2 Emissions with Engine Size
CO2_engine = data_copy.groupby(['Engine Size(L)'])['CO2 Emissions(g/km)'].mean().reset_index()
plt.figure(figsize=(18, 8))
sns.barplot(x="Engine Size(L)", y="CO2 Emissions(g/km)", data=CO2_engine,
            edgecolor=sns.color_palette("dark", 3))
plt.title('CO2 Emissions with Engine Size', fontsize=15)
plt.xlabel('Engine Size', fontsize=12)
plt.ylabel('CO2 Emissions(g/km)', fontsize=12)
plt.show()

# Remove less important Attributes
data_copy.drop(['Make', 'Model', 'Vehicle Class', 'Fuel Consumption City (L/100 km)', 'Fuel Consumption Hwy (L/100 km)',
                'Fuel Consumption Comb (mpg)'], inplace=True, axis=1)
data_copy.head()

# One Hot Encoding of Fuel Type
dummy = pd.get_dummies(data_copy['Fuel Type'], prefix="Fuel_Type", drop_first=True)
dummy.head()

frames = [data_copy, dummy]
result = pd.concat(frames, axis=1)
result.head()

# One Hot Encoding of Transmission
dummy1 = pd.get_dummies(result['Transmission'], prefix="Transmission", drop_first=True)
dummy1.head()

frames1 = [result, dummy1]
preprocesseddata = pd.concat(frames1, axis=1)
preprocesseddata

# Remove Transmission and Fuel type after One Hot Encoding
preprocesseddata.drop(['Transmission', 'Fuel Type'], inplace=True, axis=1)
preprocesseddata.head()
preprocesseddata.info()

# Rename columns
preprocesseddata.rename(
    columns={'Fuel Consumption Comb (L/100 km)': 'Fuel_Consumption_Comb', 'Engine Size(L)': 'Engine_Size',
             'CO2 Emissions(g/km)': 'CO2_Emissions'}, inplace=True)

# Correlation of CO2 Emissions with other attributes
most_correlated = preprocesseddata.corr().abs()['CO2_Emissions'].sort_values(ascending=False)
most_correlated

# Remove the attribures having less correlation with Co2 Emission
# preprocesseddata.drop(['Fuel_Type_E', 'Transmission_AS', 'Transmission_AM'],inplace=True,axis=1)
preprocesseddata.drop(['Transmission_AM'], inplace=True, axis=1)
preprocesseddata
preprocesseddata.info()

X = preprocesseddata.drop(['CO2_Emissions'], axis=1)
y = preprocesseddata['CO2_Emissions']

# Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# Evaluate several ml models by training on training set and testing on testing set
def evaluate(X_train, X_test, y_train, y_test):
    # Names of models
    model_name_list = ['Linear Regression', 'KNeighbors Regression', 'Bayesian Ridge Regression',
                       'Support Vector Regressor', 'LogisticRegression']

    # Instantiate the models
    model1 = LinearRegression()
    model2 = KNeighborsRegressor(5)
    model3 = BayesianRidge()
    model4 = LinearSVR()
    model5 = LogisticRegression(random_state=0)

    # Dataframe for results
    results = pd.DataFrame(columns=['mae', 'rmse', 'r2score', 'rcross_mean'], index=model_name_list)
    from sklearn.model_selection import cross_val_score
    # Train and predict with each model
    for i, model in enumerate([model1, model2, model3, model4, model5]):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Metrics
        mae = np.mean(abs(predictions - y_test))
        rmse = mean_squared_error(y_test, predictions, squared=False)
        r2score = r2_score(y_test, predictions)
        # Cross Validation of different models
        rcross = cross_val_score(model, X_train, y_train, cv=4)
        # Insert results into the dataframe
        model_name = model_name_list[i]
        rcross_mean = rcross.mean()
        results.loc[model_name, :] = [mae, rmse, r2score, rcross_mean]
    return results


Scaler = StandardScaler()  # We create a scaling object.
Scaler.fit(X_train)  # We fit this to x_train.
x_train_scaled = Scaler.transform(X_train)
x_test_scaled = Scaler.transform(X_test)
results = evaluate(x_train_scaled, x_test_scaled, y_train, y_test)
print(results)

figsize(12, 8)
matplotlib.rcParams['font.size'] = 12
# Root mean squared error
# ax1 = plt.subplot(1, 2, 1)
# results.sort_values('mae', ascending=True).plot.bar(y='mae', color='b', ax=ax1)
# plt.title('Model Mean Absolute Error');
# plt.ylabel('MAE');

# Median absolute percentage error
ax2 = plt.subplot(1, 2, 2)
results.sort_values('rmse', ascending=True).plot.bar(y='rmse', color='r', ax=ax2)
plt.title('Model Root Mean Squared Error');
plt.ylabel('RMSE');
plt.tight_layout()

# R2 Score
# ax3 = plt.subplot(1, 1, 1)
# results.sort_values('r2score', ascending=True).plot.bar(y='r2score', color='g', ax=ax3)
# plt.title('R2 Score');
# plt.ylabel('R2 Score');
