import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load and preprocess the data
data = pd.read_csv('https://raw.githubusercontent.com/ashiquebiniqbal/Global-Climate-Change-Data-forecsat/main/GlobalLandTemperatures.csv', parse_dates=['dt'])
data = data[['dt', 'AverageTemperature']].groupby('dt').mean().resample('MS').mean().reset_index().dropna()
data = data.set_index('dt')
for i in range(1, 13):
    data[f't_{i}'] = data['AverageTemperature'].shift(i)

data.dropna(inplace=True)

# Split the data into training and test sets
train_size = int(len(data) * 0.8)
X_train, X_test = data.iloc[:train_size, 1:], data.iloc[train_size:, 1:]
y_train, y_test = data.iloc[:train_size, 0], data.iloc[train_size:, 0]

# Train the random forest model
n_estimators = 100
rf = RandomForestRegressor(n_estimators=n_estimators)
rf.fit(X_train, y_train)

# Make one-step forecasts on the test set
y_pred = []
for i in range(len(X_test)):
    # Get the previous time step as input
    prev = X_test.iloc[i].values.reshape(1, -1)
    # Make a prediction for the next time step
    pred = rf.predict(prev)
    y_pred.append(pred[0])

# Evaluate the model using mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean squared error: {mse}")
