from data_module import DataModule
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing#this dataset is for testing purposes only
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

#This model is to be highly integrated with the gui.
#I don't think I need to add memorization or saving/loading capabilities here,
#so this will simply be a model object that the gui can interact with to create and evaluate linear regression models.
#As stated, all comments and docstrings will be in English for consistency with the rest of the codebase.
class LinearRegressionModel:
    def __init__(self):
        self.model = LinearRegression()
        self.initialized = False
        self.feature_names = None
        self.metrics_train = {'r2': None, 'mse': None}
        self.metrics_test = {'r2': None, 'mse': None}
        self.coef_ = None
        self.intercept_ = None
        self.regression_line = None
        self.summary = None

    def set_df(self,train_df = None, test_df = None):
        try:
            if not train_df.empty and not test_df.empty:
                self.x_train = train_df.iloc[:,1:]
                self.y_train = train_df.iloc[:,0]
                self.x_test  = test_df.iloc[:,1:]
                self.y_test  = test_df.iloc[:,0]
        except Exception as e:
            print(f"dataframe empty: {e}")

    def fit_and_evaluate(self):
        try:
            #In the gui you already can split data, so no need to do it here
            """Trains the model and evaluates both data splits"""
            #previous handler only works with series, but because we can do
            #multiple regression we can't use it.
            if isinstance(self.x_train, pd.Series):
                self.x_train = self.x_train.to_frame()
                self.x_test  = self.x_test.to_frame()
            self.y_train = self.y_train.to_frame()
            self.y_test  = self.y_test.to_frame()

            if not all(pd.api.types.is_numeric_dtype(self.x_train[col]) for col in self.x_train.columns):
                raise ValueError("All features must be valid numeric types")
            if not all (pd.api.types.is_numeric_dtype(self.y_train[col]) for col in self.y_train.columns):
                raise ValueError("The target must be valid numeric type")
            if self.x_train.isnull().any().any() or self.y_train.isnull().any().any():
                #We should not get here if gui is used properly
                raise ValueError("Data contains NaN")
        except Exception as e:
            #This will be used to make a Qmessage box in the future.
            return self.metrics_train, self.metrics_test, self.summary, e

        # We save feature names for formula visual representation
        self.feature_names = self.x_train.columns.tolist()

        # Fit and predictions
        # We only train the model, which is basically find the parameters
        # of our regression line, only with the train set. That's why we only
        # pass to fit the feats and target of the train model.
        self.model.fit(self.x_train, self.y_train)
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        # Train metrics and we store them
        #We don't need predicts for our own train set in order to generalize
        #and predicts things, this is not meant for that.
        #This is just to measure how well the model learned the data
        #not how much can predict with the data but how well the model learned it.
        y_train_pred = self.model.predict(self.x_train)
        self.metrics_train['mse'] = mean_squared_error(self.y_train, y_train_pred)
        self.metrics_train['r2'] = r2_score(self.y_train, y_train_pred)

        # Test metrics and we store them(DOD requests both)
        y_test_pred = self.model.predict(self.x_test)
        self.metrics_test['mse'] = mean_squared_error(self.y_test, y_test_pred)
        self.metrics_test['r2'] = r2_score(self.y_test, y_test_pred)

        self.initialized = True
        #returns a tuple with two dictionaries
        self.formula_string()
        self.summary = (f'Regression Line:\n {self.regression_line}\n\n'
                        'Train metrics:\n'
                        f'MSE : {self.metrics_train['mse']}\n'
                        f'R2  : {self.metrics_train['r2']}\n\n'
                        'Test metrics\n'
                        f'MSE : {self.metrics_test['mse']}\n'
                        f'R2  : {self.metrics_test['r2']}')
        return self.metrics_train, self.metrics_test, self.summary, None

    def formula_string(self):  # DOD requests formula representation
        if not self.initialized:
            return "Model not initialized"
        terms = []
        #sklearn returns a two dimensional array with the coeficient per
        #target, coef:.3f waits not for a 2d but 1d array so if let as before
        #we get an error. To solve this we only reduce the dimension of
        #self.coef_ with the method o np ravel(). eg: [[1,2,3,4]] -> [1,2,3,4]
        #Because we'll never have two outputs, two targets, is no use to have
        #a 2d array with coef.
        for name, coef in zip(self.feature_names, np.atleast_1d(np.ravel(self.coef_))):
            terms.append(f"{coef:.3f}*{name}")
        # Intercept with explicit sign
        b0 = float(self.intercept_)
        sign = '+' if b0 >= 0 else '-'
        self.regression_line = f"y = {' + '.join(terms)} {sign} {abs(b0):.3f}"
        return self.regression_line

    def can_plot(self):
        """Checks if the model can be plotted (1 feature)"""
        return self.initialized and len(self.feature_names) == 1

    def get_plot_item(self):
        #function used for testing purposes only
        #returning plt is the entire module. When we create a plot graph without
        #explicitly calling the class Figure "plt.figure - here we are calling the
        #method, not the class" we are storing the graph in a global Figure inside plt module.
        #Because of this we don't have control of the object figure and can't put it
        #inside a widget, if you want to show it you'll need to open a popup, but
        #can't be stored inside a widget, that's why it's better to use the
        #get_plot_figure function.

        """Generates a plot of the model predictions against actual data"""
        if not self.can_plot():
            raise ValueError("Only models with 1 feature can be plotted(two dimensions only)")
        elif not self.initialized:
            raise ValueError("Model not initialized") #We could also return None or an empty plot, but this informs the user better

        x_line = np.linspace(min(self.x_train.iloc[:,0]), max(self.x_train.iloc[:,0]), 100)
        y_line = self.model.predict(x_line.reshape(-1, 1))
        plt.figure(figsize=(12, 8))
        plt.scatter(self.x_train, self.y_train, color='blue', label='Train Data', alpha=0.5, s=20)  # alpha being transparency, s being point size
        plt.scatter(self.x_test, self.y_test, color='green', label='Test Data', alpha=0.5, s=20)
        plt.plot(x_line, y_line, color='red', label='Prediction Line', linewidth=3)
        plt.xlabel(self.feature_names[0])#DOD requests feature name in plot
        plt.ylabel("Target")
        plt.title("Linear Regression")
        plt.legend()
        return plt #Returning the plt object for further manipulation or display

    def get_plot_figure(self) -> Figure:
        """Return a Matplotlib Figure for simple linear regression (1 feature)."""
        if not self.can_plot():
            return None

        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)

        # Datos
        #Just to generate the line.
        x_line = np.linspace(
            self.x_train.iloc[:, 0].min(),
            self.x_train.iloc[:, 0].max(),
            100
        )
        y_line = self.model.predict(x_line.reshape(-1, 1))
        # Dibujar puntos y línea
        ax.scatter(self.x_train.iloc[:, 0], self.y_train, color='blue', label='Train', alpha=0.5)
        ax.scatter(self.x_test.iloc[:, 0], self.y_test, color='green', label='Test', alpha=0.5)
        ax.plot(x_line,
                y_line,
                color='red',
                linewidth=2,
                label= self.summary)

        # Etiquetas y formato
        ax.set_xlabel(f"{self.feature_names[0]}\n(feature)")
        ax.set_ylabel(f"{self.y_train.columns.tolist()[0]}\n(target)")
        ax.set_title("Linear Regression (Simple)")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)

        fig.tight_layout()
        return fig

    def predict(self, X):
        """Makes predictions on new data"""
        if not self.initialized:
            raise ValueError("Model not initialized")
        return self.model.predict(X)

    @property
    def is_initialized(self): #in case we need to check externally if the model is created
        return self.initialized

    @property
    def get_R2(self): #This two are to be called by gui after evaluation, to show results
        if self.initialized:
            return self.metrics_test['r2']#DOD requests test metrics only
        return 9999 #arbitrary high value to indicate uninitialized model, like if u get this you know something's wrong

    @property
    def get_MSE(self):
        if self.initialized:
            return self.metrics_test['mse']#DOD requests test metrics only
        return 9999 #arbitrary high value to indicate uninitialized model, like if u get this you know something's wrong


if __name__ == "__main__":
    print("Don't executes this as a main module")