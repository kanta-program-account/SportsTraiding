import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mglearn
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, ward
import graphviz
from ydata_profiling import ProfileReport
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split

class SportsTrading:
    """_summary_
    """
    
    def __init__(self):
        # Load the data and preprocess
        self.df_train = pd.read_csv("./match_snapshot_public_training - training.csv").drop(columns='uuid')
        self.df_test = pd.read_csv("./match_snapshot_public_training - training.csv").drop(columns='uuid')
        
        # A features DataFrame and a target DataFrame.
        self.df_train_X = self.df_train.drop(columns='final_delta')
        self.df_train_y = self.df_train['final_delta']
        
        # A features list and a target list.
        self.X = self.df_train_X.values
        self.y = self.df_train_y.values
        
        # Get feature and target names.
        self.feature_columns = self.df_train.drop(columns='final_delta').columns.tolist()
        self.target_column = 'final_delta'

        # Define dictionary of models.
        self.models = {
            'LogisticRegression': LogisticRegression(max_iter=2000),
            'LinearSVC': LinearSVC(max_iter=2000),
            'DecisionTreeClassifier': DecisionTreeClassifier(random_state=0),
            'KNeighborsClassifier': KNeighborsClassifier(),
            'LinearRegression': LinearRegression(),
            'RandomForestClassifier': RandomForestClassifier(random_state=0),
            'GradientBoostingClassifier': GradientBoostingClassifier(random_state=0),
            'MLPClassifier': MLPClassifier(max_iter=2000, random_state=0)
        }
        
        # Define dictionary of scalers.
        self.scalers = {
            'MinMaxScaler': MinMaxScaler(),
            'StandardScaler': StandardScaler(),
            'RobustScaler': RobustScaler(),
            'Normalizer': Normalizer(),
        }
        
        # Define dictionary of feature importances.
        self.feature_importances_map = {
            'DecisionTreeClassifier': 0,
            'RandomForestClassifier': 0,
            'GradientBoostingClassifier': 0
            }
        
        # Initialize dictionaries, a dataframe, and an array.
        self.df_test_scores_all_models = pd.DataFrame()
        self.mean_scores = []

        
    def Get(self) -> pd.DataFrame:
        """Retrieve the dataset as a DataFrame.

        Returns:
            pandas.core.frame.DataFrame: A DataFrame containing the feature data of the dataset.
        """
        
        # Adjust the display limit.
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        
        return self.df_train
    
    def Describe(self, df) -> pd.DataFrame:
        """
        Generate descriptive statistics for a dataset.

        Returns:
            pd.DataFrame: About basic statistics.
        """
        
        print(f"shape: {df.shape}")
        print(f"feature names: {self.feature_columns}")
        print(f"target name: {self.target_column}")
        print(df[self.target_column].value_counts())
        
        return df.describe()
    
    def PlotBox(self, df):
        """
        Generate boxplots for specified columns in the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.
            column_div (str): Column name to exclude from boxplots.
        """
        
        # Set the figure size for the plots
        fig = plt.figure(figsize=(20, 20))
            
        # Exclude the specified column from the list of columns to plot
        columns = df.columns
        
        # Determine layout for subplots (e.g., 4 columns per row)
        n_cols = 4
        n_rows = (len(columns) // n_cols) + 1  # Calculate required rows
        
        # Loop through the specified columns
        for column_index, column in enumerate(columns):
            # Configure the subplot layout
            plt.subplot(n_rows, n_cols, column_index + 1)

            # Display the boxplot for the current column
            df[column].plot(kind="box")

        # Adjust layout to prevent overlapping
        plt.tight_layout()
        plt.show()
    
    def RemoveOutliers(self, df, target_column):
        """
        Remove outliers from the target column in the DataFrame based on IQR.

        Args:
            df (pd.DataFrame): Input DataFrame.
            target_column (str): Column name from which to remove outliers.

        Returns:
            pd.DataFrame: DataFrame with outliers removed from the target column.
        """
        # Calculate Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = df[target_column].quantile(0.25)
        Q3 = df[target_column].quantile(0.75)
        
        # Calculate IQR
        IQR = Q3 - Q1
        
        # Define the lower and upper bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter out rows with outliers
        df_filtered = df[(df[target_column] >= lower_bound) & (df[target_column] <= upper_bound)].reset_index(drop=True)

        return df_filtered
        
    def RemoveAnomalies(self, df, column_name: str, min_value: float, max_value: float) -> None:
        """
        Filter the DataFrame to include rows where the specified column's values
        fall within the given range.

        Args:
            column_name (str): Name of the column to filter.
            min_value (float): Minimum value for the filter.
            max_value (float): Maximum value for the filter.

        Returns:
            None
        """
        
        df_no_anomalies = df[(df[column_name] >= min_value) & (df[column_name] <= max_value)].reset_index(drop=True)
        
        return df_no_anomalies
    
    def PairPlot(self, df) -> None:
        """
        Show the pairplot for each feature in the dataset.
        
        Args:
            df (pd.Dataframe, optional): The dataframe what you want to create a pairplot.
        """

        # Create a feature matrix.  
        # df_feature_matrix = df.drop(columns=[self.target_columns])  

        # Create a pairplot.
        sns.pairplot(df, hue=self.target_column, palette='Set1')
    
    def ProcessSingleModel(self, model_name: str, cv_results: dict[str, list]) -> pd.DataFrame:
            """Updates feature importances and records test scores for a single model based on cross-validation results.

            Args:
                model_name (str): 
                    The name of the model (ex. "KNeighborsClassifier", "LinearRegression").
                model (BaseEstimator): 
                    The machine learning model instance. Must support scikit-learn style APIs.
                cv_results (dict): 
                    Cross-validation results as returned by `cross_validate`. 
                    Expected keys include:
                    - 'train_score': List of training scores for each fold.
                    - 'test_score': List of test scores for each fold.
                    - 'estimator': List of fitted model instances for each fold.
                    
            Returns:
                df_test_scores_for_single_model: The all test scores for single model.
            """
            # Initialize an empty DataFrame for test scores.
            df_test_scores_for_single_model = pd.DataFrame()
            
            # Split train and test datasets according to Stratified K-Fold (SKF) rules.
            for fold_index, (train_score_for_single_fold, test_score_for_single_fold, fitted_estimator) in enumerate(zip(cv_results['train_score'], cv_results['test_score'], cv_results['estimator'])):
                
                # Update test scores DataFrame
                df_test_scores_for_single_model = pd.concat([df_test_scores_for_single_model, pd.DataFrame([test_score_for_single_fold])], ignore_index=True) # pd.concat(default: axis=0): Cobine dataframes.
                
                # Output scores.
                print("test score: {:.3f}   ".format(test_score_for_single_fold), "train score: {:.3f}".format(train_score_for_single_fold)) 
            
            print(f"mean test score: {df_test_scores_for_single_model.mean()[0]:.3f}")
            return df_test_scores_for_single_model
    
    
    
    
    
    
    
    def plot_dimensionality_reduction_results(self, X_transformed: np.ndarray, y: np.ndarray, model, feature_columns: list[str], method_name: str = "PCA") -> None:
        """
        Plot dimensionality reduction results (scatter plot and component heatmap).

        Args:
            X_transformed (np.ndarray): Transformed data (e.g., PCA or NMF output).
            y (np.ndarray): Target labels as a 1D numpy array.
            model: A fitted dimensionality reduction model (e.g., PCA or NMF).
            feature_columns (list): List of feature column names.
            method_name (str): Name of the dimensionality reduction method ("PCA" or "NMF").
        """
        
        n_components = X_transformed.shape[1]
        
        # Create a plot.
        for i in range((n_components - 1)):
            plt.figure(figsize=(8, 8))
            mglearn.discrete_scatter(X_transformed[:, i], X_transformed[:, i+1], y)
            plt.legend(np.unique(y), loc="best")
            plt.gca().set_aspect("equal")
            plt.xlabel(f"{method_name} Component {i+1}")
            plt.ylabel(f"{method_name} Component {i+2}")
            plt.title(f"{method_name} Scatter Plot: Component {i+1} vs Component {i+2}")
            plt.show()

        # Create a heatmap.
        plt.figure(figsize=(10, 5))
        plt.matshow(model.components_, cmap='viridis', fignum=1)
        plt.yticks(range(model.components_.shape[0]), [f"{method_name} Component {i+1}" for i in range(model.components_.shape[0])])
        plt.colorbar()
        plt.xticks(range(len(feature_columns)), feature_columns, rotation=90, ha='right')
        plt.xlabel("Feature")
        plt.ylabel(f"{method_name} Components")
        plt.title(f"{method_name} Component Heatmap")
        plt.show()
    
    def PlotPCA(self, X, y, feature_columns, n_components: int) -> tuple[pd.DataFrame, pd.DataFrame, PCA]: # tuple: A non-modifiable array. list: A modifiabl array.
        """Show  scatter plot of PCA scaled iris_data. 

        Args:
            n_components (int): Determine the numbar of dimensions as a parameter of PCA.

        Returns:
            X_scaled (pandas.core.frame.DataFrame): Dataframe of scaled iris data.
            df_pca (pandas.core.frame.DataFrame): Dataframe of scaled iris data after dimensionality reduction.
            pca (sklearn.decomposition._pca.PCA): A fitted PCA model that contains attributes to explain the results.
        """
        
        # Initialize the StandardScaler.
        scaler = self.scalers['StandardScaler']

        # Fit a scaler to X and transform.
        X_scaled = scaler.fit_transform(X)
        
        # Initialize a PCA.
        pca = PCA(n_components=n_components, random_state=0)

        # Fit a PCA to X_scaled data and transform using components extracted by PCA.
        X_pca = pca.fit_transform(X_scaled) 
        
        # Create a dataframe.
        df_X_scaled = pd.DataFrame(X_scaled, columns=feature_columns)

        # Create a dataframe.
        df_pca = pd.DataFrame(X_pca)

        self.plot_dimensionality_reduction_results(X_pca, y, pca, feature_columns, "PCA")
        
        # Contribution rate
        # np.set_printoptions(precision=5, suppress=True) 
        # print('explained variance ratio: {}'.format(pca.explained_variance_ratio_))
        
        # plt.bar([n for n in range(1, len(pca.explained_variance_ratio_)+1)], pca.explained_variance_ratio_)
        # plt.xlabel("PCA components")
        # plt.ylabel("rate(%)")

        return df_X_scaled, df_pca, pca
    
    def PlotNMF(self, X, y, feature_columns, n_components: int) -> tuple[pd.DataFrame, pd.DataFrame, PCA]:
        """Show  scatter plot of NMF scaled iris_data. 

        Args:
            n_components (_type_): Determine the numbar of components as a parameter of NMF.

        Returns:
            X_scaled_nmf (pandas.core.frame.DataFrame): Dataframe of scaled iris data.
            df_nmf (pandas.core.frame.DataFrame): Dataframe of scaled iris data after component extraction.
            nmf (sklearn.decomposition._pca.PCA): A fitted NMF model.
        """
        
        # Initialize a NMF.
        nmf = NMF(n_components=n_components, random_state=0, max_iter=2000)

        # Fit a NMF to X_scaled data and transform using components extracted by NMF.
        X_nmf = nmf.fit_transform(X)
        
        # Create a dataframe.
        df_X = pd.DataFrame(X, columns=feature_columns)
        
        # Create a dataframe.
        df_nmf = pd.DataFrame(X_nmf)
        
        self.plot_dimensionality_reduction_results(X_nmf, y, nmf, feature_columns, "NMF")
        
        return df_X, df_nmf, nmf
    
    def PlotTSNE(self, X, y) -> None:
        """Show t-SNE clustering. 

        This algorithm clusters the data without using labels, placing similar items close together and dissimilar items far apart.
        """
        
        # Initialize a TSNE.
        tsne = TSNE(random_state=0) 

        # Fit a TSNE and tranform.
        X_tsne = tsne.fit_transform(X)
        
        # Define a list of 11 high-visibility colors.
        colors = [
            "#FF5733",  # Bright Red-Orange
            "#33FF57",  # Bright Green
            "#3357FF",  # Bright Blue
            "#FFD700",  # Gold
            "#FF33FF",  # Magenta
            "#00FFFF",  # Cyan
            "#FF8C00",  # Dark Orange
            "#8A2BE2",  # Blue Violet
            "#FF1493",  # Deep Pink
            "#32CD32",  # Lime Green
            "#800000",  # Maroon
        ]
        
        # Create a plot.
        plt.figure(figsize=(10, 10))
        plt.xlim(X_tsne[:, 0].min(), X_tsne[:, 0].max() + 1)
        plt.ylim(X_tsne[:, 1].min(), X_tsne[:, 1].max() + 1)
        for i in range(len(X)):
            plt.text(X_tsne[i, 0], X_tsne[i, 1], str(y[i]), color=colors[y[i]], fontdict={'weight': 'bold', 'size': 9})
        plt.xlabel("t-SNE feature 0")
        plt.ylabel("t-SNE feature 1")
        
    def PlotDendrogram(self, X, truncate :bool=False) -> None:
        """Show dendrogram which visualizes the clustering process. 

        Args:
            truncate (bool, optional): Determine the type of dendrogram, original or reduced. Defaults to False.
        """
        
        # Define a linkage.
        linkage_array = ward(X) # ward: Return bridge length that stored in an array

        # Display a part of dendrogram or a completed dendrogram.
        dendrogram(linkage_array, p=10, truncate_mode='lastp') if truncate else dendrogram(linkage_array)
        
        # Get a current figure.
        fig = plt.gcf()
        
        # Set a figure size.
        fig.set_size_inches(12, 8)
        
        # Get a current axes.
        ax = plt.gca()

        # Get a min and max values from ax.
        bounds = ax.get_xbound()

        # Set labels.
        plt.xlabel("Sample")
        plt.ylabel("Cluster distance")

        plt.show()
    
    def PlotFeatureImportancesAll(self) -> None:
        """Show bar graphs which refers feature importances for each tree classifier.
        """

        # Get the number of features.
        n_features = len(self.feature_names)

        print(self.feature_importances_map)

        # Create a graph.
        for model_name, importance in self.feature_importances_map.items():
            
            plt.barh(range(n_features), importance, align='center')
            plt.yticks(np.arange(n_features), self.feature_names)
            plt.xlabel(f"Feature importance : {model_name}")
            plt.show()
            