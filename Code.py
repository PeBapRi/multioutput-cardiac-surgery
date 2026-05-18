# ============================================================
# IMPORTS AND WARNING CONFIGURATION
# ============================================================

# Ignore unnecessary warnings to keep output clean
import warnings
warnings.filterwarnings('ignore')

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

# Progress bar utility
from tqdm import tqdm

# Numerical and plotting libraries
import numpy as np
import matplotlib.pyplot as plt

# Base path where files are stored
path = 'Place your PC path here'

# Data manipulation
import pandas as pd

# Preprocessing utilities
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Dataset splitting
from sklearn.model_selection import train_test_split

# Multi-output regression support
from sklearn.multioutput import MultiOutputRegressor

# Machine learning models
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Evaluation metrics
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)

# Model saving/loading
import joblib    

# ============================================================
# FEATURE SELECTION IMPORTS
# ============================================================

# Main feature selection method
from sklearn.feature_selection import SelectKBest

# Feature selection scoring methods
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import f_classif

# Statistical feature filtering methods
from sklearn.feature_selection import SelectFdr
from sklearn.feature_selection import SelectFwe

# Permutation importance analysis
from sklearn.inspection import permutation_importance

# ============================================================
# REGRESSION MODELS
# ============================================================

from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    BayesianRidge,
    SGDRegressor
)

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    AdaBoostRegressor,
    BaggingRegressor
)

from sklearn.svm import SVR, NuSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor

# Classification metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)

# ============================================================
# LIST OF MODELS TO TEST
# ============================================================

# Currently using only Decision Tree Regressor
regressors = [
    ("DecisionTreeRegressor", DecisionTreeRegressor()),
]

# ============================================================
# FEATURE SELECTION FUNCTION
# ============================================================

def feature_selector(X_train, X_test, y_train, type, i):
    """
    Selects the best 'i' features using different statistical methods.

    Parameters:
    -----------
    X_train : Training feature set
    X_test  : Testing feature set
    y_train : Training labels
    type    : Feature selection method
    i       : Number of features to select

    Returns:
    --------
    Xt        : Selected training features
    Xtest     : Selected testing features
    cols_idxs : Indexes of selected columns
    """

    # --------------------------------------------------------
    # Feature selection method selection
    # --------------------------------------------------------

    if (type == 1):
        # ANOVA F-test for classification
        bestfeatures = SelectKBest(score_func=f_classif, k=i)

    elif(type == 2):
        # Mutual information selection
        bestfeatures = SelectKBest(score_func=mutual_info_classif, k=i)

    elif(type == 3):
        # Chi-square test
        bestfeatures = SelectKBest(score_func=chi2, k=i)

    elif(type == 4):
        # False Discovery Rate selection
        bestfeatures = SelectKBest(score_func=SelectFdr, k=i)

    elif(type == 5):
        # Family-wise error selection
        bestfeatures = SelectKBest(score_func=SelectFwe, k=i)

    # --------------------------------------------------------
    # Fit feature selector
    # --------------------------------------------------------

    fit = bestfeatures.fit(X_train, y_train)

    # Get selected feature indexes
    cols_idxs = fit.get_support(indices=True)

    # Extract selected features
    Xt = X_train.iloc[:, cols_idxs]
    Xtest = X_test.iloc[:, cols_idxs]

    return Xt, Xtest, cols_idxs

# ============================================================
# INITIAL VARIABLES
# ============================================================

featnumb = []          # Stores number of selected features
classifierName = []    # Stores classifier names
finalResult = []       # Stores final results

# ============================================================
# TARGET VARIABLES
# ============================================================

target = [
    'AF',
    'MI',
    'Poedema',
    'Pneumonia',
    'Stroke',
    'Infection',
    'AKI',
    'Bleeding',
    '1Y Mortality',
    '1 Y Reinterventions',
    '1 Y Hospital readmissions'
]

# ============================================================
# LOAD DATASET
# ============================================================

database = pd.read_excel(
    path + 'Database2.xlsx',
    sheet_name='Sheet1'
)

# Remove rows with missing values
database.dropna(axis='rows', inplace=True)

# Remove unnecessary columns
database.drop(
    ['Unnamed: 0', 'Nome', 'Target'],
    axis='columns',
    inplace=True
)

# Split into features and labels
X = database.loc[:, 'Hg Alta':]
y = database.loc[:, target]

# ============================================================
# LOAD PREVIOUSLY SELECTED FEATURES
# ============================================================

featuresNamesDF = pd.read_excel(
    path+'Outputs/FeaturesSelected4TargetNumberRandState22.xlsx'
)

featuresNamesDF = featuresNamesDF['11']
featuresNamesDF.dropna(axis=0, inplace=True)

# ============================================================
# FEATURE SELECTION OPTIMIZATION LOOP
# ============================================================

for i in range(len(target)):

    # Metric storage
    accTest = []
    recTest = []
    precTest = []
    f1Test = []

    featnumb = []

    df = pd.DataFrame()
    featsSelectedDFInd = pd.DataFrame()

    # --------------------------------------------------------
    # Test different numbers of selected features
    # --------------------------------------------------------

    for feat in range(len(featuresNamesDF)):

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y[target[i]],
            test_size=0.2,
            stratify=y[target[i]],
            random_state=22
        )

        y_test_array = y_test.to_numpy()

        # ====================================================
        # ONE-HOT ENCODING
        # ====================================================

        OHE = OneHotEncoder(handle_unknown='ignore')

        # Encode categorical variables in training set
        train = OHE.fit_transform(
            X_train.loc[:, ['Cirurgia', 'Sexo']]
        ).toarray()

        X_train[OHE.get_feature_names_out()] = train

        # Remove original categorical columns
        X_train.drop(
            ['Cirurgia', 'Sexo'],
            axis='columns',
            inplace=True
        )

        featuresColumns = X_train.columns

        # Encode categorical variables in test set
        test = OHE.transform(
            X_test.loc[:, ['Cirurgia', 'Sexo']]
        ).toarray()

        X_test[OHE.get_feature_names_out()] = test

        X_test.drop(
            ['Cirurgia', 'Sexo'],
            axis='columns',
            inplace=True
        )

        del train

        try:

            # Keep only selected features
            X_train = X_train[list(featuresNamesDF)]
            X_test = X_test[list(featuresNamesDF)]

            # =================================================
            # FEATURE SELECTION
            # =================================================

            X_trainSelected, X_testSelected, cols_idxs = feature_selector(
                X_train,
                X_test,
                y_train,
                1,
                feat + 1
            )

            # Store selected feature names
            featsSelected = []

            for colIndx in range(len(cols_idxs)):
                featsSelected.append(
                    featuresColumns[cols_idxs[colIndx]]
                )

            # Fill remaining rows with NaN
            featsSelectedName = np.ones(100 - len(cols_idxs))
            featsSelectedName[:] = np.nan

            featsSelected = np.concatenate(
                (featsSelected, featsSelectedName),
                axis=0
            )

            featsSelectedDFInd[str(feat + 1)] = featsSelected

            featsSelectedDFInd.fillna('', inplace=True)

            # Save selected features
            featsSelectedDFInd.to_excel(
                path + 'Outputs/' +
                'FeaturesSelected4Target_' +
                target[i] + '.xlsx'
            )

            # =================================================
            # MODEL TRAINING
            # =================================================

            for name, regressor in regressors:

                try:

                    # Train model
                    multioutput_model = regressor

                    multioutput_model.fit(
                        X_trainSelected,
                        y_train
                    )

                    # Predictions
                    multioutput_pred = multioutput_model.predict(
                        X_testSelected
                    ).round(decimals=0).astype('int')

                    # Training predictions
                    multioutput_predTrain = multioutput_model.predict(
                        X_trainSelected
                    ).round(decimals=0).astype('int')

                    classifierName.append(name)

                    # =================================================
                    # EVALUATION METRICS
                    # =================================================

                    rec = recall_score(
                        y_test_array,
                        multioutput_pred
                    )

                    prec = precision_score(
                        y_test_array,
                        multioutput_pred
                    )

                    accTest.append(
                        accuracy_score(
                            y_test_array,
                            multioutput_pred
                        )
                    )

                    recTest.append(rec)

                    precTest.append(prec)

                    # F1-score calculation
                    f1Test.append(
                        2 * (rec * prec) / (rec + prec)
                    )

                    featnumb.append(feat + 1)

                except Exception as e:
                    print(f"Failed to run {name}: {e}")

        except Exception as e:
            print(f"Failed to run {name}: {e}")

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    df['featNumb'] = featnumb
    df['ACC'] = accTest
    df['REC'] = recTest
    df['PREC'] = precTest
    df['F1'] = f1Test

    df.to_excel(
        path + 'Outputs/' +
        'Results_' + target[i] + '.xlsx'
    )

# ============================================================
# SECOND PHASE:
# BEST FEATURE SELECTION + PERMUTATION IMPORTANCE
# ============================================================

target = [
    'AF',
    'MI',
    'Poedema',
    'Pneumonia',
    'Stroke',
    'Infection',
    'AKI',
    'Bleeding',
    '1Y Mortality',
    '1 Y Reinterventions',
    '1 Y Hospital readmissions'
]

# Reload dataset
database = pd.read_excel(
    path + 'Database2.xlsx',
    sheet_name='Sheet1'
)

database.dropna(axis='rows', inplace=True)

database.drop(
    ['Unnamed: 0', 'Nome', 'Target'],
    axis='columns',
    inplace=True
)

X = database.loc[:, 'Hg Alta':]
y = database.loc[:, target]

df = pd.DataFrame()
bestFeaturesDF = pd.DataFrame()

# ============================================================
# LOOP THROUGH EACH TARGET
# ============================================================

for i in range(len(target)):

    accTest = []
    recTest = []
    precTest = []

    # Load best feature results
    bestResults = pd.read_excel(
        path + 'Outputs/' +
        'Results_' + target[i] + '.xlsx'
    )

    # Get configuration with best accuracy
    bestACC = bestResults[
        bestResults['ACC'] == bestResults.ACC.max()
    ]

    numbOfFeat = bestACC['featNumb'].iloc[0]

    # Load selected feature names
    featuresNamesDF = pd.read_excel(
        path + 'Outputs/' +
        'FeaturesSelected4Target_' +
        target[i] + '.xlsx'
    )

    featuresNamesDF = featuresNamesDF[str(numbOfFeat)]

    featuresNamesDF2 = pd.read_excel(
        path + 'Outputs/' +
        'FeaturesSelected4Target_' +
        target[i] + '.xlsx'
    )

    featuresNamesDF2 = featuresNamesDF2[str(numbOfFeat)]

    featuresNamesDF.dropna(axis=0, inplace=True)

    # ========================================================
    # TEST MULTIPLE RANDOM STATES
    # ========================================================

    for randVal in range(18, 23):

        # Dataset split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y[target[i]],
            test_size=0.2,
            stratify=y[target[i]],
            random_state=randVal
        )

        y_test_array = y_test.to_numpy()

        # ====================================================
        # ONE-HOT ENCODING
        # ====================================================

        OHE = OneHotEncoder(handle_unknown='ignore')

        train = OHE.fit_transform(
            X_train.loc[:, ['Cirurgia', 'Sexo']]
        ).toarray()

        X_train[OHE.get_feature_names_out()] = train

        X_train.drop(
            ['Cirurgia', 'Sexo'],
            axis='columns',
            inplace=True
        )

        featuresColumns = X_train.columns

        test = OHE.transform(
            X_test.loc[:, ['Cirurgia', 'Sexo']]
        ).toarray()

        X_test[OHE.get_feature_names_out()] = test

        X_test.drop(
            ['Cirurgia', 'Sexo'],
            axis='columns',
            inplace=True
        )

        del train

        try:

            # Keep selected features only
            X_train = X_train[list(featuresNamesDF)]
            X_test = X_test[list(featuresNamesDF)]

            X_trainSelected = X_train

            # DataFrame for permutation importance
            PermImportance = pd.DataFrame()

            PermImportance['Features'] = list(featuresNamesDF)

            # =================================================
            # TRAIN MODEL
            # =================================================

            for name, regressor in regressors:

                try:

                    multioutput_model = regressor

                    multioutput_model.fit(
                        X_trainSelected,
                        y_train
                    )

                    X_testSelected = X_test

                    # =============================================
                    # PERMUTATION IMPORTANCE
                    # =============================================

                    importancesTotal = permutation_importance(
                        multioutput_model,
                        X_trainSelected,
                        y_train,
                        n_repeats=10,
                        random_state=0
                    )

                    importances = (
                        importancesTotal.importances_mean
                    )

                    PermImportance['Mean'] = importances

                    importancesMean = pd.Series(
                        importancesTotal.importances_mean,
                        index=list(featuresNamesDF)
                    )

                    importances = (
                        importancesTotal.importances_std
                    )

                    PermImportance['Std'] = importances

                    # =============================================
                    # MODEL PREDICTIONS
                    # =============================================

                    multioutput_pred = multioutput_model.predict(
                        X_testSelected
                    ).round(decimals=0).astype('int')

                    multioutput_predTrain = multioutput_model.predict(
                        X_trainSelected
                    ).round(decimals=0).astype('int')

                    classifierName.append(name)

                    # =============================================
                    # STORE METRICS
                    # =============================================

                    accTest.append(
                        accuracy_score(
                            y_test_array,
                            multioutput_pred
                        )
                    )

                    recTest.append(
                        recall_score(
                            y_test_array,
                            multioutput_pred
                        )
                    )

                    precTest.append(
                        precision_score(
                            y_test_array,
                            multioutput_pred
                        )
                    )

                except Exception as e:
                    print(f"Failed to run {name}: {e}")

        except Exception as e:
            print(f"Failed to run {name}: {e}")

        # ====================================================
        # PLOT FEATURE IMPORTANCE
        # ====================================================

        fig, ax = plt.subplots()

        importancesMean.plot.bar(
            yerr=importancesTotal.importances_std,
            ax=ax
        )

        ax.set_title("Feature importances using permutation")
        ax.set_ylabel("Mean accuracy decrease")

        fig.tight_layout()

        plt.savefig(
            path + 'Outputs/' +
            'permutationImportanceRandomState' +
            str(randVal) + '.pdf',
            bbox_inches='tight'
        )

        plt.show()

        # Save permutation importance results
        PermImportance.to_excel(
            path + 'Outputs/' +
            'PermImportanceRandomState' +
            str(randVal) + '.xlsx'
        )

    # ========================================================
    # FINAL RESULTS STORAGE
    # ========================================================

    df['FeatNumb_' + target[i]] = [numbOfFeat]

    df['ACC_' + target[i]] = [np.mean(accTest)]

    df['REC_' + target[i]] = [np.mean(recTest)]

    df['PREC_' + target[i]] = [np.mean(precTest)]

    # Final F1-score
    df['F1_' + target[i]] = [
        2 * (
            np.mean(recTest) * np.mean(precTest)
        ) / (
            np.mean(recTest) + np.mean(precTest)
        )
    ]

    bestFeaturesDF[target[i]] = featuresNamesDF2

# ============================================================
# SAVE FINAL OUTPUT FILES
# ============================================================

df.to_excel(
    path + 'Outputs/' +
    'ResultsMonoOutput.xlsx'
)

bestFeaturesDF.to_excel(
    path + 'Outputs/' +
    'FeaturesNamesMonoOutput.xlsx'
)