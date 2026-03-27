import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

MODEL_FILE = "model.joblib"
PIPELINE_FILE = "pipeline.joblib"

#separate function to make pipeline
def buildpipeline(nums_attribs, cat_attribs):
    
    #numerical pipeline
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy = "median")),
        ("scaler", StandardScaler())
    ])

    #categorical pipeline
    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, nums_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])

    return full_pipeline

if not os.path.exists(MODEL_FILE):
    #create file
    housing = pd.read_csv("housing.csv")
    housing['income_cat'] = pd.cut(housing["median_income"], #Median income is highly correlated with median_house_value. Hence chosen for stratification
                                   bins = [0.0,1.5,3.0,4.5,6.0,np.inf],
                                   labels=[1,2,3,4,5])
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing['income_cat']):
        housing.loc[test_index].drop("income_cat", axis=1).to_csv("input.csv", index = False)
        housing = housing.loc[train_index].drop("income_cat", axis = 1)
    
    housing_labels = housing["median_house_value"].copy()
    housing_features = housing.drop("median_house_value", axis = 1)

    nums_attribs = housing_features.select_dtypes(include = ['number']).columns.tolist()
    cat_attribs = housing_features.select_dtypes(exclude = ['number']).columns.tolist()

    pipeline = buildpipeline(nums_attribs,cat_attribs) #Pipeline object created
    housing_prepared = pipeline.fit_transform(housing_features) #Data is fitted and tranform using .fit_transform

    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared, housing_labels)

    #Save model and pipeline
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)

    print("Model trained and saved")


else:
    # INFERENCE PHASE
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv("input.csv")
    transformed_input = pipeline.transform(input_data)
    predictions = model.predict(transformed_input)
    input_data["median_house_value"] = predictions

    input_data.to_csv("output.csv", index = False)
    print("Inference complete. Results saved to output.csv")


