# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/navinpuri2203/superkart/Superkart.csv"
superkart_df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# ----------------------------
# Define the target variable
# ----------------------------
target = 'ProdTaken'   # 1 if the customer purchased the package, else 0

# ----------------------------
# List of numerical features
# ----------------------------
numeric_features = [

    'ProductWeight',                 #Weight of each product
    'ProductMRP',                    #Maximum retail price of each product
    'StoreEstablishmentYear',       #Year in which the store was established
    'StoreSize',                     #Size of the store, depending on sq. feet, like high, medium, and low
    'ProductStoreSalesTotal'
]

# ----------------------------
# List of categorical features
# ----------------------------
categorical_features = [
    'ProductId',                     #Unique identifier of each product, each identifier having two letters at the beginning, followed by a number
    'ProductSugarContent',          #Sugar content of each product, like low sugar, regular, and no sugar
    'ProductAllocatedArea',         #Ratio of the allocated display area of each product to the total display area of all the products in a store
    'ProductType',                   #Broad category for each product like meat, snack foods, hard drinks, dairy, canned, soft drinks, health and hygiene, baking goods, bread, breakfast, frozen foods, fruits and vegetables, household, seafood, starchy foods, others
    'StoreId',                       #Unique identifier of each store
    'StoreLocationCityType',       #Type of city in which the store is located, like Tier 1, Tier 2, and Tier 3. Tier 1 consists of cities where the standard of living is comparatively higher than that of its Tier 2 and Tier 3 counterparts
    'StoreType',                     #Type of store depending on the products that are being sold there, like Departmental Store, Supermarket Type 1, Supermarket Type 2, and Food Mart

]


# ----------------------------
# Combine features to form X (feature matrix)
# ----------------------------
X = superkart_df[numeric_features + categorical_features]

# ----------------------------
# Define target vector y
# ----------------------------
y = superkart_df[target]

# ----------------------------
# Split dataset into training and test sets
# ----------------------------
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="navinpuri2203/superkart",
        repo_type="dataset",
    )
