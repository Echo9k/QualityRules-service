import pandas as pd
import os

# Define the directory structure
base_dir = 'data/'
output_dir = os.path.join(base_dir, 'output/')
os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

# Step 1: Standardize Description
standardize_description_data = {
    "standard_description": [
        "FIELD MUST BE UNIQUE",
        "FIELD CONTAIN A VALUE, NAS OR NULL NOT ALLOWED",
        "FIELD BE NOT NEGATIVE",
        "FIELD MUST HAVE A LENGTH EQUAL TO 7",
        "FIELD MUST HAVE A GREATER THAN 7 AND SMALLER OR EQUAL TO 12",
        "FIELD VALUE MUST BE GREATER THAN ZERO",
        "FIELD NOT CONTAIN SEQUENCES OF MORE THAN 3 NUMBERS",
        "FIELD NOT CONTAIN MORE THAN 5 REPEATED CHARACTERS"
    ],
    "user_description": [
        "The ID must be unique",
        "The ID must not be empty",
        "The ID must not be negative",
        "The ID must have a length of 7 characters",
        "The ID must be between 7 and 12 characters",
        "The field should have a length of 7 characters",
        "No more than 3 consecutive numbers are allowed",
        "The value of the field should not have more than 5 repeated characters"
    ]
}

# Create a DataFrame for standard descriptions
df_standard = pd.DataFrame(standardize_description_data)

# Save the standard descriptions dataset
df_standard.to_csv(os.path.join(base_dir, 'standard_descriptions.csv'), index=False)

# Step 2: Identify rule data
identify_rule_data = {
    "Users Description": [
        "Should contain a value, no nulls are allowed",
        "This value must not be empty",
        "The value should be at least zero and no more than 99",
        "Value should be no more than 1000",
        "Value must be positive",
        "Must have a length of 7 characters",
        "No more than 3 consecutive numbers allowed",
        "No more than 5 repeated characters"
    ],
    "Rule ID": [1, 1, 4, 3, 3, 5, 6, 7],
    "Dimension": [
        "Completeness", "Completeness", "Validity", "Validity",
        "Validity", "Validity", "Validity", "Validity"
    ],
    "Standard Description": [
        "FIELD CONTAIN A VALUE, NAS OR NULL NOT ALLOWED",
        "FIELD CONTAIN A VALUE, NAS OR NULL NOT ALLOWED",
        "FIELD VALUE MUST BE GREATER OR EQUAL THAN 0 AND LOWER THAN 100",
        "FIELD BE LOWER OR EQUAL TO 1000",
        "FIELD BE GREATER THAN ZERO",
        "FIELD MUST HAVE A LENGTH EQUAL TO 7",
        "FIELD NOT CONTAIN SEQUENCES OF MORE THAN 3 NUMBERS",
        "FIELD NOT CONTAIN MORE THAN 5 REPEATED CHARACTERS"
    ]
}

# Create a DataFrame for rule data
df_rules = pd.DataFrame(identify_rule_data)

# Save the user descriptions dataset
df_rules.to_csv(os.path.join(base_dir, 'historical_rules.csv'), index=False)


# Step 3: Create user descriptions data
user_descriptions = {
    "Users description": [
        "The ID must be unique",
        "The ID must not be empty",
        "The ID must not be negative",
        "The ID must have a length of 7 characters",
        "The ID must be between 7 and 12 characters",
        "Field a length of 7 characters",
        "No more than 3 consecutive numbers are allowed",
        "The value of the field should not have more than 5 repeated characters",
        "Should contain a value, no nulls are allowed",
        "This value must not be empty",
        "The value should be at least zero and no more than 99",
        "Value should be no more than 1000",
        "Value should be no more than 0"
    ]
}

# Create a DataFrame for user descriptions
df_user_descriptions = pd.DataFrame(user_descriptions)

# Save the user descriptions dataset
df_user_descriptions.to_csv(os.path.join(base_dir, 'user_descriptions.csv'), index=False)

# print("User descriptions CSV file created successfully!")

# Step 4: Fill in parameters (manually defined based on the rules)
ensemble_data = {
    "CDE_ID": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "Source_ID": ["Table_A", "Table_B", "Table_B", "Table_B", "Table_C", "Table_B", "Table_C", "Table_A", "Table_B"],
    "CDE": ["user_ID", "Tenure", "Tenure", "Age", "Monthly_income", "Tenure", "Account_Currency", "Customer_ID", "Product_Code"],
    "Reference": [
        "{}", "{}", "{min:0, max:100, operator1:'≥', operator2:'≤'}", "{value:0, operator:'>'}",
        "{value:1000, operator:'≤'}", "{min:0}", "{size:3}", "{seq_max:3}", "{repeat_max:5}"
    ],
    "Rule ID": [1, 1, 4, 3, 3, 6, 5, 6, 7],
    "Dimension": [
        "Completeness", "Completeness", "Validity", "Validity", "Validity",
        "Validity", "Validity", "Validity", "Validity"
    ],
    "Description": [
        "Should contain a value", "This value must not be empty", "The value should be at least zero and no more than 99",
        "Value must be positive", "Value should be at most 1000", "Value should be a number [0-9], no strings nor symbols are allowed",
        "Must have a length equal to 3", "No more than 3 consecutive numbers allowed", "No more than 5 repeated characters"
    ],
    "Standard Description": [
        "FIELD CONTAIN A VALUE, NAS OR NULL NOT ALLOWED",
        "FIELD CONTAIN A VALUE, NAS OR NULL NOT ALLOWED",
        "FIELD VALUE MUST BE GREATER OR EQUAL THAN 0 AND LOWER THAN 100",
        "FIELD BE GREATER THAN ZERO",
        "FIELD BE LOWER OR EQUAL TO 1000",
        "FIELD MUST BE NUMERIC INTEGER",
        "FIELD MUST HAVE A LENGTH EQUAL TO 3",
        "FIELD NOT CONTAIN SEQUENCES OF MORE THAN 3 NUMBERS",
        "FIELD NOT CONTAIN MORE THAN 5 REPEATED CHARACTERS"
    ]
}

# Create a final DataFrame for ensemble
df_final = pd.DataFrame(ensemble_data)

# Save the final consolidated output table
df_final.to_csv(os.path.join(output_dir, 'final_output.csv'), index=False)

print("CSV files created successfully in the specified directory structure!")
