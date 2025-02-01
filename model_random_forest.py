import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm  

# Step 1: Load your Excel file
file_path = '8000_Блоом_түвшин_датасэт.xlsx'  # Replace with your actual file path
df = pd.read_excel(file_path)

# Step 2: Inspect your dataset
print(df.head())  # Ensure your dataset has 'Question' and 'Label' columns

# Step 3: Preprocess the 'Label' column (Bloom's Taxonomy Levels)
label_encoder = LabelEncoder()
df['Label_Encoded'] = label_encoder.fit_transform(df['Category'])  # Encode categorical labels

# Step 4: Convert questions into numeric features using TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=500)  # Limiting to 500 features for simplicity
X = vectorizer.fit_transform(df['Questions']).toarray()  # Convert questions to numeric vectors

# Step 5: Define your features (X) and target (y)
y = df['Label_Encoded']  # Target variable (Bloom's Taxonomy level)

# Step 6: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train a classification model (Random Forest in this case)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Adding progress bar to monitor the training process
# Wrapping the model fitting process with tqdm
print("Training the model...")
for _ in tqdm(range(1), desc="Training Progress", total=1):
    model.fit(X_train, y_train)  # Train the model

# Step 8: Make predictions
y_pred = model.predict(X_test)

# Step 9: Evaluate the model's performance (accuracy for classification)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Step 10: Function to predict Bloom's level for a new question
def predict_bloom_level(question):
    new_question_vectorized = vectorizer.transform([question]).toarray()  # Convert the input question to numeric form
    predicted_label = model.predict(new_question_vectorized)  # Predict the label
    predicted_bloom_level = label_encoder.inverse_transform(predicted_label)  # Convert numeric label back to Bloom's level
    return predicted_bloom_level[0]  # Return the predicted Bloom's level

# Step 11: Continuously take input from the console
while True:
    question = input("Enter a question (or 'exit' to quit): ")
    if question.lower() == 'exit':
        break
    bloom_level = predict_bloom_level(question)
    print(f"The predicted Bloom's level for the question is: {bloom_level}")