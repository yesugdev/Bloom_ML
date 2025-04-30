import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# 1. Load your dataset
df = pd.read_csv("blooms_taxonomy_dataset.csv")  # Make sure the file is in your working directory
print(df.head())
print("\nClass distribution:")
print(df["Bloom's Taxonomy Level"].value_counts())

# 2. Text preprocessing
def preprocess_text(text):
    # Handle NaN values
    if isinstance(text, float):
        return ""
    # Lowercase
    text = text.lower()
    # Remove special characters
    text = ''.join([char for char in text if char.isalnum() or char == ' '])
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

df['processed_question'] = df['Exam Questions'].apply(preprocess_text)

# 3. Encode labels
le = LabelEncoder()
df['bloom_encoded'] = le.fit_transform(df["Bloom's Taxonomy Level"])

# 4. TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=1500)
X = tfidf.fit_transform(df['processed_question']).toarray()
y = df['bloom_encoded']

# 5. Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Handle class imbalance
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# 7. Initialize models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": SVC(kernel='linear', C=1, probability=True, random_state=42),
    "Naive Bayes": MultinomialNB(alpha=0.1)
}

# 8. Train and evaluate
results = {}
for name, model in models.items():
    model.fit(X_train_smote, y_train_smote)
    y_pred = model.predict(X_test)
    
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

# 9. Visualizations
plt.style.use('ggplot')

# 9.1 Accuracy comparison
plt.figure(figsize=(10, 5))
model_names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in model_names]
plt.bar(model_names, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
plt.title("Model Accuracy Comparison", fontsize=14)
plt.xlabel("Model", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.ylim(0, 1.1)
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.03, f"{acc:.3f}", ha='center', fontsize=11)
plt.tight_layout()
plt.show()

# 9.2 Confusion matrix for best model
best_model = max(results, key=lambda x: results[x]['accuracy'])
cm = results[best_model]['confusion_matrix']
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title(f'Confusion Matrix ({best_model})', fontsize=14)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

# 9.3 Word clouds by Bloom's level
plt.figure(figsize=(15, 10))
for i, level in enumerate(le.classes_):
    plt.subplot(2, 3, i+1)
    text = ' '.join(df[df["Bloom's Taxonomy Level"] == level]['processed_question'])
    wordcloud = WordCloud(width=400, height=250, background_color='white',
                          colormap='viridis').generate(text)
    plt.imshow(wordcloud)
    plt.title(f'{level}', fontsize=12)
    plt.axis('off')
plt.suptitle("Most Frequent Words by Bloom's Level", fontsize=16)
plt.tight_layout()
plt.show()

# 10. Detailed report
print(f"\n{'='*50}\nBest Performing Model: {best_model}\n{'='*50}")
print(f"Accuracy: {results[best_model]['accuracy']:.3f}")
print("\nDetailed Classification Report:")
report_df = pd.DataFrame(results[best_model]['report']).transpose()
print(report_df)

# Feature importance (for Random Forest)
if best_model == "Random Forest":
    feature_names = tfidf.get_feature_names_out()
    importances = models[best_model].feature_importances_
    top_features = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(20)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=top_features, palette='viridis')
    plt.title('Top 20 Important Features (Random Forest)')
    plt.tight_layout()
    plt.show()
