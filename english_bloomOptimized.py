import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib

nltk.download('stopwords')
nltk.download('wordnet')

# 1. Load and inspect data
df = pd.read_csv("a_bl.csv")
print("Dataset Preview:")
print(df.head())
print("\nClass Distribution:")
print(df['Category'].value_counts())

# 2. Map BT levels to Bloom's taxonomy
bloom_map = {
    'BT1': 'Remember',
    'BT2': 'Understand',
    'BT3': 'Apply',
    'BT4': 'Analyze',
    'BT5': 'Evaluate',
    'BT6': 'Create'
}
df['Bloom_Level'] = df['Category'].map(bloom_map)

# 3. Text preprocessing
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = ''.join([char for char in text if char.isalnum() or char == ' '])
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words and len(word) > 2]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

df['processed_text'] = df['Questions'].apply(preprocess_text)

# 4. Prepare features and labels
tfidf = TfidfVectorizer(
    max_features=2000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.85
)
X = tfidf.fit_transform(df['processed_text'])
y = df['Category']  # Using original BT labels

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

# 6. Handle class imbalance
print("\nClass distribution before SMOTE:", y_train.value_counts())
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print("Class distribution after SMOTE:", pd.Series(y_train_smote).value_counts())

# 7. Train Random Forest
rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=20,
    min_samples_split=5,
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)
print("\nTraining Random Forest...")
rf.fit(X_train_smote, y_train_smote)

# 8. Evaluate
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.3f}")

# 9. Set style (FIXED: using valid style name)
plt.style.use('seaborn-v0_8-darkgrid')

# Confusion Matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[f"BT{i+1}" for i in range(6)],
            yticklabels=[f"BT{i+1}" for i in range(6)])
plt.title(f'Random Forest Confusion Matrix\nAccuracy: {accuracy:.3f}', fontsize=14)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('rf_confusion_matrix.png', dpi=300)
plt.show()

# Feature Importance
feature_importances = pd.DataFrame({
    'feature': tfidf.get_feature_names_out(),
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False).head(20)

plt.figure(figsize=(12, 6))
sns.barplot(x='importance', y='feature', data=feature_importances, palette='viridis')
plt.title('Top 20 Important Features for Classification')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.show()

# Word Clouds by BT Level
plt.figure(figsize=(18, 12))
for i, (bt, level) in enumerate(bloom_map.items()):
    plt.subplot(2, 3, i+1)
    text = ' '.join(df[df['Category'] == bt]['processed_text'])
    wordcloud = WordCloud(
        width=600, 
        height=400, 
        background_color='white',
        colormap='plasma',
        max_words=50
    ).generate(text)
    plt.imshow(wordcloud)
    plt.title(f'{bt}: {level}', fontsize=12)
    plt.axis('off')
plt.suptitle("Most Frequent Words by Bloom's Level", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('wordclouds.png', dpi=300, bbox_inches='tight')
plt.show()

# 10. Save model and artifacts
joblib.dump(rf, 'random_forest_bloom_classifier.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
print("\nSaved model as 'random_forest_bloom_classifier.pkl'")

# 11. Classification Report
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=[f"BT{i+1}" for i in range(6)]))
