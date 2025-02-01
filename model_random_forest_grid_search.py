import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm 

# 1-р алхам: Excel файлыг ачааллах
file_path = '8000_Блоом_түвшин_датасэт.xlsx'  # Өөрийн файлын замыг оруулна уу
df = pd.read_excel(file_path)

# 2-р алхам: Өгөгдлийг шалгах
print(df.head())  # Өгөгдлийн 'Question' болон 'Label' багана байгаа эсэхийг шалгах

# 3-р алхам: 'Label' баганыг боловсруулж (Bloom-ийн ангилал),
# тэмдэгтэн (categorical) утгуудыг тоон (numerical) утгад хувиргах
label_encoder = LabelEncoder()
df['Label_Encoded'] = label_encoder.fit_transform(df['Category'])  # Ангиллын утгуудыг кодлох

# 4-р алхам: Асуултуудыг тоон өгөгдөл болгон хувиргахын тулд TF-IDF векторчлол ашиглах
vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))  # 1-gram болон 2-gram үгийн хэлбэрийг ашиглах
X = vectorizer.fit_transform(df['Questions']).toarray()  # Асуултуудыг тоон вектор болгон хөрвүүлэх

# 5-р алхам: Шинж чанар (X) болон зорилтот хувьсагч (y)-г тодорхойлох
y = df['Label_Encoded']  # Зорилтот хувьсагч (Bloom-ийн ангилал)

# 6-р алхам: Өгөгдлийг сургалтын (train) болон шалгалтын (test) хэсэгт хуваах
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7-р алхам: Загвар болон hyperparameter тохиргооны боломжит утгуудыг тодорхойлох
model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],  # Ойн модны (RandomForest) модны тоо
    'max_depth': [None, 10, 20, 30],  # Модны гүний хязгаар
    'min_samples_split': [2, 5, 10],  # Салбарлах хамгийн бага өгөгдлийн тоо
    'min_samples_leaf': [1, 2, 4]  # Навч дахь хамгийн бага өгөгдлийн тоо
}

# 8-р алхам: Сүлжээ хайлт (Grid Search) ба кросс баталгаажуулалт (Cross-validation) хийх
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
print("Сүлжээ хайлт хийгдэж байна...")
grid_search.fit(X_train, y_train)

# 9-р алхам: Шилдэг загварыг сонгох
best_model = grid_search.best_estimator_

# 10-р алхам: Сорилтын өгөгдөл дээр таамаглал гаргах
y_pred = best_model.predict(X_test)

# 11-р алхам: Загварын гүйцэтгэлийг үнэлэх (ангиллын нарийвчлал)
accuracy = accuracy_score(y_test, y_pred)
print(f"Нарийвчлал: {accuracy}")

# 12-р алхам: Шинэ асуултад Bloom-ийн түвшин таамаглах функц
def predict_bloom_level(question):
    new_question_vectorized = vectorizer.transform([question]).toarray()  # Шинэ асуултыг тоон өгөгдөл болгон хөрвүүлэх
    predicted_label = best_model.predict(new_question_vectorized)  # Bloom-ийн түвшинг таамаглах
    predicted_bloom_level = label_encoder.inverse_transform(predicted_label)  # Тоон утгийг анхны ангилал руу хөрвүүлэх
    return predicted_bloom_level[0]  # Bloom-ийн түвшинг буцаах

# 13-р алхам: Хэрэглэгчээс тасралтгүй асуулт авах
while True:
    question = input("Асуултаа оруулна уу (гарахын тулд 'exit' гэж бичнэ үү): ")
    if question.lower() == 'exit':
        break
    bloom_level = predict_bloom_level(question)
    print(f"Асуултын Bloom-ийн таамагласан түвшин: {bloom_level}")
