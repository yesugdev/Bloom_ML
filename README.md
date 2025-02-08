# 🎯 Сэтгэл хөдлөл ба Блүмийн ангилалын таамаглал API

## 📝 Төслийн тухай
Энэхүү төсөл нь дараах 2 үндсэн үйлдлийг гүйцэтгэнэ:
- Боловсролын асуултуудын Блүмийн түвшинг таамаглах (1-6)
- Текстэн дэх сэтгэл хөдлөлийг шинжлэх (уйтгар(0), баяр(1), хайр(2), уур(3), айдас(4))

## ⚙️ Суулгах заавар
```bash
# Төслийг татах
git clone https://github.com/yesugen/emotion-bloom-api.git
cd emotion-bloom-api

# Шаардлагатай сангуудыг суулгах
pip install -r requirements.txt

# Өгөгдлийн сан үүсгэх
python manage.py migrate

# Серверийг ажиллуулах
python manage.py runserver
```

🚀 Боломжууд
✨ REST API үйлчилгээ
📊 Excel макро интеграци
📦 Багц боловсруулалт
💾 Excel-рүү экспортлох
🔌 API Үйлчилгээ
Блүмийн ангилал
```bash
curl -X POST http://localhost:8000/api/predict/ \
-H "Content-Type: application/json" \
-d '{"question": "Монгол улсын нийслэл хот аль вэ?"}'
```

```bash
curl -X POST http://localhost:8000/api/emotion/ \
-H "Content-Type: application/json" \
-d '{"text": "Би өнөөдөр их баяртай байна"}'
```
📈 Excel интеграци
Excel програм нээх
Alt + F11 товч дарж VBA editor нээх
macro_for_excel.txt файлыг импортлох
A баганад текст оруулах
Макрог ажиллуулах
🛠️ Ашигласан технологиуд
Python 3.13+
Django 4.0+
scikit-learn
pandas
Microsoft Excel
📋 Шаардлагатай сангууд

👨‍💻 Зохиогч
Yesug Bekhbold
📄 Лиценз
MIT License