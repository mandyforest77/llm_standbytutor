FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# 💡 파일 이름을 app.py로 정확하게 맞춰줍니다!
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

