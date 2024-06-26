FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . /app/
EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "simple-chat-bot.py", "--server.port=8501", "--server.address=0.0.0.0"]
