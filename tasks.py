from celery import Celery

# 'localhost'의 6379 포트에 뜬 Redis를 브로커로 사용
app = Celery('my_tasks', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')

@app.task
def add(x, y):
    return x + y
