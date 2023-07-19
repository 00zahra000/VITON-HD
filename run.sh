
# in docker file un-comment this line:
# uvicorn api-3:app --reload --host 0.0.0.0 --port 9000

# for local run:
source venv/bin/activate
uvicorn api-3:app --reload --host 192.168.1.93 --port 8000
# gunicorn -k uvicorn.workers.UvicornWorker api-3:app -b 192.168.29.242:8000

