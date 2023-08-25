FROM python:3.7

WORKDIR /server1
COPY requirements.txt ./
COPY . ./
RUN pip3 install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y vim

COPY . .
EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001" , "--reload"]
