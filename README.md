### Create another container to manage another model in another API.

Create a new server1 folder inside the root project containing the version 2 model, which will be the second container
Create a Dockerfile in this server1 folder with the following configuration

    ```bash
    FROM python:3.7

    WORKDIR /server1
    COPY requirements.txt ./
    COPY . ./
    RUN pip3 install --no-cache-dir -r requirements.txt
    RUN apt-get update && apt-get install -y vim

    COPY . .
    EXPOSE 8001
    CMD "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001" , "--reload"  
    ```


    #Update the docker-compose.yml file by adding information from the second server.

 ```bash
version: "3.7"

services:
  server:
    build: server
    ports:
      - 8000:8000
    networks:
      AIservice:
        aliases:
          - server.docker

  server1:
    build: server1
    ports:
      - 8001:8001
    networks:
      AIservice:
        aliases:
          - server1.docker

  frontend:
    build: app
    ports:
      - 3000:3000
    networks:
      AIservice:
        aliases:
          - frontend.docker
    depends_on:
      - server
      - server1

networks:
  AIservice:
    external: true   
 ```


#Run the following command to create the Docker Compose images

 ```bash
docker-compose -f docker-compose.yml up --build
```

As a result, the following image will be displayed:

