#recipe or instruction sheet to convert your repo into dockerfile
#start with these as the base
FROM python:3.11-slim   
#this tell where to built the container like users app
WORKDIR /app

#  this copy everythinh from ur current directory to that app directory
COPY . .

#his tells Docker: "While building the image, execute this command." so it run pip install -r requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# this tells docker to open port 8000
EXPOSE 8000

#this tells Docker : when someone run this container
CMD ["python", "-m", "uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000"]