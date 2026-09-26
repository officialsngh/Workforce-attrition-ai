# HR Attrition API — production-style container
# Build: docker build -t hr-attrition-api .
# Run:   docker run -p 8000:8000 -v $(pwd)/output:/app/output hr-attrition-api

FROM python:3.11-slim

WORKDIR /app

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY . .
# Model must be present in output/ (mount volume or copy after training)
ENV MODEL_PATH=/app/output/attrition_model.pkl

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
