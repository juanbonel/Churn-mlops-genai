FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY knowledge ./knowledge
COPY models/churn_model ./models/churn_model
COPY models/model_metadata.json ./models/model_metadata.json
RUN pip install --no-cache-dir .
EXPOSE 8000
CMD ["uvicorn", "churn_project.api:app", "--host", "0.0.0.0", "--port", "8000"]
