# P7_project
P7_project of Data Scientist OC formation.

Data are organized by order of execution:
- 1: feature engineering and cleaning
- 2: best model selection
- 3: model storage
- 4: each confusion matrix generated from the models
- 5: explicability of the best model
- 6: API (using FastAPI)
- 7: Streamlit dashboard

The API to generate predictions from the best model and the dashboard were locally deployed using Docker: the dockerfiles are kept here, including the docker-compose yaml file.

The API and dashboard were deployed on the cloud using Heroku. Here are their corresponding links:
- API: https://fastapi-backend-p7.herokuapp.com/
- Dashboard: https://home-credit-prediction-p7.herokuapp.com/
