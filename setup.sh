name: Deploy to Streamlit

on:
  push:
    branches:
      - main

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
    - name: Check out repository
      uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'

    - name: Upgrade pip and install dependencies
      run: ./setup.sh

    - name: Deploy to Streamlit
      run: streamlit run app.py