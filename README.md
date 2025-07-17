# Spam Shield

Type your text to know whether it is spam or not. The project uses a custom preprocessor function and TF-IDF vectorier. The model is trained using Naive Bayes classifier on [SMS Spam Collection dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) provided by UCI.

To run the program:
1. Run the python file train.py
```
python3 scripts/train.py
```

2. Run the streamlit app
```
streamlit run app.py
```