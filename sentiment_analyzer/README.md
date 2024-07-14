# Sentiment Analyzer

This project is a sentiment analysis application that utilizes deep learning techniques to classify Twitter sentiments as Positive, Neutral, or Negative. The application is built using Flask for the web interface and employs Neural Network for text classification.

## Table of Contents

- [Installation](#installation)
- [Folder Structure](#folder-structure)
- [Usage](#usage)
- [Model Training](#model-training)
- [Web Application](#web-application)
- [License](#license)

## Installation
 
This has been provided in the projects README.md file

### Prerequisites

Make sure you have Python 3.10 installed. You will also need the following libraries:


```bash
    pip install flask numpy pandas scikit-learn tensorflow nltk joblib
```

Install the required packages. reqirements.txt has the libraries with the version used when creating this project

```bash
    pip install -r requirements.txt
```
Setup NLTK Resources

You may need to download the NLTK stopwords:

```python
    import nltk
    nltk.download('stopwords')
```

## Folder Structure

```bash
    tree    // to see the folder structure in tree format
    .
    ├── README.md
    └── sentiment_analyzer
        ├── README.md
        ├── data
        │   ├── preprocessed_data.csv
        │   ├── twitter_training.csv
        │   └── twitter_validation.csv
        ├── model
        │   ├── label_encoder.pkl
        │   ├── sentana.h5
        │   ├── sentiment_analysis_model.h5
        │   └── tfidf_vectorizer.pkl
        ├── requirements.txt
        └── src
            ├── app.py
            ├── sentana.py
            ├── static
            │   └── style.css
            └── templates
                └── index.html

    7 directories, 14 files
```
## Usage

### Running the Web Application

To start the web application, navigate to the src directory and run:

```bash
python app.py
```
Visit http://127.0.0.1:5000/ in your web browser to access the application.

### Input Text

You can enter any tweet or text in the input box, and the application will output the predicted sentiment along with the confidence score.

## Model Training

### Dataset

Other datasets can be used to get different results.

The model is trained using the sentana.py script. This script preprocesses the data, trains the sentiment analysis model, and saves the trained model and related artifacts. To train the model, run:

```bash
python sentana.py
```
This will create a preprocessed dataset and train the model using the training data.

## Web Application

The web interface is built using Flask and serves as the frontend for user interaction. The HTML template is located in the templates folder, and the styling is managed through the CSS file in the static folder.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

Feel free to modify any sections to better match your project specifics or preferences!
