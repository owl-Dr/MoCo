# MoCo: Movie Companion

MoCo is a movie recommendation system that leverages machine learning models to help users discover movies based on their preferences. This project includes data preprocessing, feature engineering, model training, and a web interface built using Streamlit.

[Watch the Project Demo Video](https://youtu.be/vYVbFgTkoJo?si=YeC3X9lAqETTNPhc)

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/your-username/MoCo.git
   cd MoCo
   ```

2. **Install the required packages:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Prepare your dataset by following the steps in the [Dataset Preparation](#dataset-preparation) section.**

## Usage

1. **Run the preprocessing and model training scripts:**
   - Open and run `data_preparation.ipynb` in the `Preprocessing` folder to generate `data.csv`.
   - Open and run `model_preparation.ipynb` in the `ModelPreparation` folder to generate `models_and_data.pkl`.

2. **Run the Streamlit application:**
   ```sh
   streamlit run app.py
   ```

## Project Structure

```
MoCo/
├── Preprocessing/
│   └── data_preparation.ipynb   # Script for preparing the dataset
├── ModelPreparation/
│   └── model_preparation.ipynb  # Script for training models and saving data
├── app.py                       # Streamlit application script
├── requirements.txt             # List of dependencies
└── README.md                    # Project README file
```

## Dataset Preparation

The dataset used for this project is derived from the TMDB 5000 Movie Dataset available on Kaggle. Follow these steps to prepare the `data.csv` file:

1. **Download the dataset** from [Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).

2. **Run the `data_preparation.ipynb` notebook** in the `Preprocessing` folder. This notebook will merge and preprocess the datasets to generate `data.csv`.

## Models and Features

### Data Preprocessing

- **Handle Missing Values**: Fill missing values in numerical columns with the mean and in text columns with empty strings.
- **Stemming**: Apply stemming to the 'overview' column using NLTK's `PorterStemmer`.
- **Normalization**: Normalize numeric columns using `MinMaxScaler`.
- **Encoding**: 
  - Encode genres using `MultiLabelBinarizer`.
  - Vectorize 'cast' and 'crew' using `CountVectorizer`.
  - Apply TF-IDF vectorization to the 'overview' column.

### Model Training

Run the `model_preparation.ipynb` notebook in the `ModelPreparation` folder. This notebook will train the models and save the data and models to `models_and_data.pkl`.

### Models

- **Singular Value Decomposition (SVD)**: Used for dimensionality reduction of combined features.
- **K-Nearest Neighbors (KNN)**: Used for finding similar movies based on cosine similarity.
- **Random Forest Regressor**: Used for predicting movie similarity.

### Saving Models

- Save dictionaries and trained models using `pickle`.

## Streamlit App

The Streamlit app provides an interactive interface for users to get movie recommendations.

### Features

- **Movie Search**: Search for a movie from the dataset.
- **Recommendations**: Get movie recommendations based on the selected movie.
- **Watchlist**: Add recommended movies to a watchlist.

### Usage

- **Select a Movie**: Choose a movie from the dropdown.
- **Get Recommendations**: Click the 'Get Recommendations' button.
- **View Watchlist**: The watchlist is displayed in the sidebar.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request.

---

**Note:** MoCo uses the TMDB API to fetch additional movie details like posters, genres, and overviews. The TMDB API may not work in India due to regional restrictions; if you face issues, please use a VPN to access the API.
