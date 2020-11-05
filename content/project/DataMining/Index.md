---
title: Data Mining
summary: This project is regarding building a model to predict used car prices for my graduate course. We took the dataset from kaggle and worked on it to build a effective model for predicting the prices.

{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# FALL 2020 -> CSE 5334 -> Data Mining\n",
    "## Assignment 2\n",
    "### Name:  Tirumala Manukonda (UTA ID# 1001662386)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Assignment is to learn about the kNN.\n",
    "\n",
    "__Summary__:\n",
    "\n",
    "-  This assignment is about implementation of KNN algorithm with the hyper parameters K values =1,3,5,7 and distance metrics = euclidean distance , euclidean distance and cosine distance\n",
    "-  Split the data of IRIS into development and test set\n",
    "-  Find the optimal parameters using development data set and plot them using barchart for the given hyperparameters\n",
    "-  Analyse the graph and get optimal hyperparameters, and from the hyperparameters getting the Final accuracy of test-set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 246,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import math\n",
    "from collections import Counter\n",
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 247,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Id</th>\n",
       "      <th>SepalLengthCm</th>\n",
       "      <th>SepalWidthCm</th>\n",
       "      <th>PetalLengthCm</th>\n",
       "      <th>PetalWidthCm</th>\n",
       "      <th>Species</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>5.1</td>\n",
       "      <td>3.5</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>Iris-setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>4.9</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>Iris-setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>4.7</td>\n",
       "      <td>3.2</td>\n",
       "      <td>1.3</td>\n",
       "      <td>0.2</td>\n",
       "      <td>Iris-setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>4.6</td>\n",
       "      <td>3.1</td>\n",
       "      <td>1.5</td>\n",
       "      <td>0.2</td>\n",
       "      <td>Iris-setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>5.0</td>\n",
       "      <td>3.6</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>Iris-setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>6</td>\n",
       "      <td>5.4</td>\n",
       "      <td>3.9</td>\n",
       "      <td>1.7</td>\n",
       "      <td>0.4</td>\n",
       "      <td>Iris-setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>7</td>\n",
       "      <td>4.6</td>\n",
       "      <td>3.4</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.3</td>\n",
       "      <td>Iris-setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>8</td>\n",
       "      <td>5.0</td>\n",
       "      <td>3.4</td>\n",
       "      <td>1.5</td>\n",
       "      <td>0.2</td>\n",
       "      <td>Iris-setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>9</td>\n",
       "      <td>4.4</td>\n",
       "      <td>2.9</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>Iris-setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>10</td>\n",
       "      <td>4.9</td>\n",
       "      <td>3.1</td>\n",
       "      <td>1.5</td>\n",
       "      <td>0.1</td>\n",
       "      <td>Iris-setosa</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Id  SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm      Species\n",
       "0   1            5.1           3.5            1.4           0.2  Iris-setosa\n",
       "1   2            4.9           3.0            1.4           0.2  Iris-setosa\n",
       "2   3            4.7           3.2            1.3           0.2  Iris-setosa\n",
       "3   4            4.6           3.1            1.5           0.2  Iris-setosa\n",
       "4   5            5.0           3.6            1.4           0.2  Iris-setosa\n",
       "5   6            5.4           3.9            1.7           0.4  Iris-setosa\n",
       "6   7            4.6           3.4            1.4           0.3  Iris-setosa\n",
       "7   8            5.0           3.4            1.5           0.2  Iris-setosa\n",
       "8   9            4.4           2.9            1.4           0.2  Iris-setosa\n",
       "9  10            4.9           3.1            1.5           0.1  Iris-setosa"
      ]
     },
     "execution_count": 247,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Reading the csv file from the local folder and displaying top 10 values\n",
    "\n",
    "df = pd.read_csv(\"C:/Users/Dell/OneDrive/Documents/Iris.csv\",sep=',')\n",
    "df.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 248,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<seaborn.axisgrid.FacetGrid at 0x1b92541d288>"
      ]
     },
     "execution_count": 248,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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6NCh+vzzz/XnP/9ZklRQUKAZM2YoMjLS47hFRUX6n//5H+3du1clJSV69NFHde2116pPnz769a9/re3bt8vlcunll19WXFycpkyZoq+//lqNGzfW/v37NXfuXC1cuFAFBQXmTHJz587VDz/8oNOnT2vWrFk6ePCgmjdvrl/+8pfmuI8//rhKSkqUk5OjMWPGqEWLFsrJyVHv3r313Xff6dtvv9VNN92ksWPH+uiMA3ASEgOEnISEBM2bN0/Lli3T3LlzFRMTozFjxujVV1/VM888o0svvVSrV6/WK6+8ouuvv16HDx9WRkaGSkpK1KdPH91222367rvvNHPmTDVr1kzz58/X2rVr1adPH4/jrl69Wo0aNdIzzzyj3Nxc3XPPPcrMzFR+fr569+6tSZMm6bHHHtPHH3+s6OhoHT9+XOnp6Tp27JiSkpIUFham+++/X9nZ2erRo4cWL16s7t27684779Sf/vQnrV27Vi1atNBFF11Ubtzo6Gjz37///nu99tprKigoUI8ePfTxxx8rNjZWN998M4kBAEkkBghBe/fuVVxcnKZPny6pdNnb+++/XwUFBZoyZYqk0r/dt2nTRpLM5XslqV27dtq3b5+aNWump59+WvXq1dPhw4fVpUuXasfdsWOHNm7cqC1btkiSiouLlZubK0m64oorJEktWrRQYWGh9u/fr06dOkkqTWQSExMrPebZ9QsaN26sH374QRdeeKHef//9cvvk5uZq8+bNateunS666CI1aNBAUVFRaty4sbk2wtkVFQGAxAAhZ/v27VqxYoXmz5+v6OhotWnTRg0aNFCzZs00Y8YMXXjhhdq4caO50mNWVpbcbrfOnDmjnTt36uKLL9aoUaP0z3/+U3FxcRo3bpyszCyemJio5s2b64EHHlBBQYHmzZunhg0bSqr4xdyuXTu98847kqQTJ05oz549kqSwsDCVlJRUOUanTp2Uk5OjLVu2qGPHjjIMQ3/+858VHR2tdu3akQAAqBaJAUJOUlKSdu3apeTkZNWrV0+GYeiJJ55Q8+bNNW7cOHOVuKefflpHjhxRcXGxRo4cqePHj2vUqFFKSEjQnXfeqQEDBuiCCy5Q48aNdeTIkQrjjB492rzTcO2112rMmDFKTU3VPffco7y8PA0ZMkRhYZXX/9500036+OOPNWjQIDVu3FgxMTGKjIzUZZddpnnz5lW50l5YWJhmz56tqVOn6vTp0/rpp5/UqVMnPfroo5XGCADnYxElwIPPPvtMK1eu1IsvvujXcXft2qVt27apd+/eys3N1R133KGPPvrITDQAwFe4YwAEoRYtWuj555/XkiVL5Ha79cc//pGkAIBfcMcAAACYmOAIAACYSAwAAICJxAAAAJhIDAAAgInEAAAAmP4/GisR4Bj67ukAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 527.875x432 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#To visualize the dataset and the labels init wrt to features SepalLengthCm,PetalLengthCm\n",
    "\n",
    "import seaborn as sns\n",
    "sns.set_style(\"whitegrid\") \n",
    "sns.FacetGrid(df, hue =\"Species\",  \n",
    "              height = 6).map(plt.scatter,  \n",
    "                              'SepalLengthCm',  \n",
    "                              'PetalLengthCm').add_legend() \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 249,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 150 entries, 0 to 149\n",
      "Data columns (total 6 columns):\n",
      " #   Column         Non-Null Count  Dtype  \n",
      "---  ------         --------------  -----  \n",
      " 0   Id             150 non-null    int64  \n",
      " 1   SepalLengthCm  150 non-null    float64\n",
      " 2   SepalWidthCm   150 non-null    float64\n",
      " 3   PetalLengthCm  150 non-null    float64\n",
      " 4   PetalWidthCm   150 non-null    float64\n",
      " 5   Species        150 non-null    object \n",
      "dtypes: float64(4), int64(1), object(1)\n",
      "memory usage: 7.2+ KB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 250,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>count</th>\n",
       "      <th>mean</th>\n",
       "      <th>std</th>\n",
       "      <th>min</th>\n",
       "      <th>25%</th>\n",
       "      <th>50%</th>\n",
       "      <th>75%</th>\n",
       "      <th>max</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Id</th>\n",
       "      <td>150.0</td>\n",
       "      <td>75.500000</td>\n",
       "      <td>43.445368</td>\n",
       "      <td>1.0</td>\n",
       "      <td>38.25</td>\n",
       "      <td>75.50</td>\n",
       "      <td>112.75</td>\n",
       "      <td>150.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>SepalLengthCm</th>\n",
       "      <td>150.0</td>\n",
       "      <td>5.843333</td>\n",
       "      <td>0.828066</td>\n",
       "      <td>4.3</td>\n",
       "      <td>5.10</td>\n",
       "      <td>5.80</td>\n",
       "      <td>6.40</td>\n",
       "      <td>7.9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>SepalWidthCm</th>\n",
       "      <td>150.0</td>\n",
       "      <td>3.054000</td>\n",
       "      <td>0.433594</td>\n",
       "      <td>2.0</td>\n",
       "      <td>2.80</td>\n",
       "      <td>3.00</td>\n",
       "      <td>3.30</td>\n",
       "      <td>4.4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>PetalLengthCm</th>\n",
       "      <td>150.0</td>\n",
       "      <td>3.758667</td>\n",
       "      <td>1.764420</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.60</td>\n",
       "      <td>4.35</td>\n",
       "      <td>5.10</td>\n",
       "      <td>6.9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>PetalWidthCm</th>\n",
       "      <td>150.0</td>\n",
       "      <td>1.198667</td>\n",
       "      <td>0.763161</td>\n",
       "      <td>0.1</td>\n",
       "      <td>0.30</td>\n",
       "      <td>1.30</td>\n",
       "      <td>1.80</td>\n",
       "      <td>2.5</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               count       mean        std  min    25%    50%     75%    max\n",
       "Id             150.0  75.500000  43.445368  1.0  38.25  75.50  112.75  150.0\n",
       "SepalLengthCm  150.0   5.843333   0.828066  4.3   5.10   5.80    6.40    7.9\n",
       "SepalWidthCm   150.0   3.054000   0.433594  2.0   2.80   3.00    3.30    4.4\n",
       "PetalLengthCm  150.0   3.758667   1.764420  1.0   1.60   4.35    5.10    6.9\n",
       "PetalWidthCm   150.0   1.198667   0.763161  0.1   0.30   1.30    1.80    2.5"
      ]
     },
     "execution_count": 250,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# view summary statistics\n",
    "df.describe().T"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 251,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Species\n",
       "Iris-setosa        50\n",
       "Iris-versicolor    50\n",
       "Iris-virginica     50\n",
       "dtype: int64"
      ]
     },
     "execution_count": 251,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Let’s now take a look at the number of instances (rows) that belong to each class.\n",
    "#We can view this as an absolute count.\n",
    "df.groupby('Species').size()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 252,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Selecting features and labels arrays:\n",
    "X = df.iloc[:, 1:5].values\n",
    "y = df.iloc[:, 5].values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 253,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Label encoder to categorize the target values\n",
    "from sklearn import preprocessing\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "le = LabelEncoder()\n",
    "y = le.fit_transform(y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 254,
   "metadata": {},
   "outputs": [],
   "source": [
    "#split the dataset into development and test sets\n",
    "X_develop, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state = 100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 255,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Computes the euclidean distance\n",
    "\n",
    "def euclidean_distance(a,b):\n",
    "    distance=np.sqrt(np.dot(a-b,a-b))\n",
    "    return distance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 256,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Computes the normalized euclidean distance\n",
    "\n",
    "def normalized_euclidean_distance(a,b):\n",
    "    distance=0.5 *np.sqrt(np.dot(a-b,a-b))\n",
    "    return distance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 257,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Computes the cosine distance\n",
    "\n",
    "def cosine_distance(a,b):\n",
    "       \n",
    "    distance=1-np.dot(a,b)/np.sqrt(np.dot(a,a)*np.dot(b,b))\n",
    "    return distance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 258,
   "metadata": {},
   "outputs": [],
   "source": [
    "#KNN algorithm with the generalised metrics k and distance\n",
    "\n",
    "class KNearestNeighbors(object):\n",
    "    \n",
    "    def __init__(self,k,distance):\n",
    "        self.k=k\n",
    "        self.distance=distance\n",
    "        self.X_train=np.asarray([])\n",
    "        self.y_train=np.asarray([])\n",
    "        \n",
    "    def fit(self,X,y):\n",
    "        self.X_train=X\n",
    "        self.y_train=y\n",
    "        \n",
    "    def predict(self,X):\n",
    "        X=X.reshape((-1,self.X_train.shape[1]))\n",
    "        \n",
    "        #Creating matrix to store distance\n",
    "        distances=np.zeros((X.shape[0],self.X_train.shape[0]))\n",
    "        for i,x in enumerate(X):\n",
    "            for j,x_train in enumerate(self.X_train):\n",
    "                distances[i,j]=self.distance(x_train,x)\n",
    "        #Storing the indices of top k elements where distance is in increasing order\n",
    "        sorted_indices=distances.argsort()[:,:self.k]\n",
    "        top_k = self.y_train[sorted_indices]  #sort and take top k\n",
    "        result = np.zeros(X.shape[0])\n",
    "        for i, values in enumerate(top_k):\n",
    "            result[i] = Counter(values).most_common(1)[0][0]\n",
    "        return result"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 259,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Calculate accuracy percentage between two lists\n",
    "def accuracy_metric(actual, predicted):\n",
    "    correct= 0\n",
    "    for i in range(len(actual)):\n",
    "        if actual[i] == predicted[i]:\n",
    "            correct += 1\n",
    "    return correct / float(len(actual)) * 100.0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 260,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Fitting clasifier to the Development set\n",
    "knn = KNearestNeighbors(1, euclidean_distance)\n",
    "knn.fit(X_develop, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 261,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Predicting the Development set results\n",
    "y_develop = knn.predict(X_develop)\n",
    "y_pred = knn.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 262,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Development data set Accuracy of k =1 and d = euclidean distance:100.0%\n"
     ]
    }
   ],
   "source": [
    "#Getting the Accuracy of the Development data set\n",
    "Dev_Accuracy = accuracy_metric(y_train,y_develop)\n",
    "print('Development data set Accuracy of k =1 and d = euclidean distance:'+repr(Dev_Accuracy)+'%')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- This is the starting combination of the hyperparameters of k=1 and distance = euclidean distance, and below will check the given k values and its accuracy wrt to distance metrics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 263,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Development data set Accuracy of k = 3 and d = euclidean distance:96.19047619047619%\n"
     ]
    }
   ],
   "source": [
    "# Now calculating with hyperparameters of k=3 and distance = euclidean distance\n",
    "knn = KNearestNeighbors(3, euclidean_distance)\n",
    "knn.fit(X_develop, y_train)\n",
    "y_develop = knn.predict(X_develop)\n",
    "y_pred = knn.predict(X_test)\n",
    "Dev_Accuracy = accuracy_metric(y_train,y_develop)\n",
    "print('Development data set Accuracy of k = 3 and d = euclidean distance:'+repr(Dev_Accuracy)+'%')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 264,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Development data set Accuracy of k =5 and d = euclidean distance:98.09523809523809%\n"
     ]
    }
   ],
   "source": [
    "# Now calculating with hyperparameters of k=5 and distance = euclidean distance\n",
    "knn = KNearestNeighbors(5, euclidean_distance)\n",
    "knn.fit(X_develop, y_train)\n",
    "y_develop = knn.predict(X_develop)\n",
    "y_pred = knn.predict(X_test)\n",
    "Dev_Accuracy = accuracy_metric(y_train,y_develop)\n",
    "print('Development data set Accuracy of k =5 and d = euclidean distance:'+repr(Dev_Accuracy)+'%')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 265,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Development data set Accuracy of k =7 and d = euclidean distance:97.14285714285714%\n"
     ]
    }
   ],
   "source": [
    "# Now calculating with hyperparameters of k=7 and distance = euclidean distance\n",
    "knn = KNearestNeighbors(7, euclidean_distance)\n",
    "knn.fit(X_develop, y_train)\n",
    "y_develop = knn.predict(X_develop)\n",
    "y_pred = knn.predict(X_test)\n",
    "Dev_Accuracy = accuracy_metric(y_train,y_develop)\n",
    "print('Development data set Accuracy of k =7 and d = euclidean distance:'+repr(Dev_Accuracy)+'%')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 267,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Development data set Accuracy of k =1 and d = normalized euclidean distance:100.0%\n"
     ]
    }
   ],
   "source": [
    "# Now calculating with hyperparameters of k=1 and distance = normalized_euclidean_distance\n",
    "knn = KNearestNeighbors(1, normalized_euclidean_distance)\n",
    "knn.fit(X_develop, y_train)\n",
    "y_develop = knn.predict(X_develop)\n",
    "y_pred = knn.predict(X_test)\n",
    "Dev_Accuracy = accuracy_metric(y_train,y_develop)\n",
    "print('Development data set Accuracy of k =1 and d = normalized euclidean distance:'+repr(Dev_Accuracy)+'%')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 268,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Development data set Accuracy of k =3 and d = normalized euclidean distance:96.19047619047619%\n"
     ]
    }
   ],
   "source": [
    "# Now calculating with hyperparameters of k=3 and distance = normalized_euclidean_distance\n",
    "knn = KNearestNeighbors(3, normalized_euclidean_distance)\n",
    "knn.fit(X_develop, y_train)\n",
    "y_develop = knn.predict(X_develop)\n",
    "y_pred = knn.predict(X_test)\n",
    "Dev_Accuracy = accuracy_metric(y_train,y_develop)\n",
    "print('Development data set Accuracy of k =3 and d = normalized euclidean distance:'+repr(Dev_Accuracy)+'%')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 269,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Development data set Accuracy of k =5 and d = normalized euclidean distance:98.09523809523809%\n"
     ]
    }
   ],
   "source": [
    "# Now calculating with hyperparameters of k=5 and distance = normalized_euclidean_distance\n",
    "knn = KNearestNeighbors(5, normalized_euclidean_distance)\n",
    "knn.fit(X_develop, y_train)\n",
    "y_develop = knn.predict(X_develop)\n",
    "y_pred = knn.predict(X_test)\n",
    "Dev_Accuracy = accuracy_metric(y_train,y_develop)\n",
    "print('Development data set Accuracy of k =5 and d = normalized euclidean distance:'+repr(Dev_Accuracy)+'%')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 270,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Development data set Accuracy of k =7 and d = normalized euclidean distance:97.14285714285714%\n"
     ]
    }
   ],
   "source": [
    "# Now calculating with hyperparameters of k=7 and distance = normalized_euclidean_distance\n",
    "knn = KNearestNeighbors(7, normalized_euclidean_distance)\n",
    "knn.fit(X_develop, y_train)\n",
    "y_develop = knn.predict(X_develop)\n",
    "y_pred = knn.predict(X_test)\n",
    "Dev_Accuracy = accuracy_metric(y_train,y_develop)\n",
    "print('Development data set Accuracy of k =7 and d = normalized euclidean distance:'+repr(Dev_Accuracy)+'%')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 280,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Development data set Accuracy of k =1 and d = cosine_distance:100.0%\n"
     ]
    }
   ],
   "source": [
    "# Now calculating with hyperparameters of k=3 and distance = cosine_distance\n",
    "knn = KNearestNeighbors(1, cosine_distance)\n",
    "knn.fit(X_develop, y_train)\n",
    "y_develop = knn.predict(X_develop)\n",
    "y_pred = knn.predict(X_test)\n",
    "Dev_Accuracy = accuracy_metric(y_train,y_develop)\n",
    "print('Development data set Accuracy of k =1 and d = cosine_distance:'+repr(Dev_Accuracy)+'%')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 279,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Development data set Accuracy of k =3 and d = cosine_distance:98.09523809523809%\n"
     ]
    }
   ],
   "source": [
    "# Now calculating with hyperparameters of k=3 and distance = cosine_distance\n",
    "knn = KNearestNeighbors(3, cosine_distance)\n",
    "knn.fit(X_develop, y_train)\n",
    "y_develop = knn.predict(X_develop)\n",
    "y_pred = knn.predict(X_test)\n",
    "Dev_Accuracy = accuracy_metric(y_train,y_develop)\n",
    "print('Development data set Accuracy of k =3 and d = cosine_distance:'+repr(Dev_Accuracy)+'%')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 273,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Development data set Accuracy of k =5 and d = cosine_distance:98.09523809523809%\n"
     ]
    }
   ],
   "source": [
    "# Now calculating with hyperparameters of k=5 and distance = cosine_distance\n",
    "knn = KNearestNeighbors(5, cosine_distance)\n",
    "knn.fit(X_develop, y_train)\n",
    "y_develop = knn.predict(X_develop)\n",
    "y_pred = knn.predict(X_test)\n",
    "Dev_Accuracy = accuracy_metric(y_train,y_develop)\n",
    "print('Development data set Accuracy of k =5 and d = cosine_distance:'+repr(Dev_Accuracy)+'%')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 275,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Development data set Accuracy of k =7 and d = cosine_distance:97.14285714285714%\n"
     ]
    }
   ],
   "source": [
    "# Now calculating with hyperparameters of k=7 and distance = cosine_distance\n",
    "knn = KNearestNeighbors(7, cosine_distance)\n",
    "knn.fit(X_develop, y_train)\n",
    "y_develop = knn.predict(X_develop)\n",
    "y_pred = knn.predict(X_test)\n",
    "Dev_Accuracy = accuracy_metric(y_train,y_develop)\n",
    "print('Development data set Accuracy of k =7 and d = cosine_distance:'+repr(Dev_Accuracy)+'%')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 281,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1080x360 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "\n",
    "\n",
    "labels = ['1', '3', '5', '7']\n",
    "euclidean_accuracy = [100.0, 96.190, 98.095, 97.142]\n",
    "normalized_euclidean_accuracy = [100.0, 96.190, 98.095, 97.142]\n",
    "cosine_accuracy = [100.0, 98.095, 98.095, 97.142]\n",
    "\n",
    "x = np.arange(len(labels))  # the label locations\n",
    "width =0.2  # the width of the bars\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(15, 5))\n",
    "rects1 = ax.bar(x - width, euclidean_accuracy, width, label='euclidean_accuracy')\n",
    "rects2 = ax.bar(x, normalized_euclidean_accuracy, width, label='normalized_euclidean_accuracy')\n",
    "rects3 = ax.bar(x + width, cosine_accuracy, width, label='cosine_accuracy')\n",
    "\n",
    "# Add some text for labels, title and custom x-axis tick labels, etc.\n",
    "ax.set_ylabel('Accuracy')\n",
    "ax.set_xlabel('K')\n",
    "ax.set_title('Accuracy of KNN With the kvalues')\n",
    "ax.set_xticks(x)\n",
    "ax.set_xticklabels(labels)\n",
    "ax.legend()\n",
    "\n",
    "\n",
    "def autolabel(rects):\n",
    "    \"\"\"Attach a text label above each bar in *rects*, displaying its height.\"\"\"\n",
    "    for rect in rects:\n",
    "        height = rect.get_height()\n",
    "        ax.annotate('{}'.format(height),\n",
    "                    xy=(rect.get_x() + rect.get_width() / 2, height),\n",
    "                    xytext=(0, 3),  # 3 points vertical offset\n",
    "                    textcoords=\"offset points\",\n",
    "                    ha='center', va='bottom')\n",
    "\n",
    "\n",
    "autolabel(rects1)\n",
    "autolabel(rects2)\n",
    "autolabel(rects3)\n",
    "\n",
    "\n",
    "fig.tight_layout()\n",
    "\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Analysis :- From the above barchart, we can understand that mostly all the values are same except at k=3, \n",
    "    The optimal performance of the algorithm would be when the k= 5 because at k =1, its makes the model sensitive to noise and \n",
    "    its leads to overfitting and increasing k values makes its underfitting, ignoring k =1 as it overfits, we can say that at \n",
    "    k=5 its has good accuracy rate and cosine distance metrices has performed better than the other distance measures, We can conclude\n",
    "    that the optimal parameters are k=5 and distance = cosine distance\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 284,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Final Test data set Accuracy:100.0%\n"
     ]
    }
   ],
   "source": [
    "#calculating test accuracy with optimal hyperparameters of above observation of k=5 and distance = cosine_distance\n",
    "knn = KNearestNeighbors(5, cosine_distance)\n",
    "knn.fit(X_develop, y_train)\n",
    "y_develop = knn.predict(X_develop)\n",
    "y_pred = knn.predict(X_test)\n",
    "#Getting the Accuracy of the test dataset\n",
    "Final_Test_Accuracy = accuracy_metric(y_test,y_pred)\n",
    "print('Final Test data set Accuracy:'+repr(Final_Test_Accuracy)+'%')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Final test accuracy for the test data is 100%, which says that our performed much better on the test data \n",
    "  when k=5 and distance = cosine distance as optimal hyperparameters"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

tags:
- Data Structures 
date: "2020-04-27T00:00:00Z"

# Optional external URL for project (replaces project detail page).
external_link: ""

image:
  caption: Photo by Toa Heftiba on Unsplash
  focal_point: Smart
---

This project is regarding building a model to predict used car prices for my graduate course. We took the dataset from kaggle and worked on it to build a effective model for predicting the prices
