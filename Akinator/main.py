import pandas as pd
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder


#Algorithm to predict the best questions to ask
def getQuestion(currentData, targetColumn):
    attributes = ['nationality', 'club', 'preferred_foot', 'team_position']
    bestAttribute = None
    maxInformation = 0

    for attribute in attributes:
        if attribute in currentData.columns and len(currentData[attribute].unique()) > 1:
            informationGain = calculateInformation(currentData, attribute, targetColumn)
            if informationGain > maxInformation:
                maxInformation = informationGain
                bestAttribute = attribute

    return bestAttribute


#Entropy to ask question
def calculateInformation(data, attribute, targetColumn):
    entropyBeforeSplit = calculateEntropy(data[targetColumn])

    uniqueValues = data[attribute].unique()
    entropyAfterSplit = 0

    for value in uniqueValues:
        subset = data[data[attribute] == value]
        weight = len(subset) / len(data)
        entropyAfterSplit += weight * calculateEntropy(subset[targetColumn])

    informationGain = entropyBeforeSplit - entropyAfterSplit
    return informationGain


def calculateEntropy(column):
    classCount = column.value_counts()
    totalInstances = len(column)

    entropy = -sum((count / totalInstances) * math.log2(count / totalInstances) for count in classCount)
    return entropy

def trainAkinatorModel(data, targetColumn, chosenColumns):
    # Encode categorical columns
    dataEncoded = encodeCategoricalColumns(data, chosenColumns, targetColumn)

    # Split the data into features and target
    X = dataEncoded[chosenColumns]
    y = dataEncoded[targetColumn]

    # Train a Decision Tree classifier
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)

    return model

def akinatorGame(data, model, chosenColumns):
    currentData = data.copy()
    targetColumn = 'short_name'


    for questionNumber in range(1,21):
        question = getQuestion(currentData, targetColumn)

        print(f"Question {questionNumber}: Is the player from {currentData[question].iloc[0]}?")
        response = input("Your response (yes/no/probably yes/probably no/i dont know): ").lower()

        while response not in ['yes', 'no', 'probably yes', 'probably no', 'i dont know']:
            print("Invalid response. Please enter 'yes', 'no', 'probably yes', 'probably no', or 'i dont know'.")
            response = input("Your response (yes/no/probably yes/probably no/i dont know): ").lower()

        if response == "i dont know":
            response = 'no'
            print("Skipping to the next question:")

        currentData = filterDataOnResponses(currentData, question, response)

        if currentData is not None and len(currentData) == 1:
            print(f"I guess {currentData[targetColumn].iloc[0]}!")
            correctGuess = input("Is my guess correct? (yes/no): ").lower()

            while correctGuess not in ['yes', 'no']:
                print("Invalid response. Please enter 'yes' or 'no'.")
                correctGuess = input("Is my guess correct? (yes/no): ").lower()

            if correctGuess == 'yes':
                print("Yay! I guessed it! Thanks for playing!")
            break


    else:
        print("I couldn't guess your player! Sorry")



def filterDataOnResponses (data, question, response):
    if data is not None:

        if response == 'yes':
            return data[data[question] == data[question].iloc[0]]
        elif response == 'no':
            return data[data[question] != data[question].iloc[0]]
        elif response == 'probably yes':
            return data[data[question] == data[question].iloc[0]]
        elif response == 'probably no':
            return data[data[question] != data[question].iloc[0]]
        elif response == "i dont know":
            return data[data[question] != data[question].iloc[0]]


        else:
            return None


def makePrediction(model, data, chosenColumns, targetColumn):
    if data is not None:
        dataEncoded = encodeCategoricalColumns(data, chosenColumns, targetColumn)
        instance = dataEncoded[chosenColumns].iloc[0].values.reshape(1, -1)
        prediction = model.predict(instance)[0]
        return prediction

    else:
        return None

def encodeCategoricalColumns(data, columns, targetColumn):
    dataEncoded = data.copy()
    labelEncoder = LabelEncoder()

    for column in columns + [targetColumn]:
        if dataEncoded[column].dtype == 'object':
            dataEncoded[column] = labelEncoder.fit_transform(dataEncoded[column])

    return dataEncoded

df = pd.read_csv('Footballers.csv')

chosenColumnsAkinator = ['nationality', 'club', 'preferred_foot', 'team_position']

akinatorModel = trainAkinatorModel(df, 'short_name', chosenColumnsAkinator)

akinatorGame(df, akinatorModel, chosenColumnsAkinator)






