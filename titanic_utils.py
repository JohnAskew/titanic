def clean_data(data):
    data["Fare"] = data["Fare"].fillna(data["Fare"].dropna().median())
    data["Age"]  = data["Age"].fillna(data["Age"]).dropna().median()

    data.loc[data["Sex"] == "male", "Sex"] = 0
    data.loc[data["Sex"] == "female", "Sex"] =1

    data["Embarked"] = data["Embarked"].fillna("S")
    data.loc[data["Embarked"] == "S", "Embarked"] = 0
    data.loc[data["Embarked"] == "C", "Embarked"] = 1
    data.loc[data["Embarked"] == "Q", "Embarked"] = 2

    data.loc[data["Pclass"] ==3, "Pclass"] == int(3)
    data.loc[data["Pclass"] ==2, "Pclass"] == int(2)
    data.loc[data["Pclass"] ==2, "Pclass"] == int(1)