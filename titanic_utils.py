#-------------------------------------#
def clean_data(data):
#-------------------------------------#
    data["fare"] = data["fare"].fillna(data["fare"].dropna().median())
    data["age"]  = data["age"].fillna(data["age"]).dropna().median()
    #data['age'].fillna(data.groupby("title")["age"].transform("median"), inplace = True)

    data.loc[data["sex"] == "male", "sex"] = 0
    data.loc[data["sex"] == "female", "sex"] =1
    #map_gender(data)

    data["embarked"] = data["embarked"].fillna("S")
    data.loc[data["embarked"] == "S", "embarked"] = 0
    data.loc[data["embarked"] == "C", "embarked"] = 1
    data.loc[data["embarked"] == "Q", "embarked"] = 2

    data.loc[data["pclass"] ==3, "pclass"] == int(3)
    data.loc[data["pclass"] ==2, "pclass"] == int(2)
    data.loc[data["pclass"] ==2, "pclass"] == int(1)

#-------------------------------------#
def map_title(data):
#-------------------------------------#
    data['title'] = data['name'].str.extract(' ([A-Za-z]+)\.', expand = False)
    print(f'Title value counts :\n',data['title'].value_counts())
    title_mapping  = {"Mr":0
                    , "Miss":1
                    , "Mrs":2
                    , "Master":3, "Dr":3, "Rev":3, "Col":3, "Major":3, "Mlle": 3, "Countess":3
                    , "Ms":3, "Lady":3, "Jonkheer":3, "Don":3, "Dona":3, "Mme":3, "Capt":3, "Sir":3
                    ,
                  }
    data['title'] = data['title'].map(title_mapping)

#-------------------------------------#
def map_gender(data):
#-------------------------------------#
    sex_mapping = {"male":0, "female":1}
    data['sex'] = data['sex'].map(sex_mapping)

#-------------------------------------#
def map_fare(data):
#-------------------------------------#
    data['fare'].fillna(data.groupby('pclass')['fare'].transform("median"), inplace = True)

#-------------------------------------#
def map_fare2(data):
#-------------------------------------#
    data.loc[data['fare']  <= 17, 'fare'] = 0,
    data.loc[ (data['fare'] > 17) & (data['fare'] <= 30), 'fare'] = 1,
    data.loc[ (data['fare'] > 30) & (data['fare'] <= 100), 'fare'] = 2,
    data.loc[data['fare']   > 100, 'fare'] = 3,

#-------------------------------------#
def map_embarked(data):
#-------------------------------------#
    embark_mapping = {"S": 0, "C":1, "Q":2}
    data['embarked'] = data['embarked'].map(embark_mapping)
    data['embarked'] = data['embarked'].fillna(0)

#-------------------------------------#
def map_cabin(data):
#-------------------------------------#
    data['cabin'] = data['cabin'].str[:1]

#-------------------------------------#
def map_cabin2(data):
#-------------------------------------#
    cabin_mapping = {"A":0, "B":0.4, "C":0.8, "D":1.2, "E":1.6,"F":2,"G":2.4,"T":2.8}
    data['cabin'] = data['cabin'].map(cabin_mapping)
    data['cabin'].fillna(data.groupby('pclass')['cabin'].transform("median"), inplace = True)

#-------------------------------------#
def map_familysize(data):
#-------------------------------------#
    data['familysize'] = data['sibsp'] + data['parch'] +1

#-------------------------------------#
def map_familysize2(data):
#-------------------------------------#
    family_mapping = {1:0, 2:0.4, 3:0.8, 4:1.2, 5:1.6, 6:2, 7:2.4, 8:2.8, 9:3.2, 10:3.6, 11:4}
    data['familysize'] = data['familysize'].map(family_mapping)

    

