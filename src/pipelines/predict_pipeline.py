from typing import Dict, Any
import sys
import pickle
import pandas as pd

from src.exception import CustomException
from src.utils import get_params


class CustomData:
    def __init__(
            self,
            gender: str, 
            race_ethnicity: str, 
            parental_level_of_education: str, 
            lunch: str, 
            test_preparation_course: str, 
            reading_score: int, 
            writing_score: int        
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
        
    def convert_to_dataframe(self):
        try: 
            record = pd.DataFrame(
                {
                    "gender": [self.gender], 
                    "race_ethnicity": [self.race_ethnicity], 
                    "parental_level_of_education": [self.parental_level_of_education], 
                    "lunch": [self.lunch], 
                    "test_preparation_course": [self.test_preparation_course], 
                    "reading_score": [self.reading_score], 
                    "writing_score": [self.writing_score]
                }
            )
            return record
        except Exception as err:
            raise CustomException(err, sys)


class PredictPipeline:
    def __init__(self):
        self.params: Dict[str, Any] = get_params(r"./conf/parameters.yml")

    def predict(self, input_record: pd.DataFrame) -> float:
        try:
            ft = pickle.load(open(r"./artifacts/feature_transformer.pkl", "rb"))
            ohe_categories = []
            for array in ft.transformers_[1][1].categories_:
                ohe_categories += [
                    category.lower().replace(" ", "_").replace("/", "_")
                    for category in array.tolist()
                ]
            ft_features = self.params["numeric_features"] + ohe_categories + self.params["ordinal"]["features"]
            input_record = pd.DataFrame(
                ft.transform(input_record),
                columns=ft_features,
                index=input_record.index.tolist()
            )
            model = pickle.load(open(r"./artifacts/model.pkl", "rb"))
            mrmr_features = model.feature_names_in_.tolist()
            prediction = model.predict(input_record[mrmr_features])[0]
            return prediction
        except Exception as err:
            raise CustomException(err, sys)
