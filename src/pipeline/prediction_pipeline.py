import os
import sys
import pandas as pd
from src.exception.exception import customexception
from src.logger.logging import logging
from src.utils.utils import load_object
from src.components.data_transformation import DataTransformation


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model_path = os.path.join("artifacts", "model.pkl")

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            scaled_fea = preprocessor.transform(features)
            pred = model.predict(scaled_fea)

            return pred

        except Exception as e:
            raise customexception(e, sys)


class CustomData:
    def __init__(self,
                 area: float,
                 bedroom: float,
                 bathroom: float,
                 seller_type: str,
                 layout_type: str,
                 property_type: str,
                 furnish_type: str,
                 locality: str,
                 city: str):

        self.area = area
        self.bedroom = bedroom
        self.bathroom = bathroom
        self.seller_type = seller_type
        self.layout_type = layout_type
        self.property_type = property_type
        self.furnish_type = furnish_type
        self.locality = locality
        self.city = city

    def get_data_as_dataframe(self):
        try:
            data_transformation = DataTransformation()  # Instantiate DataTransformation
            data_transformation.initialize_data_transformation(train_path=os.path.join("artifacts","train.csv"),test_path=os.path.join("artifacts","test.csv"))

            encoded_user_city = data_transformation.city_encoder.transform([self.city])[0]
            locality_encoding_map = data_transformation.locality_means.to_dict()
            encoded_user_locality = locality_encoding_map.get(self.locality, None)
            custom_data_input_dict = {
                'area': [self.area],
                'bedroom': [self.bedroom],
                'bathroom': [self.bathroom],
                'seller_type': [self.seller_type],
                'layout_type': [self.layout_type],
                'property_type': [self.property_type],
                'furnish_type': [self.furnish_type],
                'locality_encoded': [encoded_user_locality],
                'city_encoded': [encoded_user_city]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occurred in prediction pipeline')
            raise customexception(e, sys)

# Rest of the code remains the same
