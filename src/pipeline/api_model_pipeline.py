from ..model.api_model.api_model import OpenRouter_Model
from .base_pipeline import BasePipeline
import os


from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("OpenRouterKey")


class APIModelPipeline(BasePipeline):

    def __init__(self, model_name_or_path:str, data_list_json, num_save_chunk:int , client_url:str, api_key:str= API_KEY):
        self.model_name_or_path= model_name_or_path
        self.data_list_json = data_list_json
        self.num_save_chunk = num_save_chunk
        self.model_pipeline = OpenRouter_Model(model_name_or_path= self.model_name_or_path, api_key= api_key, client_url= client_url)




