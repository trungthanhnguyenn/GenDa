from ..base_model import BaseModel
from ..model_template import APIModelTemplate
from openai import OpenAI


class OpenRouter_Model(BaseModel):

    def __init__(self, model_name_or_path:str, client_url:str, api_key:str):

        super().__init__(model_name_or_path)
        self.client_url = client_url
        self.api_key = api_key
        self.template_processor = APIModelTemplate(model_name= self.model_name_or_path)

        try:
            self.client_model = OpenAI(base_url = self.client_url, api_key = self.api_key )
            print(f"Sucessfully init client for model {self.model_name_or_path}")
        except Exception as e:
            print(f'Got error {e} when init model')
            quit()


    def generate(self, inputs):

        completion = self.client_model.chat.completions.create(
        extra_headers={
            "HTTP-Referer": "<YOUR_SITE_URL>", 
            "X-Title": "<YOUR_SITE_NAME>",
        },
        extra_body={},
        model="meta-llama/llama-4-maverick:free",
        messages = self.template_processor.apply_conversasion(question= inputs['question'], answer= inputs['anwser'])
        )


        return (completion.choices[0].message.content)













