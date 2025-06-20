# from ..src.pipeline.api_model_pipeline import APIModelPipeline

from src.pipeline.api_model_pipeline import APIModelPipeline
from datasets import load_dataset


model_name= 'meta-llama/llama-4-maverick:free'



data = load_dataset('tmnam20/ViMedAQA','drug',split= 'train')
data = data.select(list(range(10)))



data_reasoning = []

for item in data:
    question = item['question']
    anwser = item['answer']
    context = item['context']

    item = {'question': f'{question}. Hãy dựa vào context là {context} để đưa ra câu trả lời',
            'anwser': anwser}
    data_reasoning.append(item)

api_pipeline = APIModelPipeline(model_name_or_path= model_name, data_list_json=data_reasoning, num_save_chunk=5, client_url='https://openrouter.ai/api/v1')
api_pipeline.generate_synthetic_data(save_dir_path='data_gen', push_hf_id='ChaosAiVision/Medical_reasoning')

