from abc import ABC, abstractmethod
from typing import List
import os
import json

from datasets import load_dataset, Dataset, DatasetDict

from ..model.base_model import BaseModel
import time
from tqdm.auto import tqdm

class BasePipeline(ABC):

    def __init__(self, model_name_or_path:str,data_list_json, num_save_chunk):

        self.model_name_or_path= model_name_or_path
        self.data_list_json = data_list_json
        self.num_save_chunk = num_save_chunk
        self.model_pipeline = BaseModel(model_name_or_path= self.model_name_or_path)

    def load_chunk_from_json(self, save_dir_path, chunk_idx):
        chunk_file = os.path.join(save_dir_path, f"synthetic_chunk_{chunk_idx}.json")
        if os.path.exists(chunk_file):
            with open(chunk_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def save_chunk_to_json(self, data_chunk, save_dir_path, chunk_idx):
        chunk_file = os.path.join(save_dir_path, f"synthetic_chunk_{chunk_idx}.json")
        with open(chunk_file, "w", encoding="utf-8") as f:
            json.dump(data_chunk, f, ensure_ascii=False, indent=2)

    # def generate_synthetic_data(self, save_dir_path: str, push_hf_id:str):
    #     if not os.path.exists(save_dir_path):
    #         os.makedirs(save_dir_path, exist_ok=True)

    #     chunk_idx = 0
    #     item_idx = 0
    #     total_items = len(self.data_list_json)
    #     all_data = []

    #     while item_idx < total_items:
    #         chunk_file = os.path.join(save_dir_path, f"synthetic_chunk_{chunk_idx}.json")
    #         if os.path.exists(chunk_file):
    #             chunk_data = self.load_chunk_from_json(save_dir_path, chunk_idx)
    #             all_data.extend(chunk_data)
    #             item_idx += self.num_save_chunk
    #             chunk_idx += 1
    #             continue

    #         data_synthetic_list = []
    #         for _ in range(self.num_save_chunk):
    #             if item_idx >= total_items:
    #                 break
    #             item = self.data_list_json[item_idx]
    #             cot_output = self.model_pipeline.generate(inputs=item)
    #             time.sleep(5)

    #             if len(cot_output) >= 200:
    #                 item['chain_of_though'] = cot_output
    #                 data_synthetic_list.append(item)
    #             item_idx += 1

    #         if data_synthetic_list:
    #             self.save_chunk_to_json(data_synthetic_list, save_dir_path, chunk_idx)
    #             all_data.extend(data_synthetic_list)
    #         chunk_idx += 1
        
    #     self.save_chunk_to_json(data_chunk= all_data, save_dir_path= save_dir_path, chunk_idx= 0000000)
    #     if push_hf_id is not None:
    #         merged_dataset = Dataset.from_list(all_data)
    #         data_dict = DatasetDict({
    #         "train": merged_dataset
    #     })

    #         data_dict.push_to_hub(repo_id=push_hf_id, max_shard_size='500MB')


    def generate_synthetic_data(self, save_dir_path: str, push_hf_id: str = None):
            if not os.path.exists(save_dir_path):
                os.makedirs(save_dir_path, exist_ok=True)

            total_items = len(self.data_list_json)
            all_data = []
            chunk_idx = 0

            for start_idx in tqdm(range(0, total_items, self.num_save_chunk), desc="Generating synthetic data"):
                chunk_file = os.path.join(save_dir_path, f"synthetic_chunk_{chunk_idx}.json")
                if os.path.exists(chunk_file):
                    chunk_data = self.load_chunk_from_json(save_dir_path, chunk_idx)
                    all_data.extend(chunk_data)
                    chunk_idx += 1
                    continue

                data_synthetic_list = []
                for item_idx in range(start_idx, min(start_idx + self.num_save_chunk, total_items)):
                    item = self.data_list_json[item_idx]
                    cot_output = self.model_pipeline.generate(inputs=item)
                    time.sleep(5)
                    if len(cot_output) >= 200:
                        item['chain_of_though'] = cot_output
                        data_synthetic_list.append(item)

                if data_synthetic_list:
                    self.save_chunk_to_json(data_synthetic_list, save_dir_path, chunk_idx)
                    all_data.extend(data_synthetic_list)
                chunk_idx += 1

            # Save all merged data
            self.save_chunk_to_json(data_chunk=all_data, save_dir_path=save_dir_path, chunk_idx=0000)
            if push_hf_id is not None:
                merged_dataset = Dataset.from_list(all_data)
                data_dict = DatasetDict({
                    "train": merged_dataset
                })
                data_dict.push_to_hub(repo_id=push_hf_id, max_shard_size='500MB')


        

        