import os
import re
import zipfile
from huggingface_hub import snapshot_download
import pandas as pd
import glob  
from tqdm import tqdm  
from vlmeval.smp import *
from ..video_base import VideoBaseDataset

from .utils import (
    extract_number_from_prediction,
    extract_option_from_prediction,
    calculate_metric_score_with_relative_error,
    calculate_metric_score_with_relative_error_consider_zero,
)

class OpenBench(VideoBaseDataset):
    TYPE = 'VQA'
    DEFAULT_REPO_ID = "HarmlessSR07/OpenBench"   

    def __init__(self, data_path, dataset='OpenBench', nframe=32, fps=-1, download=False, **kwargs):
        self.data_path = data_path
        self.dataset_name = dataset
        self.download = download
        self.repo_id = kwargs.get('repo_id', self.DEFAULT_REPO_ID)
        
        self.force_download = kwargs.get('force_download', False)
        self.force_unzip = kwargs.get('force_unzip', False)

        ret = self.prepare_dataset()
        self.data_root = ret['root']      
        self.data_file = ret['data_file'] 

        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Data file not found at {self.data_file}. Did you set download=True?")
            
        self.data = pd.read_parquet(self.data_file)
        if 'video_id' in self.data.columns and 'video' not in self.data.columns:
            self.data.rename(columns={'video_id': 'video'}, inplace=True)

        videos = list(set(self.data['video']))
        videos.sort()
        self.videos = videos
        
        self.pack = kwargs.get('pack', False) 
        self.nframe = nframe
        self.fps = fps
        if self.fps > 0 and self.nframe > 0:
            raise ValueError('fps and nframe should not be set at the same time')
        if self.fps <= 0 and self.nframe <= 0:
            raise ValueError('fps and nframe should be set at least one valid value')
        
        lmu_root = LMUDataRoot()
        self.frame_root = os.path.join(lmu_root, 'images', self.dataset_name)
        os.makedirs(self.frame_root, exist_ok=True)
        self.frame_tmpl = 'frame-{}-of-{}.jpg'
        self.frame_tmpl_fps = 'frame-{}-of-{}-{}fps.jpg'

    @classmethod
    def supported_datasets(cls):
        return ['OpenBench']

    def prepare_dataset(self):
        parquet_path = os.path.join(self.data_path, 'data.parquet')
        video_root = os.path.join(self.data_path)

        if os.path.exists(parquet_path) and os.path.exists(video_root) and not (self.download or self.force_download):
            return {'root': video_root, 'data_file': parquet_path}

        if not self.download and not os.path.exists(parquet_path):
             raise FileNotFoundError(
                f"Dataset not found at {self.data_path}. Set `download=True` in config to download."
            )

        print(f"[OpenBench] Syncing data from {self.repo_id}...")

        download_path = snapshot_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                local_dir=self.data_path,
                allow_patterns=["*.parquet", "*.zip"], 
                force_download=self.force_download,
                tqdm_class=tqdm
            )
        
        zip_files = glob.glob(os.path.join(download_path, "*.zip"))
        
        if len(zip_files) > 0:
            for zip_file in tqdm(zip_files, desc="Processing Zips"):
                marker_file = zip_file + ".extracted"
                if self.force_unzip or not os.path.exists(marker_file):
                    self._unzip_single_file(zip_file, video_root)
                    with open(marker_file, 'w') as f: f.write("done")
                else:
                    pass

        return {'root': video_root, 'data_file': parquet_path}

    def _unzip_single_file(self, zip_file, target_dir):

        try:
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(target_dir)
        except Exception as e:
            print(f"Error unzipping {zip_file}: {e}")
            raise e

    def build_prompt(self, line, video_llm, **kwargs):

        if isinstance(line, int):
            line = self.data.iloc[line]
        

        frame_paths = self.save_video_frames(line['video'])
        question_text = line['question']
        question_category = line.get('category', 'unknown') 

        prompt_text = ""
        
        # Preamble text, common to all prompts
        preamble_num_tagged = (
            "These are frames of a video.\n"
            "In the video, objects are identified by numeric tags shown nearby.\n"
            "With that in mind, please answer the following question based on the video."
        )

        # NA prompt
        if question_category in ["absolute_distance", "relative_direction_angular", "trajectory_length"]:
            instruction = "Your answer must be only the final numeric value, without units or any other text."
            prompt_text = f"{preamble_num_tagged}\nQuestion: {question_text}\n\n{instruction}\n"

        # NA prompt that needs video length(and time)
        # The video reader packages video length along with the frames, so no need to give extra information in the text prompt.
        elif question_category in ["absolute_speed", "absolute_displacement", "object_3d_localization", "depth_aware_counting"]:
            video_length = round(line.get('video_length', 0), 2)
            preamble_length = (
                # f"This video is {video_length} seconds long.\nYou will be provided with {self.nframe} separate frames uniformly sampled from a video, the frames are provided in chronological order of the video."
            )   # We found that mentioning fps confuses the model, as video-llm has its own way of transferring video metadata in its loader
            instruction = "Your answer must be only the final numeric value, without units or any other text."
            prompt_text = f"{preamble_num_tagged}\n{preamble_length}\nQuestion: {question_text}\n\n{instruction}"

        # MCQ prompt
        elif question_category in ["relative_distance", "relative_direction_categorical","relative_direction_categorical_cardinal","relative_direction_categorical_ordinal"]:
            instruction = "Your answer must be only the single letter (e.g., A, B, C, or D) of the correct option."
            
            options = line.get('options', [])
            options_text = "\n".join(options)
            prompt_text = f"{preamble_num_tagged}\nQuestion: {question_text}\n{options_text}\n\n{instruction}"

        # Qualitative Ego-Motion does not need numerical tags, so the prompt is a bit different.    
        elif question_category == "trajectory_description":
            instruction = "Your answer must be only the single letter (e.g., A, B, C, or D) of the correct option."
            options = line.get('options', [])
            options_text = "\n".join(options)
            prompt_text = f"Question: {question_text}\n{options_text}\n\n{instruction}"

        
        ### below branches for ablation study ###
#         elif question_category in ["dist_abla_origin", "dist_abla_obj1", "dist_abla_obj2", "dist_abla_pose", "dist_abla_all"]:
#             formula = ""
#             instruction = ( "To solve this, apply the following formula: $Distance = || (R \\cdot p_2 + T) - p_1 ||$. \n"
# "In this formula, $p_1$ is the 3D position in the camera coordinate of first queried object observed at the earlier time $t_1$, and $p_2$ is the 3D position of second queried object observed in the camera coordinate at the later time $t_2$. The matrix R and vector T represent the rotation and translation the camera pose has changed at time $t_2$ in related to time $t_1$. \n"
# "Let's think step by step. If any piece of information required to use the formula is not present in the text, you must infer it from the video and then use it in the formula. \n"
# "Give the final numeric value answer at the end of your output.")
#             prompt_text = f"{preamble_num_tagged}\nQuestion: {question_text}\n\n{instruction}\n"

#         elif question_category in ["dist_abla_all_woformula"]:
#             formula = ""
#             instruction = ("Note that the matrix R and vector T represent the rotation and translation the camera pose has changed at time $t_2$ in related to time $t_1$. \n"
# "Let's think step by step.\n"
# "Give the final numeric value answer at the end of your output.")
#             prompt_text = f"{preamble_num_tagged}\nQuestion: {question_text}\n\n{instruction}\n"
        
#         elif question_category in ["speed_abla_origin", "speed_abla_obj1", "speed_abla_obj2", "speed_abla_pose", "speed_abla_all"]:
#             instruction = ( "To solve this, apply the following formula: $Speed = \frac{|| (R \\cdot p_2 + T) - p_1 ||}{t_2 - t_1}$. \n"
# "In this formula, $p_1$ is the 3D position in the camera coordinate of the queried object observed at the start time $t_1$, and $p_2$ is the 3D position of the same queried object observed in the camera coordinate at the later time $t_2$. The matrix R and vector T represent the rotation and translation the camera pose has changed at time $t_2$ in related to time $t_1$. \n"
# "Let's think step by step. If any piece of information required to use the formula is not present in the text, you must infer it from the video and then use it in the formula. \n"
# "Give the final numeric value answer at the end of your output.")
#             prompt_text = f"{preamble_num_tagged}\nQuestion: {question_text}\n\n{instruction}\n"

#         elif question_category in ["speed_abla_all_woformula"]:
#             formula = ""
#             instruction = ("Note that the matrix R and vector T represent the rotation and translation the camera pose has changed at time $t_2$ in related to time $t_1$. \n"
# "Let's think step by step.\n"
# "Give the final numeric value answer at the end of your output.")
#             prompt_text = f"{preamble_num_tagged}\nQuestion: {question_text}\n\n{instruction}\n"

        else:
            # Fallback for unknown categories, uses the old generic prompt
            print(f"Warning: Unknown question category '{question_category}'. Using a generic prompt.")
            instruction = "Your answer must be only the final numeric value, without units or any other text."
            prompt_text = f"{preamble_num_tagged}\nQuestion: {question_text}\n\n{instruction}\n"

        prompt_text = prompt_text + "The answer is:"
        msgs = [dict(type='text', value=prompt_text)]
        
        if video_llm:
            video_path = os.path.join(self.data_root, line['video'] + '.mp4')
            if os.path.exists(video_path):
                msgs.append(dict(type='video', value=video_path))
            else:
                print(f"Warning: {video_path} file not found.")
        else:
            frame_paths = self.save_video_frames(line['video'])

            video_len_raw = line.get('video_length') 
            video_len = 0 
            
            if isinstance(video_len_raw, (int, float)) and video_len_raw > 0:
                video_len = round(video_len_raw, 2)
            
            if video_len > 0:
                num_frames = len(frame_paths) 
                time_context_prompt = (
                    f"The video is {video_len} seconds long. "
                    f"The following {num_frames} frames are uniformly sampled from it "
                    "in chronological order:"
                )
                msgs.append(dict(type='text', value=time_context_prompt))
            
            for frame_path in frame_paths:
                msgs.append(dict(type='image', value=frame_path))
            
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        """
        Core function for evaluating model predictions.
        """
        # Load the evaluation file as a DataFrame
        merged_df = load(eval_file)

        # Drop rows with missing answer or question_type
        merged_df.dropna(subset=['answer', 'question_type'], inplace=True)

        def calculate_score(row):
            prediction = row['prediction']
            answer = row['answer']
            q_type = row['question_type']
            q_class = row['category']
            
            if q_type == 'numerical':
                if q_class in ['absolute_speed', 'absolute_displacement']:
                    return calculate_metric_score_with_relative_error_consider_zero(prediction, answer, Thres=0.30)
                if q_class in ["trajectory_length"]:
                    return calculate_metric_score_with_relative_error_consider_zero(prediction, answer, Thres=2.0)
                return calculate_metric_score_with_relative_error(prediction, answer)
            elif q_type == 'mcq':
                pred_opt = extract_option_from_prediction(prediction)
                return 1 if pred_opt == str(answer) else 0
            
            return 0

        # Compute score for each row
        merged_df['score'] = merged_df.apply(calculate_score, axis=1)
        
        # Save the DataFrame with the new 'score' column to an Excel file
        excel_path = eval_file.rsplit('.', 1)[0] + ".xlsx"
        merged_df.to_excel(excel_path, index=False)

        # Compute overall accuracy
        overall_accuracy = merged_df['score'].mean()
        
        # Compute accuracy by category
        category_accuracy = merged_df.groupby('category')['score'].mean().reset_index()
        category_accuracy.rename(columns={'score': 'accuracy'}, inplace=True)
        report_df = category_accuracy.set_index('category')
        report_df.loc['Overall (Weighted Avg)'] = overall_accuracy

        return report_df
