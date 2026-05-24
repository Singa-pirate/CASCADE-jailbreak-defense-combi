import pandas as pd
import requests
from tqdm import tqdm
from time import sleep, time
from dotenv import load_dotenv

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[3]))

from data.classes.base import DatasetBase
from data.classes.jbb_behaviours import JbbBehavioursDataset

load_dotenv()

class GoogleTranslator:
    translation_url = 'https://translate.googleapis.com/translate_a/single'
    fallback_translation_url = 'https://translation.googleapis.com/language/translate/v2'
    request_delay = 1
    rate_limit_cooldown = 60 # wait for 60 seconds on rate limit

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.rate_limited = False
        self.rate_limit_expiry_timestamp = None
    
    def normal_translate(self, text, src_lang='auto', target_lang='en'):
        url = f"{GoogleTranslator.translation_url}?client=gtx&sl={src_lang}&tl={target_lang}&dt=t&q={text}"
        data = requests.get(url).json()
        sleep(GoogleTranslator.request_delay)

        res = ''.join([s[0] for s in data[0]])
        return res

    def fallback_translate(self, text, src_lang=None, target_lang='en'):
        url = f"{GoogleTranslator.fallback_translation_url}?key={self.api_key}"
        payload = {
            'q': text,
            'target': target_lang,
            'format': 'text'
        }
        if src_lang is not None and src_lang != 'auto':
            payload['source'] = src_lang
        response = requests.post(url, data=payload)
        data = response.json()
        translated_text = data['data']['translations'][0]['translatedText']
        sleep(GoogleTranslator.request_delay)
        return translated_text
    
    def reliable_translate(self, text, src_lang='auto', target_lang='en'):
        # check if rate limit expired
        if self.rate_limited:
            current_timestamp = time()
            if current_timestamp > self.rate_limit_expiry_timestamp:
                self.rate_limited = False
        
        if self.rate_limited:
            return self.fallback_translate(text, src_lang, target_lang)
        else:
            try:
                return self.normal_translate(text, src_lang, target_lang)
            except Exception as e:
                print(f"Normal translation failed with error: {e}. Switching to fallback translator.")
                self.rate_limited = True
                self.rate_limit_expiry_timestamp = time() + GoogleTranslator.rate_limit_cooldown
                return self.fallback_translate(text, src_lang, target_lang)


def translate(text, src_lang='auto', target_lang='en'):
    # init translator to global context
    global google_translator
    if 'google_translator' not in globals():
        import os
        api_key = os.getenv('google_translate_api_key')
        google_translator = GoogleTranslator(api_key=api_key)
    return google_translator.reliable_translate(text, src_lang, target_lang)

def MJ_translate_dataset(dataset: DatasetBase, name: str = None) -> DatasetBase:
    goals = dataset.load_goals()
    
    target_languages = ['zh-CN', 'it', 'vi', 'ar', 'ko', 'th', 'bn', 'sw', 'jv']
    
    translations_data = []
    
    for goal in tqdm(goals):
        translation_row = {'en': goal}
        
        for lang_code in target_languages:
            translated = translate(text=goal, src_lang='en', target_lang=lang_code)
            translation_row[lang_code] = translated
        
        translations_data.append(translation_row)
    
    df = pd.DataFrame(translations_data)
    save_path = 'data/' + (f'{name}_' if name else '') + 'MJ_translated.csv'
    df.to_csv(save_path, index=False)

    print(f"Translated dataset saved to {save_path}")

if __name__ == "__main__":
    dataset = JbbBehavioursDataset()
    MJ_translate_dataset(dataset, name="jbb_behaviours")
