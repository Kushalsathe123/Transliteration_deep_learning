import sys
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import json


indoarab_num = [chr(alpha) for alpha in range(48, 58)]
english_lower_script = [chr(alpha) for alpha in range(97, 123)]  
devanagari_script = ['ऄ', 'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ','ऍ', 'ऎ', 'ए', 'ऐ','ऑ', 'ऒ', 'ओ', 'औ','ऋ','ॠ','ऌ','ॡ','ॲ', 'ॐ','क', 'ख', 'ग', 'घ', 'ङ', 'च', 'छ', 'ज', 'झ', 'ञ', 'ट', 'ठ', 'ड', 'ढ', 'ण','त', 'थ', 'द', 'ध', 'न', 'ऩ', 'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ऱ', 'ल','ळ', 'ऴ', 'व', 'श', 'ष', 'स', 'ह', 'क़', 'ख़', 'ग़', 'ज़', 'ड़', 'ढ़', 'फ़', 'य़','्', 'ा', 'ि', 'ी', 'ु', 'ू', 'ॅ', 'ॆ', 'े', 'ै', 'ॉ', 'ॊ', 'ो', 'ौ','ृ', 'ॄ', 'ॢ', 'ॣ', 'ँ', 'ं', 'ः', '़', '॑',  'ऽ',chr(0x200c), chr(0x200d), ]     

class Vectorization():
    def __init__(self, lang_script = english_lower_script):
        self.glyphs = lang_script

        self.char2idx = {}
        self.idx2char = {}
        self._create_index()


    def _create_index(self):

        self.char2idx['_'] = 0  
        self.char2idx['$'] = 1  
        self.char2idx['#'] = 2  
        self.char2idx['*'] = 3  
        self.char2idx["'"] = 4  
        self.char2idx['%'] = 5  
        self.char2idx['!'] = 6  

        for idx, char in enumerate(self.glyphs):
            self.char2idx[char] = idx + 7 
        for char, idx in self.char2idx.items():
            self.idx2char[idx] = char


    def size(self):
        return len(self.char2idx)


    def word_to_vec(self, word):
        try:
            vec = [self.char2idx['$']] 
            for i in list(word):
                vec.append(self.char2idx[i])
            vec.append(self.char2idx['#']) 

            vec = np.asarray(vec, dtype=np.int64)
            return vec

        except Exception as error:
            print("Error In word:", word, "Error Char not in Token:", error)
            sys.exit()


    def vec_to_word(self, vector):

        char_list = []
        for i in vector:
            char_list.append(self.idx2char[i])

        word = "".join(char_list).replace('$','').replace('#','') 
        word = word.replace("_", "").replace('*','') 
        return word


class TranslitDataset(Dataset):
    def __init__(self, src_glyph_obj, tgt_glyph_obj,
                    json_file, file_map = "LangEn",
                    padding = True, max_seq_size = None,
                 ):
        if file_map == "LangEn": 
            tgt_str, src_str = self._json2_k_v(json_file)
        elif file_map == "EnLang": 
            src_str, tgt_str = self._json2_k_v(json_file)
        else:
            raise Exception('Unknown JSON structure')

        self.src_glyph = src_glyph_obj
        self.tgt_glyph = tgt_glyph_obj

        __svec = self.src_glyph.word_to_vec
        __tvec = self.tgt_glyph.word_to_vec
        self.src = [ __svec(s)  for s in src_str]
        self.tgt = [ __tvec(s)  for s in tgt_str]

        self.tgt_class_weights = self._char_class_weights(self.tgt)

        self.padding = padding
        if max_seq_size:
            self.max_tgt_size = max_seq_size
            self.max_src_size = max_seq_size
        else:
            self.max_src_size = max(len(t) for t in self.src)
            self.max_tgt_size = max(len(t) for t in self.tgt)

    def __getitem__(self, index):
        x_sz = len(self.src[index])
        y_sz = len(self.tgt[index])
        if self.padding:
            x = self._pad_sequence(self.src[index], self.max_src_size)
            y = self._pad_sequence(self.tgt[index], self.max_tgt_size)
        else:
            x = self.src[index]
            y = self.tgt[index]
        return x,y, x_sz

    def __len__(self):
        return len(self.src)


    def _json2_k_v(self, json_file):
        with open(json_file, 'r', encoding = "utf-8") as f:
            data = json.load(f)

        x = []; y = []
        for k in data:
            for v in data[k]:
                x.append(k); y.append(v)

        return x, y


    def _pad_sequence(self, x, max_len):
        padded = np.zeros((max_len), dtype=np.int64)
        if len(x) > max_len: padded[:] = x[:max_len]
        else: padded[:len(x)] = x
        return padded

    def _char_class_weights(self, x_list, scale = 10):
        from collections import Counter
        full_list = []
        for x in x_list:
            full_list += list(x)
        count_dict = dict(Counter(full_list))

        class_weights = np.ones(self.tgt_glyph.size(), dtype = np.float32)
        for k in count_dict:
            class_weights[k] = (1/count_dict[k]) * scale

        return class_weights

def create_dataloader(data_file, src_glyph_obj, tgt_glyph_obj, batch_size, shuffle=True):
    dataset = TranslitDataset(data_file, src_glyph_obj, tgt_glyph_obj)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def load_data(data_file, src_glyph_obj, tgt_glyph_obj, batch_size):
    train_dataloader = create_dataloader(data_file, src_glyph_obj, tgt_glyph_obj, batch_size)
    return train_dataloader

