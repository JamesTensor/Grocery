import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

#基于Ripplet的food_embedding，来计算相似度，没有上线仅是自己看看
class FoodSimilarity(object):
    def __init__(self, emb_file, name2id_mapping_file):
        super(FoodSimilarity, self).__init__()
        self.mapping = dict()
        with open(name2id_mapping_file, 'r', encoding='utf-8') as mfile:
            lines = mfile.read().strip().split('\n')
            for line in lines:
                try:
                    name, _, r1Mid, _ = line.strip().split('\t')
                    self.mapping[name] = r1Mid
                except Exception as e:
                    print(line)
                    raise e

        with open(emb_file, 'rb') as f:
            _ = pickle.load(f)
            ingre_emb = pickle.load(f)
            recipe_id = pickle.load(f)
            _ = pickle.load(f)

        self.emb = {recipe_id[i]:ingre_emb[i] for i in range(len(recipe_id))}

    def get_cosine_distance(self, rec1, rec2):
        if isinstance(rec1, str):
            rec1 = [rec1]

        if isinstance(rec2, str):
            rec2 = [rec2]

        list1 = [self.emb[self.mapping.get(each, None)] for each in rec1 if self.mapping.get(each, None) in self.emb]
        list2 = [self.emb[self.mapping.get(each, None)] for each in rec2 if self.mapping.get(each, None) in self.emb]
        if len(list1) > 0 and len(list2) > 0:
            cosine_distance = 1 - cosine_similarity(list1, list2)
            return cosine_distance
        else:
            return None
