import numpy as np
import OpenAttack as oa
import datasets # use the Hugging Face's datasets library


class MyClassifier(oa.Classifier):

    def __init__(self):
        self.victim = oa.DataManager.loadVictim("BERT.SST")
    
    def get_pred(self, input_):
        res = self.victim.get_prob(input_)
        res = res > 0.5
        res = res.astype(np.float32)
        return res

    # access to the classification probability scores with respect input sentences
    def get_prob(self, input_):
        return self.victim.get_prob(input_)


# change the SST dataset into 2-class
def dataset_mapping(x):
    return {
        "x": x["sentence"],
        "y": [1 if x["label"] > 0.5 else 0, 1 if x["label"] > 0.5 else 0],
    }

# choose a trained victim classification model
# victim = oa.DataManager.loadVictim("BERT.SST")
victim = MyClassifier()

# choose 20 examples from SST-2 as the evaluation data 
dataset = datasets.load_dataset("sst", split="train[:20]").map(function=dataset_mapping)

# choose PWWS as the attacker and initialize it with default parameters
# attacker = oa.attackers.PWWSAttacker()
attacker = oa.attackers.BAEMCLttacker('/home/percent1/models/nlp/text-classification/pretrained/bert-base-uncased')

# prepare for attacking
# metrics = [oa.metric.Fluency(), oa.metric.GrammaticalErrors(), oa.metric.SemanticSimilarity()]
metrics = []
attack_eval = oa.AttackEval(attacker, victim, metrics=metrics)

# launch attacks and print attack results 
attack_eval.eval(dataset, visualize=True)
