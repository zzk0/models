import OpenAttack as oa
import datasets # use the Hugging Face's datasets library
# change the SST dataset into 2-class
def dataset_mapping(x):
    return {
        "x": x["sentence"],
        "y": 1 if x["label"] > 0.5 else 0,
    }
# choose a trained victim classification model
victim = oa.DataManager.loadVictim("BERT.SST")
# choose 20 examples from SST-2 as the evaluation data 
dataset = datasets.load_dataset("sst", split="train[:20]").map(function=dataset_mapping)
# choose PWWS as the attacker and initialize it with default parameters
attacker = oa.attackers.PWWSAttacker()
# prepare for attacking
metrics = [oa.metric.Fluency(), oa.metric.GrammaticalErrors(), oa.metric.SemanticSimilarity()]
attack_eval = oa.AttackEval(attacker, victim, metrics=metrics)
# launch attacks and print attack results 
attack_eval.eval(dataset, visualize=True)
