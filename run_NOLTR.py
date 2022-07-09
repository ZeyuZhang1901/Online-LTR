import sys
import os

sys.path.append('../')
from dataset.Preprocess import PreprocessDataset
from Ranker.NeuralRanker import NeuralRanker
from Click_Model.CM_model import CM
from utils import evl_tool
import numpy as np
import multiprocessing as mp
import pickle

from torch.utils.tensorboard import SummaryWriter

FEATURE_SIZE = 46
NUM_INTERACTION = 1000
lr = 0.01

def run(train_set, test_set, ranker, num_interation, click_model):
    ndcg_scores = []
    cndcg_scores = []
    query_set = train_set.get_all_querys()
    index = np.random.randint(query_set.shape[0], size=num_interation)  # randomly select NUM_INTERACTION queries
    num_interation = 0
    for i in index:  # for each selected query
        num_interation += 1
        qid = query_set[i]

        result_list, scores = ranker.get_query_result_list(train_set, qid)  # get a result list with the ranker

        clicked_doc, click_label, _ = click_model.simulate(  # simulate
            qid, result_list, train_set)
        if len(clicked_doc) > 0:
            ranker.update(click_label, result_list,  # ranker update if some docs are clicked
                          train_set.get_all_features_by_query(qid))

        all_result = ranker.get_all_query_result_list(test_set)  # with each query, docids selected by score, from top to bottom
        ndcg = evl_tool.average_ndcg_at_k(test_set, all_result, 10)
        cndcg = evl_tool.query_ndcg_at_k(train_set, result_list, qid, 10)

        ndcg_scores.append(ndcg)
        cndcg_scores.append(cndcg)
        final_weight = ranker.get_current_weights()

        if num_interation % 100 == 0:
            print(num_interation, ndcg, cndcg)

    return ndcg_scores, cndcg_scores, final_weight


def job(model_type, f, train_set, test_set):
    if model_type == "perfect":
        pc = [0.0, 0.5, 1.0]
        ps = [0.0, 0.0, 0.0]
    elif model_type == "navigational":
        pc = [0.05, 0.5, 0.95]
        ps = [0.2, 0.5, 0.9]
    elif model_type == "informational":
        pc = [0.4, 0.7, 0.9]
        ps = [0.1, 0.3, 0.5]
    
    Learning_rate = lr
    cm = CM(pc, ps)

    # create result folders
    if not os.path.exists("./results_Neural/Fold{}/{}".format(f,model_type)):
        os.mkdir("./results_Neural/Fold{}/{}".format(f,model_type))
    if not os.path.exists(f"./results_Neural/Fold{f}/{model_type}/checkpoints"):
        os.mkdir(f"./results_Neural/Fold{f}/{model_type}/checkpoints")

    ranker = NeuralRanker(FEATURE_SIZE, Learning_rate)
    for r in range(1, 6):
        np.random.seed(r)
        print("************************************************************************")
        print(f"Folder: Fold{f}\tModel type: {model_type}")
        print(f"Training {r} start...")
        ndcg_scores, cndcg_scores, final_weight = run(train_set, test_set,
                                                      ranker, NUM_INTERACTION,
                                                      cm)    

        # SummaryWriter
        writer = SummaryWriter('./results_Neural/Fold{}/{}'.format(f,model_type))
        for i in range(len(ndcg_scores)):
            writer.add_scalars(f'Folder{f}_under_{model_type}_Model', {'ndcg_scores': ndcg_scores[i],
                                          'cndcg_scores':cndcg_scores[i]},
                                          i+(r-1)*NUM_INTERACTION)
        writer.close()
        print("")  # line feed

        # model save
        checkpoint_path = f"./results_Neural/Fold{f}/{model_type}/checkpoints"
        ranker.save_model(path=checkpoint_path+"model",
                            globalstep=r*NUM_INTERACTION, 
                            write_graph= (True if r==1 else False))
        print(f"Training {r} finish...")


if __name__ == "__main__":
    click_models = ["informational", "navigational", "perfect"]
    dataset_fold = "./dataset/MQ2007"
    output_fold = "./Output/MQ2007"
    # for 5 folds
    for f in range(1, 6):
        training_path = "{}/Fold{}/train.txt".format(dataset_fold, f)
        test_path = "{}/Fold{}/test.txt".format(dataset_fold, f)
        train_set = PreprocessDataset(training_path, FEATURE_SIZE)
        test_set =  PreprocessDataset(test_path, FEATURE_SIZE)

        # for 3 click_models
        # for click_model in click_models:
        #     mp.Process(target=job,
        #                args=(click_model, f, train_set, test_set)).start()
        #     break
        # break
        for click_model in click_models:
            job(click_model, f, train_set, test_set)
