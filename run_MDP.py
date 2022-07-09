import sys

sys.path.append('../')
from dataset.Preprocess import PreprocessDataset
import numpy as np
from Click_Model.CM_model import CM
from Ranker.MDPRankerV2 import MDPRankerV2
from utils import evl_tool
from utils.utility import get_DCG_rewards, get_DCG_MDPrewards
import multiprocessing as mp
import pickle
import os

from torch.utils.tensorboard import SummaryWriter

FEATURE_SIZE = 46
NUM_INTERACTION = 1000
LR = 0.001
ETA = 1


# %%
def run(train_set, test_set, ranker, eta, reward_method, num_interation,
        click_model):
    ndcg_scores = []
    cndcg_scores = []
    query_set = train_set.get_all_querys()
    index = np.random.randint(query_set.shape[0], size=num_interation)
    num_iter = 0
    for i in index:
        qid = query_set[i]
        result_list = ranker.get_query_result_list(train_set, qid)
        clicked_doces, click_labels, _ = click_model.simulate(
            qid, result_list, train_set)

        # no click then skip
        if len(clicked_doces) == 0:
            if num_iter % 1000 == 0:
                all_result = ranker.get_all_query_result_list(test_set)
                ndcg = evl_tool.average_ndcg_at_k(test_set, all_result, 10)
                ndcg_scores.append(ndcg)

            cndcg = evl_tool.query_ndcg_at_k(train_set, result_list, qid, 10)
            cndcg_scores.append(cndcg)
            num_iter += 1
            continue

        propensities = np.power(
            np.divide(1, np.arange(1.0,
                                   len(click_labels) + 1)), eta)

        # directly using pointwise rewards
        # rewards = get_DCG_rewards(click_labels, propensities, reward_method)

        # using listwise rewards
        rewards = get_DCG_MDPrewards(click_labels,
                                     propensities,
                                     reward_method,
                                     gamma=0)

        # ranker.record_episode(qid, result_list, rewards)

        ranker.update_policy(qid, result_list, rewards, train_set)

        if num_iter % 1000 == 0:
            all_result = ranker.get_all_query_result_list(test_set)
            ndcg = evl_tool.average_ndcg_at_k(test_set, all_result, 10)
            ndcg_scores.append(ndcg)
            print(f"iteration={num_iter}: ndcg={ndcg}")
        cndcg = evl_tool.query_ndcg_at_k(train_set, result_list, qid, 10)
        cndcg_scores.append(cndcg)

        # print(num_iter)
        num_iter += 1
    return ndcg_scores, cndcg_scores


def job(model_type, learning_rate, eta, reward_method, f, train_set, test_set,
        num_features):

    # if model_type == "perfect":
    #     pc = [0.0, 0.2, 0.4, 0.8, 1.0]
    #     ps = [0.0, 0.0, 0.0, 0.0, 0.0]
    # elif model_type == "navigational":
    #     pc = [0.05, 0.3, 0.5, 0.7, 0.95]
    #     ps = [0.2, 0.3, 0.5, 0.7, 0.9]
    # elif model_type == "informational":
    #     pc = [0.4, 0.6, 0.7, 0.8, 0.9]
    #     ps = [0.1, 0.2, 0.3, 0.4, 0.5]
    #
    if model_type == "perfect":
        pc = [0.0, 0.5, 1.0]
        ps = [0.0, 0.0, 0.0]
    elif model_type == "navigational":
        pc = [0.05, 0.5, 0.95]
        ps = [0.2, 0.5, 0.9]
    elif model_type == "informational":
        pc = [0.4, 0.7, 0.9]
        ps = [0.1, 0.3, 0.5]
    cm = CM(pc, ps)

    # create result folders
    if not os.path.exists("./results_MDP/Fold{}/{}".format(f,model_type)):
        os.makedirs("./results_MDP/Fold{}/{}".format(f,model_type))
    if not os.path.exists(f"./results_MDP/Fold{f}/{model_type}/checkpoints"):
        os.makedirs(f"./results_MDP/Fold{f}/{model_type}/checkpoints")

    ranker = MDPRankerV2(256,
                        num_features,
                        learning_rate,
                        loss_type='pairwise')
    for r in range(1, 16):
        np.random.seed(r)
        print("************************************************************************")
        print("MDP Adam:  Folder: Fold{}\tModel: {}\teta:{}\treward:{}".format(
            f, model_type, eta, reward_method, r))
        print(f"Training {r} start...")
        
        ndcg_scores, cndcg_scores = run(train_set, test_set, ranker, eta,
                                        reward_method, NUM_INTERACTION, cm)

        # os.makedirs(os.path.dirname("{}/Fold{}/".format(output_fold, f)),
        #             exist_ok=True)  # create directory if not exist
        # with open(
        #         "{}/fold{}/{}_run{}_ndcg.txt".format(output_fold, f,
        #                                              model_type, r),
        #         "wb") as fp:
        #     pickle.dump(ndcg_scores, fp)
        # with open(
        #         "{}/fold{}/{}_run{}_cndcg.txt".format(output_fold, f,
        #                                               model_type, r),
        #         "wb") as fp:
        #     pickle.dump(cndcg_scores, fp)
        #
        # print("MDP MSLR10K fold{} {} eta{} reward{} run{} done!".format(f, model_type, eta, reward_method, r))

        # SummaryWriter
        writer = SummaryWriter('./results_MDP/Fold{}/{}'.format(f,model_type))
        for i in range(len(cndcg_scores)):
            writer.add_scalar(f'cndcg_Fold{f}_under_{model_type}_Model',cndcg_scores[i],
                                          i+(r-1)*NUM_INTERACTION)
            if i%1000 == 0:
                writer.add_scalar(f'ndcg_Fold{f}_under_{model_type}_Model',ndcg_scores[int(i/1000)],
                                          int((i+(r-1)*NUM_INTERACTION)/1000))
        writer.close()

        # model save
        checkpoint_path = f"./results_MDP/Fold{f}/{model_type}/checkpoints"
        ranker.save_model(path=checkpoint_path+"model",
                            globalstep=r*NUM_INTERACTION, 
                            write_graph= (True if r==1 else False))
        print(f"Training {r} finish...")
        print("")  # line feed


if __name__ == "__main__":
    learning_rate = LR
    eta = ETA
    reward_method = "both"

    click_models = ["informational", "perfect"]
    # click_models = ["perfect"]
    dataset_fold = "./dataset/MQ2007"

    # for 5 folds
    for f in range(1, 6):
        training_path = "{}/Fold{}/train.txt".format(dataset_fold, f)
        test_path = "{}/Fold{}/test.txt".format(dataset_fold, f)
        train_set = PreprocessDataset(training_path,
                                 FEATURE_SIZE,
                                 query_level_norm=False)
        test_set = PreprocessDataset(test_path,
                                FEATURE_SIZE,
                                query_level_norm=False)
        # %%
        processors = []
        # for click_model in click_models:
            # p = mp.Process(target=job,
            #                args=(click_model, learning_rate, eta,
            #                      reward_method, f, train_set, test_set,
            #                      FEATURE_SIZE))
        #     p.start()
            # processors.append(p)
    # for p in processors:
    #     p.join()

        for click_model in click_models:
            job(click_model, learning_rate, eta,
                reward_method, f, train_set, test_set,
                FEATURE_SIZE)
        
