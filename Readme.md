# Experiment on Offline Policy Evaluation and Learning

## Dataset

class `PreprocessDataset(AbstractDataset)`

- **Attributes**:

  - `_path` *str*, the path where the dataset is

  - `_feature_size` *int*, size of features, in LeToR, 46

  - `_query_docid_get_features` *dict*, a dict with depth of 2, search *features* by *query document pair*

    - structure: 

      ```python
      - query1:
      	- docid11: featurevector1
      	- docid12: featurevector2
      	- ...
      	- docid1n: featurevectorn
      - query2: 
      	- docid21: featurevector1
      	- docid22: featurevector2
      	- ...
      - ...
      ```

  - `_query_get_docids` *dict*, a dict with depth 1, search *docids* by *query*

    - structure:

      ```python
      - query1: [docid1, docid2, ...]
      - query2: [docid1, docid2, ...]
      - ...
      ```

  - `_query_get_all_features` *dict*, a dict with depth 1, search *features* by *query* directly

    - structure:

      ```python
      - query1: np.array([[feature_1],[feature_2], ...[feature_n]])
      - query2: np.array([[feature_1],[feature_2], ...[feature_n]])
      - ...
      ```

  - `_query_docid_get_rel` *dict*, a dict with depth 2, search *relevance* by *query document pair*

    - structure:

      ```python
      - query1:
      	- docid11: rel_1
      	- docid12: rel_2
      	- ...
      	- docid1n: rel_n
      - query2: 
      	- docid21: rel_1
      	- docid22: rel_2
      	- ...
      - ...
      ```

  - `_query_pos_docids` *dict*, a dict with depth 1, store *relevant documents* by query

    - structure:

      ```python
      - query1: [docid_1, docid_2, ..., docid_n]
      - query2: [docid_1, docid_2, ..., docid_n]
      ...
      ```

    - **Note that** only relevant query-doc pairs will be record in the dict

  - `_query_relevant_labels` *dict*, a dict with depth 1, store lists of *relevance*  by *query*

    - structure:

      ```python
      - query1: [rel_1, rel_2, ..., rel_n]
      - query2: [rel_1, rel_2, ..., rel_n]
      ...
      ```

  - `_query_level_norm` *bool*, set 1 if want to do query level norm, otherwise 0

  - `_binary_label` *bool*, set 1 if relevance label is binary, otherwise 0

  - `_comments` *dict*, store comment(str) of each query-doc pair

    - structure:

      ```python
      - query1: ['commment_1', 'comment_2', ..., 'comment_n']
      - query2: ['comment_1', 'comment_2', ..., 'comment_n']
      - ...
      ```

- **Some Methods**: mainly `get` methods

## Ranker

- Neural Net Ranker with ***framework*** below:

<img src="C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20220707210346290.png" alt="image-20220707210346290" style="zoom:80%;" />

- ***framework*** of MDP Ranker, however,  is simple: 

​		a ***linear layer***:  $\text{scores}_{1\times n-doc} = \left(\text{input-docs}_{n\_doc\times \text{n\_features}}\cdot W_{\text{n\_features}\times 1}\right)^T$ 

​		a ***softmax*** layer to get *probability* of each doc

## Click Model

- **Attributes**
  - `parameter_dict` *dict*, depth of 2
  - `stat_dict` *dict*, depth of 2
  - `alpha`, `beta` the parameters in the model: $P(C=1\mid d,k)=\alpha_k P(R=1\mid d) + \beta_k$
  - `pc` *list*, click probability
- **Methods**
  - `set_probs(self, pc)` get click prob.
  - `simulate(self, query, result_list, dataset)` simulate user clicks on the result list made by the ranker, conditioned on some query
  - 

## Process of the Whole Model

### 1. Preprocess 

- `dataset`: get ***data*** from dataset and ***collect*** them in the right format (as mentioned in "Dataset" part), together with their `getter` methods

- `query set`: get all the ***queries*** in the query set

### 2. Things to do in each `run()` 

1. First, randomly sample `num_iteration` queries in the query set
2. For each query:
   1. ***form a*** ***result list*** using `ranker.get_query_result_list(train_set, qid)`
      - How to get a result list?
        1. Firstly, get all the ***features*** and ***docs*** by the given query, respectively
        2. Secondly, get all the ***scores*** of the docs by feeding `get_scores()` with features
        3. In ***each position*** in result list: 
           1. get ***probs*** of each doc by feeding `softmax()` with left docs' scores
           2. randomly choose a doc under the probs' distribution
           3. ***append*** the chosen doc in the result list, and then ***delete*** it in the score list and positions list
        4. stop till all the docs are in the result list
   2. ***simulate clicks*** by the click model, and get ***click docs*** and ***click labels***
      - format of ***click docs*** and ***click labels***
        - ***click docs***: *list*, contains all the clicked docids
        - ***click labels***: *np.array*, len=len(result_list), get 1 if the corresponding doc is clicked, otherwise 0
      - Specific process depends on the click model the user use. In our ***Cascade model***, we suppose the user goes through the result from top to bottom in sequence, from the first position
        1. randomly sample a ***click prob*** from (0,1] uniformly
        2. get ***relevance*** of the doc-query pair, and get the corresponding ***assumed click prob*** ***(pc)*** by its relevance under different assumption (informational, navigational, perfect)
        3. if `click_prob <= pc[relevance]`, which means the doc is clicked, we append it to the click docs list and flip the click label of the doc. And then break the loop. Otherwise, search for next position until a doc was clicked or the whole result list is searched.
   3. If no click exists, just ***evaluate** **on the test_set*** (more details in the following parts), otherwise, we ***update our policy*** and then do validation
      1. ***get propensities*** $P\left(o_{k}=1 \mid k\right)=\left(\frac{1}{k}\right)^{\eta}$   <span style="color:red"> This assumption need to be changed in the future</span>
      2. ***get rewards*** as matric of update: `get_DCG_MDPrewards()`
         - get ***DCG rewards***: 
           - positive: $R_{IPS+}(s_t,a_t)=\frac{\lambda(t)}{P(o_{t+1}=1\mid t+1)}\cdot c_m(a_t)$
           - negative: $R_{IPS-}(s_t,a_t)=\frac{\lambda(t)}{P(o_{t+1}=1\mid t+1)}\cdot c_m(a_t)-\lambda(t)$
           - both: $R_{BOTH}(s_t,a_t) = R_{IPS+} + R_{IPS-}$
         - get ***MDP DCG rewards*** if loss type is pair-wise
           - $G_t = \sum_{m=t}^{M}\gamma^{m-t}\cdot R(s_m, a_m)$, where $M$ is the index of the end doc, $R(s_m, a_m)$ is the DCG rewards defined  as below
         - ***update policy*** via `ranker.update_policy()`
           1. get `feature_matrix (n_doc x n_features)` and clear out the gradient
           2. calculate gradient according to the loss type
              - if *pointwise*: apply gradient to each position in the result list and then update via SGD or Adam
              - if *pairwise*: for each position in the list, apply gradient to it and all the docs after, then update via SGD or Adam
              - <span style="color:red">not quite sure how the loss is defined and the gradient is applied</span>
   4. ***Evaluation via DCG measures***: `average_ndcg_at_k()` and `query_ndcg_at_k()`
      - `average_ndcg_at_k()`: ***calculate*** ***ndcg*** (after normalized by an ideal list) and then ***average*** on all the queries. Remember, this operation is done on the ***test set*** as a ***validation***. 
        - Firstly, from test set, get the result lists of all the queries by `ranker.get_all_query_result_list()` 
          - return `query_result_list`, a *dict*  with queries as keys,  score list as value
          - For each query, the result list is sorted by scores (<span style="color:red">in MDP, only a linear layer</span>) from larger to smaller
        - Then for each query, we look at the ***first k*** docs in two ranking lists: ***the one generated by our MDPRanker*** and ***the ideal list ordered by relevance label***
          - matric: $\text{DCG@k} = \sum_{i=1}^{k}\frac{2^{rel_i}-1}{\log_2(i+1)}$
          - for each first-k doc, calculate DCG@k matric on both two ranking lists. Trivially, the DCG score of the ideal list must be larger than or equal to our ranking list
          - calculate $\text{nDCG@k} = \text{DCG@k} / \text{DCG@k}_{ideal}$
        - ***Average*** on all the queries (if `count_bad_query == True`,  take queries with no relevant docs into consideration, otherwise don't)
      - `query_ndcg_at_k()`: just calculate nDCG@k on current query, if no relevant docs in this query, return 0. The latter are the same as `average_ndcg_at_k()`

### 3. Hyperparameters

- `FEATURE_SIZE`: based on dataset MQ2007, 46
- `NUM_ITERATION`: empirically , 50000 (In experiment, the optimizer is ***Adam***, feel that the ranker doesn't converge, maybe larger)
- `LR`: learning rate, empirically, 0.001
- `ETA`: propensity factor $\eta$, empirically, 1
- `gamma`: decay factor in MDP reward, empirically, 0.99
- `reward_method`: we use `both` method
- `click_models`: two model: ***informational and perfect***
  - informational: `pc = [0.4, 0.7, 0.9]` (noisy)
  - perfect: `pc = [0.0, 0.5, 1.0]` (proportional  to relevance label)
