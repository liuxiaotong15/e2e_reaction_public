# e2e_reaction

For A+B->C reactions:

1. python generate_reaction.py (input: 'qm9.db'; output: 'reactions.txt')

2. python gen_boc.py (input: 'qm9.db'; output: 'qm9_boc_lst.txt')

3. python vali_reac.py (input: 'reactions.txt', 'qm9_boc_lst.txt'; output: 'dataset.lst')

4. python train.py (input: 'dataset.lst'; output: stdout results)

5. to compare with the megnet reactions property.

    5.1 python megnet_prediction.py (input: 'qm9.db', 'pretrained megnet model from github', output: 'megnet_gibbs.lst')
    
    5.2 python vali_reac_megnet.py (input: 'reactions.txt', 'megnet_gibbs.lst', output: stdout an accuracy)

For A -> C reactions (finding most stable molecule prediction):

P.S. if needed. python gen_boc.py (input: 'qm9.db'; output: 'qm9_boc_lst.txt')

1. python molecule_order.py (input: 'qm9.db'; output: 'qm9_gibbs_order.txt') 

    generate a list (row id in db, order in group, length of group), all molecules in 1 group have same fomular. There are 621 groups in qm9.db.

2. python generate_reaction_AC.py (input: 'qm9.db'; output: 'reactions.txt', 'reactions_AC_test_group.txt')

    current 10 of 621 formular group are used for test in 'reactions_AC_test_group.txt', which contain all A->C and C->A reactions.

    611 of 621 formular group generate the file 'reactions.txt', which only contain A->C reactions(Gibbs_A > Gibbs_C, C is more stable)

3. python vali_reac_AC.py (input: 'reactions.txt', 'reactions_AC_test_group.txt', 'qm9_boc_lst.txt'; output: 'dataset.lst', 'test_dataset.lst')

    generate 10M pos and 10M neg reactions. from 'reactions.txt'
    (BoC_A-BoC_C) is labeled as 1 
    (BoC_C-BoC_A) is labeled as -1 

    generate all BoC from 'reactions_AC_test_group.txt' to 'test_dataset.lst', format is:
    id1, id2, BoC(id1) - BoC(id2)

4. python train_AC.py (input: 'dataset.lst', 'test_dataset.lst'; output: stdout results and 'competition_with_ret.txt')

    to train and test model from dataset.lst and order the id in the test_group, 'competition_with_ret.txt' format is id1, id2, predict 1/0

    To compare with Megnet Gibbs prediction results:

    4.1 python megnet_prediction.py (input: 'qm9.db', 'pretrained megnet model from github', output: 'megnet_gibbs.lst')

5. python final_order.py (input: 'competition_with_ret.txt', 'megnet_gibbs.lst', stdout results and 'predict_actual_order_lst.txt', 'megnet_predict_actual_order_lst.txt')

6. python data_analysis.py ('predict_actual_order_lst.txt' or 'megnet_predict_actual_order_lst.txt') to draw the result.

For new BoC basis generation:

1. python generate_clustering_data.py (input: 'qm9.db', output: 'data_dict.lst' -> a dict, key is the symbol string, value is sorted distances)

2. python clustering.py (input: 'data_dict.lst', output: 'clst_dict.dct')

3. python gen_boc_new.py (input: 'clst_dict.dct', output: 'qm9_id_boc.lst')

4. python vali_reac_new.py (input: 'reactions.txt', 'qm9_id_boc.lst' , output: 'dataset_new.lst')