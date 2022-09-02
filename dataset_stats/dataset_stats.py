import matplotlib.pyplot as plt
import seaborn as sns
import json

#%%
'''
To compute statistics on distribution of number of triples in a dataset
'''

train_dict = {}
val_dict = {}
test_both_dict = {}
test_seen_dict = {}
test_unseen_dict = {}
split_list = ['train', 'val', 'test_both', 'test_seen', 'test_unseen']
dir_name = 'ana_stats_ANA'

with open(f'/Users/josephkeenan/nlp/internship_2022/webnlg_exp_repo/dataset_stats/{dir_name}/{dir_name}.json', 'r') as graph_stats:
    graph_stats = json.load(graph_stats)

    dict_list = [train_dict, val_dict, test_both_dict, test_seen_dict, test_unseen_dict]

    for idx, split in enumerate(graph_stats):
        for number in graph_stats[split]:
            dict_list[idx][number] = len(graph_stats[split][number])


for idx, d in enumerate(dict_list):
    keys = sorted(list(d.keys()))
    values = [float(d[key]) for key in keys]
    plt.figure()
    sns.barplot(x=keys, y=values)
    plt.title(f'Distribution of Number of Triples in {split_list[idx]} Base Dataset Split')
    plt.xlabel('Number of triples')
    plt.ylabel('Number of inputs')
    plt.savefig(f'/Users/josephkeenan/nlp/internship_2022/webnlg_exp_repo/dataset_stats/{dir_name}/{split_list[idx]}_plot_ANA.png')
    print(f'Successfully created {split_list[idx]} plot')
    print(d)
    print('\n')

#%%
'''
To compute percentage of subjects, predicates and objects that occur in train set but not test sets
'''

test_keys = ['test_both', 'test_seen', 'test_unseen']

test_both = {}
test_seen = {}
test_unseen = {}
test_key_dicts = [test_both, test_seen, test_unseen]

with open('/Users/josephkeenan/nlp/internship_2022/webnlg_exp_repo/dataset_stats/entities_BASE.json') as entities:
    entities = json.load(entities)

    for idx, testkey in enumerate(test_keys):
        for key in entities[testkey]:
            unseen_ent_count = 0
            for entity in entities[testkey][key]:
                if entity not in entities['train'][key]:
                    unseen_ent_count += 1

            print(f'{unseen_ent_count} unseen {key} in {testkey}. That is {round(unseen_ent_count/len(entities[testkey][key]), 5) * 100}% of the {testkey} data.')
            test_key_dicts[idx][f'unseen {key} percent'] = round(unseen_ent_count/len(entities[testkey][key]), 5) * 100
        print('\n')

        with open(f'/Users/josephkeenan/nlp/internship_2022/webnlg_exp_repo/dataset_stats/entity_stats_{testkey}_BASE.json', 'w') as f:
            json.dump(test_key_dicts[idx], f)


#%%
'''
To compute percentage of subjects, predicates, and objects that occur in the Base dataset but not the Analogy dataset
'''

keys = ['train', 'val', 'test_both', 'test_seen', 'test_unseen']

train = {}
val = {}
test_both = {}
test_seen = {}
test_unseen = {}
key_dicts = [train, val, test_both, test_seen, test_unseen]

base = open('/Users/josephkeenan/nlp/internship_2022/webnlg_exp_repo/dataset_stats/entity_stats/entities_BASE.json')
base_dict = json.load(base)
ana = open('/Users/josephkeenan/nlp/internship_2022/webnlg_exp_repo/dataset_stats/entity_stats/entities_ANA.json')
ana_dict = json.load(ana)

for idx, split in enumerate(base_dict):
    for entity_split in base_dict[split]:
        unseen_ent_count = 0
        for entity in base_dict[split][entity_split]:
            if entity not in ana_dict[split][entity_split]:
                unseen_ent_count += 1
        key_dicts[idx][f'percent unused {entity_split} in ANA'] = round(unseen_ent_count/len(base_dict[split][entity_split]), 5) * 100
        print(f'{split} {entity_split}: {round(unseen_ent_count/len(base_dict[split][entity_split]), 5) * 100}%')
    print('\n')

    with open(f'/Users/josephkeenan/nlp/internship_2022/webnlg_exp_repo/dataset_stats/entity_stats_{split}_ANAvBASE.json', 'w') as f:
        json.dump(key_dicts[idx], f)



base.close()
ana.close()
