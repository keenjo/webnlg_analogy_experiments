import json
import sys
import numpy as np
from xml.dom import minidom
import glob
from pathlib import Path
import re
import unidecode
import os
from tqdm import tqdm
import csv
import numpy as np
from collections import defaultdict

folder = sys.argv[1]

datasets = ['train', 'dev', 'test_both', 'test_seen', 'test_unseen']

OG = True
if OG is True: # This means the dev and test targets will be organized into 3 files each
    print(f'OG is {OG}. Data will be preprocessed to work with training script.')
else:
    # This means the test targets will be split into as many files as the maximum number of lexicalizations found in the data
    # This is the organization that is needed to run the evaluation script
    print(f'OG is {OG}. Data will be preprocessed to work with evaluation script.')
'''
- If OG is set to False the data will be preprocessed to work with the evaluation scripts
- If OG is set to True the data will be preprocessed to work with the Control Prefixes training script
    - The only difference is the number of folders everything is organized into in the end and whether to include eval_crf and eval_meteor files
'''

def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    d = [m.group(0) for m in matches]
    new_d = []
    for token in d:
        token = token.replace('(', '')
        token_split = token.split('_')
        for t in token_split:
            new_d.append(t)
    return new_d


def get_nodes(n):
    n = n.strip()
    n = n.replace('(', '')
    n = n.replace('\"', '')
    n = n.replace(')', '')
    n = n.replace(',', ' ')
    n = n.replace('_', ' ')

    n = unidecode.unidecode(n)

    return n


def get_relation(n):
    n = n.replace('(', '')
    n = n.replace(')', '')
    n = n.strip()
    n = n.split()
    n = "_".join(n)
    return n


def process_triples(mtriples):
    nodes = []
    edge_list = []  # List of edges for each triple
    sub_list = []  # List of subjects for each triple

    for idx, m in enumerate(mtriples):
        ms = m.firstChild.nodeValue
        ms = ms.strip().split(' | ')
        n1 = ms[0]
        n2 = ms[2]
        nodes1 = get_nodes(n1)
        sub_list.append(nodes1)

        nodes2 = get_nodes(n2)

        edge = get_relation(ms[1])
        edge_list.append(edge)

        edge_split = camel_case_split(edge)
        edges = ' '.join(edge_split)

        nodes.append('<H>')
        nodes.extend(nodes1.split())

        nodes.append('<R>')
        nodes.extend(edges.split())

        nodes.append('<T>')
        nodes.extend(nodes2.split())

    return nodes, sub_list, edge_list


def get_data_dev_test(file_, train_cat, dataset):

    itr = 0
    datapoints_list = []
    cats = set()
    # cats_list = []

    xmldoc = minidom.parse(file_)
    entries = xmldoc.getElementsByTagName('entry')
    cont = 0
    for e in entries:
        added_cat = False
        cat = e.getAttribute('category')
        if dataset == 'dev' or dataset == 'test_both':
            cats.add(cat)
            # cats_list.append(cat)
            added_cat = True
            cont += 1
        elif dataset == 'test_seen' and cat in train_cat:
            cats.add(cat)
            # cats_list.append(cat)
            added_cat = True
            cont += 1
        elif dataset == 'test_unseen' and cat not in train_cat:
            cats.add(cat)
            # cats_list.append(cat)
            added_cat = True
            cont += 1

        if added_cat == True:
            mtriples = e.getElementsByTagName('mtriple')
            nodes, sub_list, edge_list = process_triples(mtriples)
            # Have process_triples output your dictionary and take it back in each time
            lexs = e.getElementsByTagName('lex')
            datapoints = {}

            surfaces = []
            for idx, l in enumerate(lexs):
                # l = l.firstChild.nodeValue.strip().lower()
                l = l.firstChild.nodeValue.strip()
                new_doc = ' '.join(re.split('(\W)', l))
                new_doc = ' '.join(new_doc.split())
                # new_doc = tokenizer.tokenize(new_doc)
                # new_doc = ' '.join(new_doc)
                surfaces.append((l, new_doc.lower()))
            #datapoints.append([nodes, surfaces, sub_list, edge_list, [00, 00], cat])
            datapoints['graph'] = nodes
            datapoints['lex'] = surfaces
            datapoints['subjects'] = sub_list
            datapoints['predicates'] = edge_list
            datapoints['sim_scores'] = []
            datapoints['category'] = cat
            datapoints_list.append(datapoints)
            # Append a tuple containing ONE triple and the LIST of lexicalizations to the datapoints list

    return datapoints_list, cats, cont


def get_data(file_):
    datapoints_list = []
    cats = set()
    # cats_list = []

    xmldoc = minidom.parse(file_)
    entries = xmldoc.getElementsByTagName('entry')
    cont = 0
    for e in entries:
        cat = e.getAttribute('category')
        cats.add(cat)
        # cats_list.append(cat)

        cont += 1

        mtriples = e.getElementsByTagName('mtriple')
        nodes, sub_list, edge_list = process_triples(mtriples)

        lexs = e.getElementsByTagName('lex')

        for idx, l in enumerate(lexs):
            datapoints = {}
            # l = l.firstChild.nodeValue.strip().lower()
            l = l.firstChild.nodeValue.strip()
            new_doc = ' '.join(re.split('(\W)', l))
            new_doc = ' '.join(new_doc.split())
            # new_doc = tokenizer.tokenize(new_doc)
            # new_doc = ' '.join(new_doc)
            #datapoints.append([nodes, (l, new_doc.lower()), sub_list, edge_list, [00, 00], cat])
            datapoints['graph'] = nodes
            datapoints['lex'] = (l, new_doc.lower())
            datapoints['subjects'] = sub_list
            datapoints['predicates'] = edge_list
            datapoints['sim_scores'] = []
            datapoints['category'] = cat
            datapoints_list.append(datapoints)
            # Append a tuple containing ONE triple and ONE lexicalization to the datapoints list
            # edge_compare.append(edges)

    return datapoints_list, cats, cont


def find_analogies(datapoints, split):
    ana_count = 0
    for x in tqdm(range(len(datapoints)), desc='Finding Analogies'):
        # inner_ana_count = 0
        for y in range(len(datapoints)):
            x_subject_set = set(datapoints[x]['subjects'])
            y_subject_set = set(datapoints[y]['subjects'])
            inter_sub = x_subject_set & y_subject_set
            unique_subjects = (len(x_subject_set) + len(y_subject_set)) - len(inter_sub)  # Number of unique items in the two subject lists combined
            sub_sim_score = (len(inter_sub)) / unique_subjects  # Similarity score via percentage of overlap over unique subjects
            if sorted(x_subject_set) != sorted(y_subject_set) and sub_sim_score < 0.3:
                # If the subjects are not identical and their subject similarity score is below threshold
                x_predicate_set = set(datapoints[x]['predicates'])
                y_predicate_set = set(datapoints[y]['predicates'])
                inter_pred = x_predicate_set & y_predicate_set
                unique_predicates = (len(x_predicate_set) + len(y_predicate_set)) - len(inter_pred)
                pred_sim_score = len(inter_pred) / unique_predicates  # Similarity score via percentage of overlap over unique predicates
                if pred_sim_score > 0.6:  # If the predicate similarity score is above threshold
                    datapoints, ana_count = add_analogies(datapoints, x, y, ana_count, split=split)
                    # inner_ana_count += 1
                    # ana_count += 1
                    datapoints[x]['sim_scores'].append(sub_sim_score)
                    datapoints[x]['sim_scores'].append(pred_sim_score)
                    # Idea if we wanted to look for more than 1 analogy
                    # add a count inside the x loop
                    # Also may be best to add these additional analogies to the same list as the original,
                    # so you'd have to make some slight changes to add_analogies function as well as the surface list organization,
                    # but nothing major, probably just adding another 'for loop'
                    break

    #if split == 'train' or split == 'dev':
    print(f'{ana_count}/{len(datapoints)} total analogies')
    #else:
        #data_count = [len(item) for item in datapoints['ana_input'] if item != []]
        #print(f'{ana_count}/{sum(data_count)} total analogies')
    return datapoints


def add_analogies(datapoints, x, y, ana_count, split):
    orig_triple = datapoints[x]['graph'].copy()
    triple_addition = datapoints[y]['graph'].copy()
    if split != 'train':
        #if split == 'dev' or 'test' in split:  # Basically if we are dealing with the 'dev/val' or one of the 'test' splits
            # We have the extra level of indexing here because in the non-train splits there is a possibility to have multiple lexicalizations
            # For our dev/val set we just choose the first one in the list
        lex_addition = datapoints[y]['lex'][0][0].split()
        #else:  # If the split is 'test'
            #lex_additions = [lex[0].split() for lex in list(set(datapoints[y]['lex']))]
    else:  # If we ARE dealing with the 'train' split
        lex_addition = datapoints[y]['lex'][0].split()

    #if split == 'train' or split == 'dev':  # If we are not dealing with one of the three 'test' splits we will only add one lexicalization
    lex_addition.append('<<G>>')  # Add graph separator to separate analogy graphs from main graph
    triple_addition.append('<L>')  # Add lexicalization separator
    combined_addition = triple_addition + lex_addition + orig_triple
    datapoints[x]['ana_input'] = combined_addition
    ana_count += 1
    #else:  # If we ARE in fact dealing with one of the 'test' splits we will add a new input for each lexicalization
        #new_additions = []
        #triple_addition.append('<L>')  # Add lexicalization separator
        #for lex_addition in lex_additions:
            #lex_addition.append('<<G>>')  # Add graph separator to separate analogy graphs from main graph
            #new_additions.append(triple_addition + lex_addition + orig_triple)
            #ana_count += 1

        #datapoints[x]['ana_input'].append(new_additions)
    return datapoints, ana_count


# Work on getting this function to work, it'd be a lot more elegant below
'''
def organize_final_data(nodes, node, sur, surface_lists, surface_eval_lists, count):

    surfaces, surfaces_2, surfaces_3, surfaces_4, surfaces_5 = zip(*surface_lists)
    surfaces_eval, surfaces_2_eval, surfaces_3_eval, surfaces_4_eval, surfaces_5_eval = zip(*surface_eval_lists)

    nodes.append(' '.join(node))
    if part != 'train':
        surfaces.append(sur[0][0])
        surfaces_eval.append(sur[0][1])
        if len(sur) > 1:
            surfaces_2.append(sur[1][0])
            surfaces_2_eval.append(sur[1][1])
        else:
            surfaces_2.append('')
            surfaces_2_eval.append('')
        if len(sur) > 2:
            surfaces_3.append(sur[2][0])
            surfaces_3_eval.append(sur[2][1])
        else:
            surfaces_3.append('')
            surfaces_3_eval.append('')
        if 'test' in part:  # only continue on to these steps for test set
            if len(sur) > 3:
                surfaces_4.append(sur[3][0])
                surfaces_4_eval.append(sur[3][1])
            else:
                surfaces_4.append('')
                surfaces_4_eval.append('')
            if len(
                    sur) > 4 and part != 'test_seen':  # exclude test_seen because none of the triples in the seen split have more than 4 lexicalizations so this list would be empty
                surfaces_5.append(sur[4][0])
                surfaces_5_eval.append(sur[4][1])
            else:
                surfaces_5.append('')
                surfaces_5_eval.append('')
    else:
        surfaces.append(sur[0])
        surfaces_eval.append(sur[1])

    surface_lists = list(zip(surfaces, surfaces_2, surfaces_3, surfaces_4, surfaces_5))
    surface_eval_lists = list(zip(surfaces_eval, surfaces_2_eval, surfaces_3_eval, surfaces_4_eval, surfaces_5_eval))

    return surface_lists, surface_eval_lists, nodes, count
'''

train_cat = set()
dataset_points = {}
for split in tqdm(datasets, desc='Datasets'):  # Datasets list: ['train', 'dev', 'test_both', 'test_seen', 'test_unseen']
    total_pred_dict = {}  # Dictionary where the keys are predicates and the values are a dictionary with the keys are 'triple' and 'lex' containing triples and lexicalizations for that predicate
    print(f'Dataset split: {split}')
    cont_all = 0
    datapoints = []
    all_cats = set()
    # all_cats_list = []
    if 'test' in split:
        split_set = 'test'
        files = [folder + '/' + split_set + '/rdf-to-text-generation-test-data-with-refs-en.xml']
    else:
        files = Path(folder + '/' + split).rglob('*.xml')

    files = sorted(list(files))

    for idx, filename in enumerate(files):
        filename = str(filename)

        if split == 'train':
            datapoint, cats, cont = get_data(filename)
            # total_pred_dict = dict_merge(total_pred_dict, pred_dict)
        else:
            datapoint, cats, cont = get_data_dev_test(filename, train_cat, split)
            # total_pred_dict = dict_merge(total_pred_dict, pred_dict)

        cont_all += cont

        all_cats.update(cats)
        for data in datapoint:
            datapoints.append(data)
    print(f'DATAPOINT: {datapoints[0]}')

    if split == 'train':
        train_cat = all_cats
        print(f'Number of {split} texts/lexicalizations: {len(datapoints)}')
        datapoints = find_analogies(datapoints, split=split)
        '''
        Datapoints list organization for 'train': List of tuples in which each tuple contains two items:
            - The first item is a list in which each item is a token in the triple that can be combined into a string using the join method
            - The second item is a tuple containing two items: the lexicalization written normally for the target file and 
              the lexicalization in all lowercase for the target evaluation files
        - The number of items in the train datapoints list is the number of lexicalizations/texts, 
          but will NOT be the number of UNIQUE triples (since each triple may have multiple lexicalizations)
        '''
    else:
        print(f'Number of {split} texts/lexicalizations: {sum([len(datapoints[num]["lex"]) for num in range(len(datapoints))])}')
        datapoints = find_analogies(datapoints, split=split)
        '''
        Datapoints list organization for all other dataset splits: List of tuples in which each tuple contains two items:
            - The first item is a list in which each item is a token of a triple, just like the train datapoints list
            - The second item is a LIST of tuples in which each tuple contains the same two items as the train datapoints list:
              the lexicalization written normally for the target file and the lexicalization in all lowercase for the target evaluation files
        - So in order to get the number of texts/lexicalizations for these datapoints lists we need to look at the length 
          of item [x][1] in each tuple (which is the list of lexicalizations for each triple)
        - Here the number of items in datapoints will NOT give us the number of lexicalizations/texts,
          but it WILL give us the number of UNIQUE triples
        '''
    print('cont', cont_all)
    print('len cat', len(all_cats))
    print('cat', all_cats)

    dataset_points[split] = datapoints


# This section is to update test_both so it is the combination of test_seen and test_unseen (and thus contains the identical analogies)
dataset_points['test_both'] = dataset_points['test_seen'] + dataset_points['test_unseen']
ana_count = len([datapoint for datapoint in dataset_points['test_both'] if 'ana_input' in datapoint.keys()])
print(f'{ana_count}/{len(dataset_points["test_both"])} total REVISED TEST BOTH analogies')


path = os.path.dirname(
    os.path.realpath(__file__)) + '/webnlg_prep_ANA_ONELEX_OG/'  # Change file name back when experimenting is over
if not os.path.exists(path):
    os.makedirs(path)

os.system("rm " + path + '/*')

for idx, datapoint_split in enumerate(dataset_points):

    part = datasets[idx]

    if part == 'dev':
        part = 'val'

    nodes = []

    surfaces = []
    surfaces_2 = []
    surfaces_3 = []
    surfaces_4 = []
    surfaces_5 = []

    surfaces_eval = []
    surfaces_2_eval = []
    surfaces_3_eval = []
    surfaces_4_eval = []
    surfaces_5_eval = []

    # This doesn't work, cannot zip empty lists; work in progress
    #surface_lists = list(zip(surfaces, surfaces_2, surfaces_3, surfaces_4, surfaces_5))
    #surface_eval_lists = list(zip(surfaces_eval, surfaces_2_eval, surfaces_3_eval, surfaces_4_eval, surfaces_5_eval))

    overlap_scores = []

    total_data_count = 0
    cats_list = []
    for datapoint in dataset_points[datapoint_split]:
        '''
        #Here the source triples and the lexicalizations are being organized into their respective files
        #- Source triples are placed into the .source file

        #Note that lexicalizations are split into three files
        #- At the very least every source should have at least one lexicalization which will be located in the default .target[_eval] files
        #- If the source has 2 or 3 lexicalizations they will be placed in the .target2[_eval] and .target3[_eval] files respectively

        #*My explanation of the datapoints lists in lines 189 and 199 can help you understand the indexing using below
         #to organize these files

         Datapoint list indexing:
         - 0: list of list of triples
         - 1: tuple of lexicalizations
            - if train split -> tuple of 2 lexicalizations (normal, lowercase)
            - if test or dev split -> list of these couplet tuples
        - 2: list of subjects for each triple in datapoint[0]
        - 3: list of predicates for each triple in datapoint[0]
        - 4: list containing the sub_sim_score and pred_sim_score for each analogy [Scores are set to 0 if no analogy is found]
        - 5: category of each input triple
        - 6: new input with analogy triple/graph & lexicalizations along with the original triple (g1 <L> s1 <<G>> g2) [ONLY IF AN ANALOGY IS FOUND]

        '''

        sur = datapoint['lex']
        if 'ana_input' in datapoint.keys():  # We only want to include data for which analogies were found
            node = datapoint['ana_input']
            total_data_count += 1
            nodes.append(' '.join(node))
            overlap_scores.append(datapoint['sim_scores'])
            cats_list.append(datapoint['category'])
            if part != 'train':
                surfaces.append(sur[0][0])
                surfaces_eval.append(sur[0][1])
                if len(sur) > 1:
                    surfaces_2.append(sur[1][0])
                    surfaces_2_eval.append(sur[1][1])
                else:
                    surfaces_2.append('')
                    surfaces_2_eval.append('')
                if len(sur) > 2:
                    surfaces_3.append(sur[2][0])
                    surfaces_3_eval.append(sur[2][1])
                else:
                    surfaces_3.append('')
                    surfaces_3_eval.append('')
                if 'test' in part and OG is False:  # only continue on to these steps for test set
                    if len(sur) > 3:
                        surfaces_4.append(sur[3][0])
                        surfaces_4_eval.append(sur[3][1])
                    else:
                        surfaces_4.append('')
                        surfaces_4_eval.append('')
                    if len(sur) > 4 and part != 'test_seen':  # exclude test_seen because none of the triples in the seen split have more than 4 lexicalizations so this list would be empty
                        surfaces_5.append(sur[4][0])
                        surfaces_5_eval.append(sur[4][1])
                    else:
                        surfaces_5.append('')
                        surfaces_5_eval.append('')
            else:
                surfaces.append(sur[0])
                surfaces_eval.append(sur[1])

    if 'test' in part and OG is False:
        target1_name = '.target1'
    else:
        target1_name = '.target'
    with open(path + '/' + part + target1_name, 'w', encoding='utf8') as f:
        f.write('\n'.join(surfaces))
        f.write('\n')
    if part != 'train':
        with open(path + '/' + part + '.target2', 'w', encoding='utf8') as f:
            f.write('\n'.join(surfaces_2))
            f.write('\n')
        with open(path + '/' + part + '.target3', 'w', encoding='utf8') as f:
            f.write('\n'.join(surfaces_3))
            f.write('\n')
    if 'test' in part and OG is False:  # only do for test set
        with open(path + '/' + part + '.target4', 'w', encoding='utf8') as f:
            f.write('\n'.join(surfaces_4))
            f.write('\n')
        if part != 'test_seen':  # exclude test_seen because none of the triples in this split have more than 4 lexicalizations so this file would be empty
            with open(path + '/' + part + '.target5', 'w', encoding='utf8') as f:
                f.write('\n'.join(surfaces_5))
                f.write('\n')
    if 'test' in part and OG is False:
        target1_eval_name = '.target1_eval'
    else:
        target1_eval_name = '.target_eval'
    with open(path + '/' + part + target1_eval_name, 'w', encoding='utf8') as f:
        f.write('\n'.join(surfaces_eval))
        f.write('\n')
    if part != 'train':
        with open(path + '/' + part + '.target2_eval', 'w', encoding='utf8') as f:
            f.write('\n'.join(surfaces_2_eval))
            f.write('\n')
        with open(path + '/' + part + '.target3_eval', 'w', encoding='utf8') as f:
            f.write('\n'.join(surfaces_3_eval))
            f.write('\n')
        if OG:
            path_c = os.path.dirname(os.path.realpath(__file__))
            os.system("python " + path_c + '/' + "convert_files_crf.py " + path + '/' + part)
            os.system("python " + path_c + '/' + "convert_files_meteor.py " + path + '/' + part)

    if 'test' in part and OG is False:  # only do for test set
        with open(path + '/' + part + '.target4_eval', 'w', encoding='utf8') as f:
            f.write('\n'.join(surfaces_4_eval))
            f.write('\n')
        if part != 'test_seen':  # exclude test_seen because none of the triples in this split have more than 4 lexicalizations so this file would be empty
            with open(path + '/' + part + '.target5_eval', 'w', encoding='utf8') as f:
                f.write('\n'.join(surfaces_5_eval))
                f.write('\n')


    labels = ['Subject overlap', 'Predicate overlap']
    with open(path + '/' + part + '.overlap_scores.csv', 'w', encoding='utf8') as f:
        write = csv.writer(f)
        write.writerow(labels)
        for score in overlap_scores:
            write.writerow(score)

    print(f'{part} datapoints: {total_data_count}')
    print(f'{part} cats check: {len(cats_list)}')  # Verifying that the length of the categories list is the same as the number of datapoints

    # Creating numpy arrays for the sources and categories
    if part != 'test_unseen':
        cats_dict = defaultdict(lambda: len(cats_dict))
        cats_map = [cats_dict[cat_] for cat_ in cats_list]
        cats_map = np.array(cats_map).astype(int)
    else:
        cats_map = np.array([16 for x in range(len(cats_list))]).astype(int)
    source_array = np.ones(len(cats_list)).astype(int)

    with open(path + '/' + part + '.source', 'w', encoding='utf8') as f:
        f.write('\n'.join(nodes))
        f.write('\n')

    np_source_path = path + '/' + part + '.source.npy'
    np.save(np_source_path, source_array)

    np_cats_path = path + '/' + part + '.source_cat.npy'
    np.save(np_cats_path, cats_map)

print('Preprocessing Finished')
print(f'Data located in {path}')