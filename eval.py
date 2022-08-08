__author__='thiagocastroferreira'

"""
Author: Organizers of the 2nd WebNLG Challenge
Date: 23/04/2020
Description:
    This script aims to evaluate the output of data-to-text NLG models by 
    computing popular automatic metrics such as BLEU (two implementations), 
    METEOR, chrF++, TER and BERT-Score.
    
    ARGS:
        usage: eval.py [-h] -R REFERENCE -H prediction [-lang LANGUAGE] [-nr NUM_REFS]
               [-m METRICS] [-nc NCORDER] [-nw NWORDER] [-b BETA]

        optional arguments:
          -h, --help            show this help message and exit
          -R REFERENCE, --reference REFERENCE
                                reference translation
          -H PREDICTION, --prediction PREDICTION
                                prediction translation
          -lang LANGUAGE, --language LANGUAGE
                                evaluated language
          -nr NUM_REFS, --num_refs NUM_REFS
                                number of references
          -m METRICS, --metrics METRICS
                                evaluation metrics to be computed
          -nc NCORDER, --ncorder NCORDER
                                chrF metric: character n-gram order (default=6)
          -nw NWORDER, --nworder NWORDER
                                chrF metric: word n-gram order (default=2)
          -b BETA, --beta BETA  chrF metric: beta parameter (default=2)

    EXAMPLE:
        ENGLISH: 
            python3 eval.py -R data/en/references/reference -H data/en/prediction -nr 4 -m bleu,meteor,chrf++,ter,bert,bleurt
        RUSSIAN:
            python3 eval.py -R data/ru/reference -H data/ru/prediction -lang ru -nr 1 -m bleu,meteor,chrf++,ter,bert
"""
import sys
import argparse
import codecs
import copy
import os
import pyter
import nltk
import subprocess
import re

#from metrics.bleurt.bleurt import score as bleurt_score
from metrics.chrF import computeChrF
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from razdel import tokenize
from tabulate import tabulate
from statistics import mean
import evaluate


BLEU_PATH = 'metrics/multi-bleu-detok.perl'
METEOR_PATH = 'metrics/meteor-1.5/meteor-1.5.jar'

bertscore = evaluate.load('bertscore')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", help="path to references file", type=str, required=True)
    parser.add_argument("--predictions", help="path to predictions file", type=str, required=True)
    parser.add_argument("--target-file", help="base name of target file in references path [(test_both | test_seen | test_unseen) + (.target | .target_eval)]", type=str, required=True)
    parser.add_argument("--metrics", help="evaluation metrics to be computed", type=str, default='bleu,meteor,bert')
    parser.add_argument("--language", help="evaluated language", type=str, default='en')
    parser.add_argument("--num-refs", help="maximum number of references in dataset split (same as number of reference files for that split) [default=4]", type=int, default=4)
    parser.add_argument("--ncorder", help="chrF metric: character n-gram order [default=6]", type=int, default=6)
    parser.add_argument("--nworder", help="chrF metric: word n-gram order [default=2]", type=int, default=2)
    parser.add_argument("--beta", help="chrF metric: beta parameter [default=2]", type=float, default=2.0)
    args = parser.parse_args()
    print(f'Evaluation references: {args.target_file}')
    print(f'Evaluation predictions: {args.predictions}')
    return args


def format_data(args):
    '''
    EDIT THIS FUNCTION SO THAT IT FORMATS DATA DIFFERENTLY ONLY FOR BERT SCORE SINCE I USE A DIFFERENT METHOD,
    OR SEE IF OLD METHOD WORKS FINE
    '''

    print('PARSING INPUTS...')

    # references
    references = []
    for i in range(args.num_refs):
        fname = args.references + args.target_file + str(i + 1) if args.num_refs > 1 else args.references
        with codecs.open(fname, 'r', 'utf-8') as f:
            texts = f.read().split('\n')
            for j, text in enumerate(texts):
                if len(references) <= j:
                    references.append([text])
                else:
                    references[j].append(text)

    # references tokenized
    references_tok = copy.copy(references)
    for i, refs in enumerate(references_tok):
        if args.language == 'ru':
            references_tok[i] = [' '.join([_.text for _ in tokenize(ref)]) for ref in refs]
        else:
            references_tok[i] = [' '.join(nltk.word_tokenize(ref)) for ref in refs]

    # prediction
    with codecs.open(args.predictions, 'r', 'utf-8') as f:
        predictions = f.read().split('\n')

    # These two commands below will remove the last item in each list which is empty because there is a blank line at the end of each file
    predictions = predictions[:-1]
    references = references[:-1]

    # predictions tokenized
    predictions_tok = copy.copy(predictions)
    if args.language == 'ru':
        predictions_tok = [' '.join([_.text for _ in tokenize(hyp)]) for hyp in predictions_tok]
    else:
        predictions_tok = [' '.join(nltk.word_tokenize(hyp)) for hyp in predictions_tok]

    print('FINISHING PARSING INPUTS...')

    return references, references_tok, predictions, predictions_tok


def run(args):
    metrics = args.metrics.lower().split(',')
    references, references_tok, predictions, predictions_tok = format_data(args)

    result = {}
    print('STARTING EVALUATION...')
    if 'bleu' in metrics:
        bleu = bleu_score(args.references, args.predictions, args.target_file)
        result['BLEU'] = bleu

        b = bleu_nltk(references_tok, predictions_tok)
        result['BLEU_NLTK'] = b
    if 'meteor' in metrics:
        meteor = meteor_score(references, predictions, args.num_refs)
        result['METEOR'] = meteor
    if 'chrf++' in metrics:
        chrf, _, _, _ = chrF_score(references, predictions, args.num_refs, args.nworder, args.ncorder, args.beta)
        result['chrf++'] = chrf
    if 'ter' in metrics:
        ter = ter_score(references_tok, predictions_tok, args.num_refs)
        result['TER'] = ter
    if 'bert' in metrics:
        P, R, F1 = bert_score_(references, predictions, lang=args.language)
        result['BERT PRECISION'] = P
        result['BERT RECALL'] = R
        result['BERT F1'] = F1
    if 'bleurt' in metrics and args.language == 'en':
        s = bleurt(references, predictions, args.num_refs)
        result['BLEURT'] = s
    print('FINISHING EVALUATION')

    return result


def bleu_score(references, predictions, target_file):
    print('COMPUTING BLEU...')
    ref_files = []
    for i in range(5):

        ref_files.append(references + target_file + str(i + 1))

    command = 'perl {0} {1} < {2}'.format(BLEU_PATH, ' '.join(ref_files), predictions)
    result = subprocess.check_output(command, shell=True)
    try:
        bleu_results = float(re.findall('BLEU = (.+?),', str(result))[0])
    except:
        print('ERROR COMPUTING BLEU.')
        bleu_results = -1

    print('FINISHING COMPUTING BLEU...')
    return bleu_results


def meteor_score(references, prediction, num_refs, lng='en'):
    print('COMPUTING METEOR...')
    preds_tmp = 'meteor_hypothesis.txt'
    refs_tmp = 'meteor_references.txt'
    print(f'References file: {refs_tmp}')
    print(f'References length: {len(references)}')
    print(f'Predictions file: {preds_tmp}')
    print(f'Predictions length: {len(prediction)}')
    with codecs.open(preds_tmp, 'w', 'utf-8') as f:
        f.write('\n'.join(prediction))

    linear_references = []
    for refs in references:
        for i in range(num_refs):
            linear_references.append(refs[i])

    with codecs.open(refs_tmp, 'w', 'utf-8') as f:
        f.write('\n'.join(linear_references))

    try:
        command = f'java -Xmx2G -jar {METEOR_PATH} '
        command += f'{preds_tmp} {refs_tmp} -l {lng} -norm -r {num_refs}'
        result = subprocess.check_output(command, shell=True)
        meteor = result.split(b'\n')[-2].split()[-1]
    except:
        print('ERROR COMPUTING METEOR.')
        meteor = -1

    try:
        os.remove(preds_tmp)
        os.remove(refs_tmp)
    except:
        pass
    print('FINISHING COMPUTING METEOR...')
    return float(meteor)


def bert_score_(references, predictions, lang):
    print('COMPUTING BERT SCORE...')
    for i, refs in enumerate(references):
        references[i] = [ref for ref in refs if ref.strip() != '']

    bertscore_results = bertscore.compute(predictions=predictions, references=references, lang=lang)

    mean_precision = float(mean(bertscore_results['precision']))
    mean_recall = float(mean(bertscore_results['recall']))
    mean_f1 = float(mean(bertscore_results['f1']))

    print('FINISHING COMPUTING BERT SCORE...')
    return mean_precision, mean_recall, mean_f1


def bleu_nltk(references, prediction):
    # check for empty lists
    references_, prediction_ = [], []
    for i, refs in enumerate(references):
        refs_ = [ref for ref in refs if ref.strip() != '']
        if len(refs_) > 0:
            references_.append([ref.split() for ref in refs_])
            prediction_.append(prediction[i].split())

    chencherry = SmoothingFunction()
    return corpus_bleu(references_, prediction_, smoothing_function=chencherry.method3)


def chrF_score(references, prediction, num_refs, nworder, ncorder, beta):
    print('COMPUTING CHRF++...')
    hyps_tmp, refs_tmp = 'prediction_chrF', 'reference_chrF'

    # check for empty lists
    references_, prediction_ = [], []
    for i, refs in enumerate(references):
        refs_ = [ref for ref in refs if ref.strip() != '']
        if len(refs_) > 0:
            references_.append(refs_)
            prediction_.append(prediction[i])

    with codecs.open(hyps_tmp, 'w', 'utf-8') as f:
        f.write('\n'.join(prediction_))

    linear_references = []
    for refs in references_:
        linear_references.append('*#'.join(refs[:num_refs]))

    with codecs.open(refs_tmp, 'w', 'utf-8') as f:
        f.write('\n'.join(linear_references))

    rtxt = codecs.open(refs_tmp, 'r', 'utf-8')
    htxt = codecs.open(hyps_tmp, 'r', 'utf-8')

    try:
        totalF, averageTotalF, totalPrec, totalRec = computeChrF(rtxt, htxt, nworder, ncorder, beta, None)
    except:
        print('ERROR COMPUTING CHRF++.')
        totalF, averageTotalF, totalPrec, totalRec = -1, -1, -1, -1
    try:
        os.remove(hyps_tmp)
        os.remove(refs_tmp)
    except:
        pass
    print('FINISHING COMPUTING CHRF++...')
    return totalF, averageTotalF, totalPrec, totalRec


def ter_score(references, prediction, num_refs):
    print('COMPUTING TER...')
    ter_scores = []
    for hyp, refs in zip(prediction, references):
        candidates = []
        for ref in refs[:num_refs]:
            if len(ref) == 0:
                ter_score = 1
            else:
                try:
                    ter_score = pyter.ter(hyp.split(), ref.split())
                except:
                    ter_score = 1
            candidates.append(ter_score)

        ter_scores.append(min(candidates))

    print('FINISHING COMPUTING TER...')
    return sum(ter_scores) / len(ter_scores)


def bleurt(references, prediction, num_refs, checkpoint = "metrics/bleurt/bleurt-base-128"):
    refs, cands = [], []
    for i, hyp in enumerate(prediction):
        for ref in references[i][:num_refs]:
            cands.append(hyp)
            refs.append(ref)

    scorer = bleurt_score.BleurtScorer(checkpoint)
    scores = scorer.score(refs, cands)
    scores = [max(scores[i:i+num_refs]) for i in range(0, len(scores), num_refs)]
    return round(sum(scores) / len(scores), 2)


if __name__ == '__main__':

    final_results = run(parse_args())

    for result in final_results:
        print(f'{result}: {final_results[result]}')