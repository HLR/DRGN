import os
import argparse
from multiprocessing import cpu_count
from utils.convert_csqa import convert_to_entailment
from utils.convert_obqa import convert_to_obqa_statement
from utils.conceptnet import extract_english, construct_graph
from utils.grounding import create_matcher_patterns, ground
from utils.graph import generate_adj_data_from_grounded_concepts__use_LM

input_paths = {
    'csqa': {
        'train': '/localscratch/chen.zheng/data/drgn/data/csqa/train_rand_split.jsonl',
        'dev': '/localscratch/chen.zheng/data/drgn/data/csqa/dev_rand_split.jsonl',
        'test': '/localscratch/chen.zheng/data/drgn/data/csqa/test_rand_split_no_answers.jsonl',
    },
    'obqa': {
        'train': '/localscratch/chen.zheng/data/drgn/data/obqa/OpenBookQA-V1-Sep2018/Data/Main/train.jsonl',
        'dev': '/localscratch/chen.zheng/data/drgn/data/obqa/OpenBookQA-V1-Sep2018/Data/Main/dev.jsonl',
        'test': '/localscratch/chen.zheng/data/drgn/data/obqa/OpenBookQA-V1-Sep2018/Data/Main/test.jsonl',
    },
    'obqa-fact': {
        'train': '/localscratch/chen.zheng/data/drgn/data/obqa/OpenBookQA-V1-Sep2018/Data/Additional/train_complete.jsonl',
        'dev': '/localscratch/chen.zheng/data/drgn/data/obqa/OpenBookQA-V1-Sep2018/Data/Additional/dev_complete.jsonl',
        'test': '/localscratch/chen.zheng/data/drgn/data/obqa/OpenBookQA-V1-Sep2018/Data/Additional/test_complete.jsonl',
    },
    'cpnet': {
        'csv': '/localscratch/chen.zheng/data/drgn/data/cpnet/conceptnet-assertions-5.6.0.csv',
    },
}

output_paths = {
    'cpnet': {
        'csv': '/localscratch/chen.zheng/data/drgn/data/cpnet/conceptnet.en.csv',
        'vocab': '/localscratch/chen.zheng/data/drgn/data/cpnet/concept.txt',
        'patterns': '/localscratch/chen.zheng/data/drgn/data/cpnet/matcher_patterns.json',
        'unpruned-graph': '/localscratch/chen.zheng/data/drgn/data/cpnet/conceptnet.en.unpruned.graph',
        'pruned-graph': '/localscratch/chen.zheng/data/drgn/data/cpnet/conceptnet.en.pruned.graph',
    },
    'csqa': {
        'statement': {
            'train': '/localscratch/chen.zheng/data/drgn/data/csqa/statement/train.statement.jsonl',
            'dev': '/localscratch/chen.zheng/data/drgn/data/csqa/statement/dev.statement.jsonl',
            'test': '/localscratch/chen.zheng/data/drgn/data/csqa/statement/test.statement.jsonl',
        },
        'grounded': {
            'train': '/localscratch/chen.zheng/data/drgn/data/csqa/grounded/train.grounded.jsonl',
            'dev': '/localscratch/chen.zheng/data/drgn/data/csqa/grounded/dev.grounded.jsonl',
            'test': '/localscratch/chen.zheng/data/drgn/data/csqa/grounded/test.grounded.jsonl',
        },
        'graph': {
            'adj-train': '/localscratch/chen.zheng/data/drgn/data/csqa/graph/train.graph.adj.pk',
            'adj-dev': '/localscratch/chen.zheng/data/drgn/data/csqa/graph/dev.graph.adj.pk',
            'adj-test': '/localscratch/chen.zheng/data/drgn/data/csqa/graph/test.graph.adj.pk',
        },
    },
    'obqa': {
        'statement': {
            'train': '/localscratch/chen.zheng/data/drgn/data/obqa/statement/train.statement.jsonl',
            'dev': '/localscratch/chen.zheng/data/drgn/data/obqa/statement/dev.statement.jsonl',
            'test': '/localscratch/chen.zheng/data/drgn/data/obqa/statement/test.statement.jsonl',
            'train-fairseq': '/localscratch/chen.zheng/data/drgn/data/obqa/fairseq/official/train.jsonl',
            'dev-fairseq': '/localscratch/chen.zheng/data/drgn/data/obqa/fairseq/official/valid.jsonl',
            'test-fairseq': '/localscratch/chen.zheng/data/drgn/data/obqa/fairseq/official/test.jsonl',
        },
        'grounded': {
            'train': '/localscratch/chen.zheng/data/drgn/data/obqa/grounded/train.grounded.jsonl',
            'dev': '/localscratch/chen.zheng/data/drgn/data/obqa/grounded/dev.grounded.jsonl',
            'test': '/localscratch/chen.zheng/data/drgn/data/obqa/grounded/test.grounded.jsonl',
        },
        'graph': {
            'adj-train': '/localscratch/chen.zheng/data/drgn/data/obqa/graph/train.graph.adj.pk',
            'adj-dev': '/localscratch/chen.zheng/data/drgn/data/obqa/graph/dev.graph.adj.pk',
            'adj-test': '/localscratch/chen.zheng/data/drgn/data/obqa/graph/test.graph.adj.pk',
        },
    },
    'obqa-fact': {
        'statement': {
            'train': '/localscratch/chen.zheng/data/drgn/data/obqa/statement/train-fact.statement.jsonl',
            'dev': '/localscratch/chen.zheng/data/drgn/data/obqa/statement/dev-fact.statement.jsonl',
            'test': '/localscratch/chen.zheng/data/drgn/data/obqa/statement/test-fact.statement.jsonl',
            'train-fairseq': '/localscratch/chen.zheng/data/drgn/data/obqa/fairseq/official/train-fact.jsonl',
            'dev-fairseq': '/localscratch/chen.zheng/data/drgn/data/obqa/fairseq/official/valid-fact.jsonl',
            'test-fairseq': '/localscratch/chen.zheng/data/drgn/data/obqa/fairseq/official/test-fact.jsonl',
        },
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=['common'], choices=['common', 'csqa', 'hswag', 'anli', 'exp', 'scitail', 'phys', 'socialiqa', 'obqa', 'obqa-fact', 'make_word_vocab'], nargs='+')
    parser.add_argument('--path_prune_threshold', type=float, default=0.12, help='threshold for pruning paths')
    parser.add_argument('--max_node_num', type=int, default=200, help='maximum number of nodes per graph')
    parser.add_argument('-p', '--nprocs', type=int, default=cpu_count(), help='number of processes to use')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')

    args = parser.parse_args()
    if args.debug:
        raise NotImplementedError()

    routines = {
        'common': [
            {'func': extract_english, 'args': (input_paths['cpnet']['csv'], output_paths['cpnet']['csv'], output_paths['cpnet']['vocab'])},
            {'func': construct_graph, 'args': (output_paths['cpnet']['csv'], output_paths['cpnet']['vocab'],
                                               output_paths['cpnet']['unpruned-graph'], False)},
            {'func': construct_graph, 'args': (output_paths['cpnet']['csv'], output_paths['cpnet']['vocab'],
                                               output_paths['cpnet']['pruned-graph'], True)},
            {'func': create_matcher_patterns, 'args': (output_paths['cpnet']['vocab'], output_paths['cpnet']['patterns'])},
        ],
        'csqa': [
            {'func': convert_to_entailment, 'args': (input_paths['csqa']['train'], output_paths['csqa']['statement']['train'])},
            {'func': convert_to_entailment, 'args': (input_paths['csqa']['dev'], output_paths['csqa']['statement']['dev'])},
            {'func': convert_to_entailment, 'args': (input_paths['csqa']['test'], output_paths['csqa']['statement']['test'])},
            {'func': ground, 'args': (output_paths['csqa']['statement']['train'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['csqa']['grounded']['train'], args.nprocs)},
            {'func': ground, 'args': (output_paths['csqa']['statement']['dev'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['csqa']['grounded']['dev'], args.nprocs)},
            {'func': ground, 'args': (output_paths['csqa']['statement']['test'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['csqa']['grounded']['test'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['csqa']['grounded']['train'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['csqa']['graph']['adj-train'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['csqa']['grounded']['dev'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['csqa']['graph']['adj-dev'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['csqa']['grounded']['test'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['csqa']['graph']['adj-test'], args.nprocs)},
        ],

        'obqa': [
            {'func': convert_to_obqa_statement, 'args': (input_paths['obqa']['train'], output_paths['obqa']['statement']['train'], output_paths['obqa']['statement']['train-fairseq'])},
            {'func': convert_to_obqa_statement, 'args': (input_paths['obqa']['dev'], output_paths['obqa']['statement']['dev'], output_paths['obqa']['statement']['dev-fairseq'])},
            {'func': convert_to_obqa_statement, 'args': (input_paths['obqa']['test'], output_paths['obqa']['statement']['test'], output_paths['obqa']['statement']['test-fairseq'])},
            {'func': ground, 'args': (output_paths['obqa']['statement']['train'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['obqa']['grounded']['train'], args.nprocs)},
            {'func': ground, 'args': (output_paths['obqa']['statement']['dev'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['obqa']['grounded']['dev'], args.nprocs)},
            {'func': ground, 'args': (output_paths['obqa']['statement']['test'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['obqa']['grounded']['test'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['obqa']['grounded']['train'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['obqa']['graph']['adj-train'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['obqa']['grounded']['dev'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['obqa']['graph']['adj-dev'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['obqa']['grounded']['test'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['obqa']['graph']['adj-test'], args.nprocs)},
        ],
    }

    for rt in args.run:
        for rt_dic in routines[rt]:
            rt_dic['func'](*rt_dic['args'])

    print('Successfully run {}'.format(' '.join(args.run)))


if __name__ == '__main__':
    main()
    # pass
