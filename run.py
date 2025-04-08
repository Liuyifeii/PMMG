import argparse
import os
import sys
import pickle
import re
import requests
import yaml

from logging import getLogger, StreamHandler, FileHandler, Formatter, INFO, DEBUG
from importlib import import_module
from numpy.random import default_rng
from rdkit import RDLogger

from chemtsv2.mcts import MCTS, State
from chemtsv2.utils import loaded_model, get_model_structure_info
from chemtsv2.preprocessing import smi_tokenizer

sys.path.append(os.getcwd())


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="ChemTSv2 Runner",
        usage=f"python {os.path.basename(__file__)} -c CONFIG_FILE"
    )
    parser.add_argument("-c", "--config", type=str, required=True, help="path to a config file")
    parser.add_argument("-d", "--debug", action='store_true', help="enable debug mode")
    parser.add_argument("-g", "--gpu", type=str, help="specify GPU(s), e.g., 0 or 0,1")
    parser.add_argument("--input_smiles", type=str, help="extend from this input SMILES")
    return parser.parse_args()


def initialize_logger(log_level, output_dir):
    logger = getLogger(__name__)
    logger.setLevel(log_level)
    logger.propagate = False

    formatter = Formatter("%(asctime)s : %(levelname)s : %(message)s ")

    log_file = os.path.join(output_dir, "run.log")
    file_handler = FileHandler(filename=log_file, mode='w')
    stream_handler = StreamHandler()

    for handler in [file_handler, stream_handler]:
        handler.setLevel(log_level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def apply_default_config(conf):
    default_params = {
        'c_val': 1.0,
        'threshold_type': 'time',
        'hours': 1,
        'generation_num': 1000,
        'simulation_num': 3,
        'expansion_threshold': 0.995,
        'flush_threshold': -1,
        'infinite_loop_threshold_for_selection': 1000,
        'infinite_loop_threshold_for_expansion': 20,
        'fix_random_seed': False,
        'use_lipinski_filter': False,
        'lipinski_filter': {
            'module': 'filter.lipinski_filter',
            'class': 'LipinskiFilter',
            'type': 'rule_of_5'},
        'use_radical_filter': False,
        'radical_filter': {
            'module': 'filter.radical_filter',
            'class': 'RadicalFilter'},
        'use_pubchem_filter': False,
        'pubchem_filter': {
            'module': 'filter.pubchem_filter',
            'class': 'PubchemFilter'},
        'use_sascore_filter': False,
        'sascore_filter': {
            'module': 'filter.sascore_filter',
            'class': 'SascoreFilter',
            'threshold': 3.5},
        'use_ring_size_filter': False,
        'ring_size_filter': {
            'module': 'filter.ring_size_filter',
            'class': 'RingSizeFilter',
            'threshold': 6},
        'use_pains_filter': False,
        'pains_filter': {
            'module': 'filter.pains_filter',
            'class': 'PainsFilter',
            'type': ['pains_a']},
        'include_filter_result_in_reward': False,
        'model_setting': {
            'model_json': 'model/model.tf25.json',
            'model_weight': 'model/model.tf25.best.ckpt.h5'},
        'output_dir': 'result',
        'reward_setting': {
            'reward_module': 'reward.logP_reward',
            'reward_class': 'LogP_reward'},
        'batch_reward_calculation': False,
        'policy_setting': {
            'policy_module': 'policy.ucb1',
            'policy_class': 'Ucb1'},
        'token': 'model/tokens.pkl',
        'leaf_parallel': False,
        'leaf_parallel_num': 4,
        'qsub_parallel': False,
        'save_checkpoint': False,
        'restart': False,
        'checkpoint_file': "chemtsv2.ckpt.pkl",
        'neutralization': False,
    }
    for key, value in default_params.items():
        conf.setdefault(key, value)


def get_enabled_filter_modules(conf):
    pattern = re.compile(r'^use.*filter$')
    modules = []
    for key, value in conf.items():
        if pattern.match(key) and value is True:
            module_key = key.replace('use_', '')
            module_info = conf[module_key]
            filter_class = getattr(import_module(module_info['module']), module_info['class'])
            modules.append(filter_class)
    return modules


def download_sascore_data():
    if not os.path.exists('data/sascorer.py'):
        url = 'https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score/sascorer.py'
        with open('data/sascorer.py', 'w') as f:
            f.write(requests.get(url).text)
    if not os.path.exists('data/fpscores.pkl.gz'):
        url = 'https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score/fpscores.pkl.gz'
        with open('data/fpscores.pkl.gz', 'wb') as f:
            f.write(requests.get(url).content)


def main():
    args = parse_arguments()

    with open(args.config, "r") as f:
        conf = yaml.load(f, Loader=yaml.SafeLoader)

    apply_default_config(conf)
    os.makedirs(conf['output_dir'], exist_ok=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = "-1" if args.gpu is None else args.gpu

    log_level = DEBUG if args.debug else INFO
    logger = initialize_logger(log_level, conf["output_dir"])

    if not args.debug:
        RDLogger.DisableLog("rdApp.*")

    conf["debug"] = args.debug
    if args.debug:
        conf["fix_random_seed"] = True

    download_sascore_data()

    reward_conf = conf['reward_setting']
    reward_calculator = getattr(import_module(reward_conf["reward_module"]), reward_conf["reward_class"])

    policy_conf = conf['policy_setting']
    policy_evaluator = getattr(import_module(policy_conf["policy_module"]), policy_conf["policy_class"])

    conf['max_len'], conf['rnn_vocab_size'], conf['rnn_output_size'] = get_model_structure_info(
        conf['model_setting']['model_json'], logger)
    model = loaded_model(conf['model_setting']['model_weight'], logger, conf)

    if args.input_smiles:
        logger.info(f"Extend mode: input SMILES = {args.input_smiles}")
        conf["input_smiles"] = args.input_smiles
        conf["tokenized_smiles"] = smi_tokenizer(args.input_smiles)

    if conf['threshold_type'] == 'time':
        conf.pop('generation_num', None)
    elif conf['threshold_type'] == 'generation_num':
        conf.pop('hours', None)

    logger.info(f"========== Configuration ==========")
    for k, v in conf.items():
        logger.info(f"{k}: {v}")
    logger.info(f"GPU devices: {os.environ['CUDA_VISIBLE_DEVICES']}")
    logger.info(f"===================================")

    conf['filter_list'] = get_enabled_filter_modules(conf)
    conf['random_generator'] = default_rng(1234) if conf['fix_random_seed'] else default_rng()

    with open(conf['token'], 'rb') as f:
        tokens = pickle.load(f)
    logger.debug(f"Loaded tokens are {tokens}")

    init_state = State(position=conf["tokenized_smiles"]) if args.input_smiles else State()
    mcts = MCTS(
        root_state=init_state,
        conf=conf,
        tokens=tokens,
        model=model,
        reward_calculator=reward_calculator,
        policy_evaluator=policy_evaluator,
        logger=logger
    )
    mcts.search()
    logger.info("Finished!")


if __name__ == "__main__":
    main()
