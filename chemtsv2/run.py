import argparse
from logging import getLogger, StreamHandler, FileHandler, Formatter, INFO, DEBUG
from importlib import import_module
import os
import sys
sys.path.append(os.getcwd())
import pickle
import re
import requests
import yaml

from numpy.random import default_rng
from rdkit import RDLogger

from chemtsv2.mcts import MCTS, State
from chemtsv2.utils import loaded_model, get_model_structure_info
from chemtsv2.preprocessing import smi_tokenizer


def get_parser():
    parser = argparse.ArgumentParser(
        description="",
        usage=f"python {os.path.basename(__file__)} -c CONFIG_FILE"
    )
    parser.add_argument(
        "-c", "--config", type=str, required=True,
        help="path to a config file"
    )
    parser.add_argument(
        "-d", "--debug", action='store_true',
        help="debug mode"
    )
    parser.add_argument(
        "-g", "--gpu", type=str,
        help="constrain gpu. (e.g. 0,1)"
    )
    parser.add_argument(
        "--input_smiles", type=str,
        help="SMILES string (Need to put the atom you want to extend at the end of the string)"
    )
    return parser.parse_args()


def get_logger(level, save_dir):#创建一个日志记录器
    logger = getLogger(__name__)
    logger.setLevel(level)
    logger.propagate = False

    formatter = Formatter("%(asctime)s : %(levelname)s : %(message)s ")

    fh = FileHandler(filename=os.path.join(save_dir, "run.log"), mode='w')
    fh.setLevel(level)
    fh.setFormatter(formatter)
    sh = StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def set_default_config(conf):#将conf文件中的参数传进来，如果文件中没有设置的则采用下列设置好的默认值
    conf.setdefault('c_val', 1.0)#如果字典中不存在键为'c_val'的项，则将其设置为默认值1.0
    conf.setdefault('threshold_type', 'time')
    conf.setdefault('hours', 1) 
    conf.setdefault('generation_num', 1000)
    conf.setdefault('simulation_num', 3)
    conf.setdefault('expansion_threshold', 0.995)
    conf.setdefault('flush_threshold', -1)
    conf.setdefault('infinite_loop_threshold_for_selection', 1000)
    conf.setdefault('infinite_loop_threshold_for_expansion', 20)
    conf.setdefault('fix_random_seed', False)

    conf.setdefault('use_lipinski_filter', False)
    conf.setdefault('lipinski_filter', {
        'module': 'filter.lipinski_filter',
        'class': 'LipinskiFilter',
        'type': 'rule_of_5'})
    conf.setdefault('use_radical_filter', False)
    conf.setdefault('radical_filter', {
        'module': 'filter.radical_filter',
        'class': 'RadicalFilter'})
    conf.setdefault('use_pubchem_filter', False) 
    conf.setdefault('pubchem_filter', {
        'module': 'filter.pubchem_filter',
        'class': 'PubchemFilter'}) 
    conf.setdefault('use_sascore_filter', False)
    conf.setdefault('sascore_filter', {
        'module': 'filter.sascore_filter',
        'class': 'SascoreFilter',
        'threshold': 3.5})
    conf.setdefault('use_ring_size_filter', False)
    conf.setdefault('ring_size_filter', {
        'module': 'filter.ring_size_filter',
        'class': 'RingSizeFilter',
        'threshold': 6})
    conf.setdefault('use_pains_filter', False)
    conf.setdefault('pains_filter', {
        'module': 'filter.pains_filter',
        'class': 'PainsFilter',
        'type': ['pains_a']})
    conf.setdefault('include_filter_result_in_reward', False)

    conf.setdefault('model_setting', {
        'model_json': 'model/model.tf25.json',
        'model_weight': 'model/model.tf25.best.ckpt.h5'})
    conf.setdefault('output_dir', 'result')
    conf.setdefault('reward_setting', {
        'reward_module': 'reward.logP_reward',
        'reward_class': 'LogP_reward'})
    conf.setdefault('batch_reward_calculation', False)
    conf.setdefault('policy_setting', {
        'policy_module': 'policy.ucb1',
        'policy_class': 'Ucb1'})
    conf.setdefault('token', 'model/tokens.pkl')

    conf.setdefault('leaf_parallel', False)
    conf.setdefault('leaf_parallel_num', 4)
    conf.setdefault('qsub_parallel', False)
    
    conf.setdefault('save_checkpoint', False)
    conf.setdefault('restart', False)
    conf.setdefault('checkpoint_file', "chemtsv2.ckpt.pkl")

    conf.setdefault('neutralization', False)
    
    

def get_filter_modules(conf):
    pat = re.compile(r'^use.*filter$')
    module_list = []
    for k, frag in conf.items():#这段代码的作用是遍历conf字典中的所有键值对，并将键赋值给k，将值赋值给frag
        if not pat.search(k) or frag != True:
            continue
        _k = k.replace('use_', '')
        module_list.append(getattr(import_module(conf[_k]['module']), conf[_k]['class']))
    return module_list


def main():
    args = get_parser()
    with open(args.config, "r") as f:#这段代码是用来打开一个文件的，其中args.config是文件的路径。打开文件后，可以对文件进行读取操作。使用with语句可以确保文件在使用完后自动关闭，避免资源浪费和文件损坏的风险
        conf = yaml.load(f, Loader=yaml.SafeLoader)
    set_default_config(conf)#将conf文件中的参数传进来
    os.makedirs(conf['output_dir'], exist_ok=True)#这段代码是用来创建一个目录的。其中，conf['output_dir']是目录的路径，exist_ok=True表示如果目录已经存在，则不会抛出异常。也就是说，如果目录不存在，则会创建该目录。
    os.environ['CUDA_VISIBLE_DEVICES'] = "-1" if args.gpu is None else args.gpu3#这段代码的作用是设置CUDA可见设备的环境变量。如果args.gpu为None，则将CUDA_VISIBLE_DEVICES设置为-1，表示不使用GPU；否则将其设置为args.gpu指定的GPU设备。这通常用于在使用GPU时控制使用哪些设备。

    # set log level 设置日志级别
    conf["debug"] = args.debug
    log_level = DEBUG if args.debug else INFO#这段代码是一个条件语句，它的作用是根据传入的参数来设置日志的级别。如果传入的参数中包含了 debug，则将日志级别设置为 DEBUG，否则将日志级别设置为 INFO。这样可以在调试时输出更详细的日志信息，而在正式运行时则只输出必要的信息。
    logger = get_logger(log_level, conf["output_dir"])#这段代码的作用是获取一个日志记录器，并设置日志级别和输出目录。其中，log_level是日志级别，conf是一个配置对象，包含了输出目录等信息。get_logger是一个函数，用于获取日志记录器。
    if not args.debug:
        RDLogger.DisableLog("rdApp.*")

    if args.debug:
        conf['fix_random_seed'] = True

    # download additional data if files don't exist下载附加信息
    if not os.path.exists('data/sascorer.py'):
        url = 'https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score/sascorer.py'
        with open('data/sascorer.py', 'w') as f:
            f.write(requests.get(url).text)
    if not os.path.exists('data/fpscores.pkl.gz'):
        url = 'https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score/fpscores.pkl.gz'
        with open('data/fpscores.pkl.gz', 'wb') as f:
            f.write(requests.get(url).content)
    
    rs = conf['reward_setting']
    reward_calculator = getattr(import_module(rs["reward_module"]), rs["reward_class"])
    ps = conf['policy_setting']
    policy_evaluator = getattr(import_module(ps['policy_module']), ps['policy_class'])
    conf['max_len'], conf['rnn_vocab_size'], conf['rnn_output_size'] = get_model_structure_info(conf['model_setting']['model_json'], logger)
    model = loaded_model(conf['model_setting']['model_weight'], logger, conf)  #WM300 not tested  
    if args.input_smiles is not None:
        logger.info(f"Extend mode: input SMILES = {args.input_smiles}")#在日志中记录
        conf["input_smiles"] = args.input_smiles
        conf["tokenized_smiles"] = smi_tokenizer(conf["input_smiles"])

    if conf['threshold_type'] == 'time':  # To avoid user confusion避免用户混淆
        conf.pop('generation_num')#删除generation_num
    elif conf['threshold_type'] == 'generation_num':
        conf.pop('hours')

    logger.info(f"========== Configuration ==========")
    for k, v in conf.items():
        logger.info(f"{k}: {v}")
    logger.info(f"GPU devices: {os.environ['CUDA_VISIBLE_DEVICES']}")
    logger.info(f"===================================")
            
    conf['filter_list'] = get_filter_modules(conf)

    conf['random_generator'] = default_rng(1234) if conf['fix_random_seed'] else default_rng()#这段代码的作用是根据配置文件中的参数来选择是否使用固定的随机种子生成器。如果配置文件中的参数"fix_random_seed"为True，则使用固定的随机种子生成器，并将种子设置为1234；否则使用默认的随机种子生成器。这个功能通常用于在调试和测试时保证随机数的可重复性。

    with open(conf['token'], 'rb') as f:
        tokens = pickle.load(f)
    logger.debug(f"Loaded tokens are {tokens}")
    state = State() if args.input_smiles is None else State(position=conf["tokenized_smiles"])
    mcts = MCTS(root_state=state, conf=conf, tokens=tokens, model=model, reward_calculator=reward_calculator, policy_evaluator=policy_evaluator, logger=logger)
    mcts.search()
    logger.info("Finished!")


if __name__ == "__main__":
    main()
