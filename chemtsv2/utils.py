import copy
from functools import wraps
import itertools
import sys
import time

import joblib
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Embedding, GRU
import numpy as np
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from chemtsv2.misc.manage_qsub_parallel import run_qsub_parallel


def calc_execution_time(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        elapsed_time = time.time() - start
        print(f"Execution time of {f.__name__}: {elapsed_time} sec")
        return result
    return wrapper


def expanded_node(model, state, val, logger, threshold=0.995):#基于给定的输入状态，根据预训练模型的预测结果，确定应该拓展哪些节点
    get_int = [val.index(state[j]) for j in range(len(state))]
    x = np.reshape(get_int, (1, len(get_int)))
    model.reset_states()
    preds = model.predict_on_batch(x)
    state_preds = np.squeeze(preds)  # the sum of state_pred is equal to 1
    sorted_idxs = np.argsort(state_preds)[::-1]
    sorted_preds = state_preds[sorted_idxs]
    for i, v in enumerate(itertools.accumulate(sorted_preds)):
        if v > threshold:
            i = i if i != 0 else 1  # return one index if the first prediction value exceeds the threshold.
            break 
    logger.debug(f"indices for expansion: {sorted_idxs[:i]}")
    return sorted_idxs[:i]


def node_to_add(all_nodes, val, logger):
    added_nodes = [val[all_nodes[i]] for i in range(len(all_nodes))]
    logger.debug(added_nodes)
    return added_nodes


def back_propagation(node, reward):#反向传播，根据奖励信号来更新网络中的权重
    while node != None:#如果他还有父节点
        node.update(reward)
        node = node.state.parent_node
          #def update(self, reward):
           #self.state.visits += 1
           #self.state.total_reward += reward

def chem_kn_simulation(model, state, val, conf):#这个函数用模型来预测下一个字符的可能性分布，并以此为依据生成序列
    # 初始化变量
    end = "\n"  # 用于标识生成结束的标记
    position = []  # 用于存储生成序列的位置信息
    position.extend(state)  # 将输入的初始状态添加到位置信息中
    get_int = [val.index(position[j]) for j in range(len(position))]  # 获取初始状态在val列表中的索引值，就是数字，代表他在这个列表中的位置
    x = np.reshape(get_int, (1, len(get_int)))  # 将获取的索引值转换为模型输入的形状
    model.reset_states()  # 重置模型的状态

    # 生成序列
    while not get_int[-1] == val.index(end):  # 当序列末尾不是结束标记时
        preds = model.predict_on_batch(x)  # 使用模型预测下一个字符的概率分布
        state_pred = np.squeeze(preds)  # 去除多余的维度
        next_int = conf['random_generator'].choice(range(len(state_pred)), p=state_pred)  # 根据概率选择下一个字符的索引
        get_int.append(next_int)  # 将选取的索引添加到生成序列中
        x = np.reshape([next_int], (1, 1))  # 将下一个字符的索引转换为模型输入的形状
        if len(get_int) > conf['max_len']:  # 如果生成的序列长度超过了最大长度限制
            break  # 结束序列生成
    return get_int  # 返回生成的整数序列



def build_smiles_from_tokens(all_posible, val):
    # 定义一个函数，接受两个参数：all_posible 和 val

    total_generated = all_posible
    # 将 all_posible 赋值给 total_generated 变量

    generate_tokens = [val[total_generated[j]] for j in range(len(total_generated) - 1)]
    # 生成一个列表 generate_tokens，其中的元素是通过索引从 val 中获取的值
    # 这里通过遍历 total_generated 中的索引来获取对应的 val 中的值
    # 由于 range(len(total_generated) - 1) 表示索引范围，-1 是为了避免超出索引范围

    generate_tokens.remove("&")
    # 从 generate_tokens 列表中移除 "&" 这个元素

    return ''.join(generate_tokens)
    # 将 generate_tokens 列表中的元素通过空字符串连接起来，形成一个字符串
    # 最后返回这个构建好的 SMILES 表示法字符串



def has_passed_through_filters(smiles, conf):#检查分子是否通过所有的fliter
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:  # default check
        return False#如果转换失败直接结束，转换成功了才会进行下一步
    checks = [f.check(mol, conf) for f in conf['filter_list']]#将分子对所有过滤器进行检查
    return all(checks)#检查所有都通过才会返回true，有一个不通过就会false


def neutralize_atoms(mol):
    #https://baoilleach.blogspot.com/2019/12/no-charge-simple-approach-to.html
    #https://www.rdkit.org/docs/Cookbook.html#neutralizing-molecules
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    return mol


def get_model_structure_info(model_json, logger):
    with open(model_json, 'r') as f:
        loaded_model_json = f.read()
    loaded_model = model_from_json(loaded_model_json)
    logger.info(f"Loaded model_json from {model_json}")
    input_shape = None
    vocab_size = None
    output_size = None
    for layer in loaded_model.get_config()['layers']:
        config = layer.get('config')
        if layer.get('class_name') == 'InputLayer':
            input_shape = config['batch_input_shape'][1]
        if layer.get('class_name') == 'Embedding':
            vocab_size = config['input_dim']
        if layer.get('class_name') == 'TimeDistributed':
            output_size = config['layer']['config']['units']
    if input_shape is None or vocab_size is None or output_size is None:
        logger.error('Confirm if the version of Tensorflow is 2.5. If so, please consult with ChemTSv2 developers on the GitHub repository. At that time, please attach the file specified as `model_json`')
        sys.exit()
            
    return input_shape, vocab_size, output_size

    
def loaded_model(model_weight, logger, conf):
    model = Sequential()
    model.add(Embedding(input_dim=conf['rnn_vocab_size'], output_dim=conf['rnn_vocab_size'],
                        mask_zero=False, batch_size=1))
    model.add(GRU(256, batch_input_shape=(1, None, conf['rnn_vocab_size']), activation='tanh',
                  return_sequences=True, stateful=True))
    model.add(GRU(256, activation='tanh', return_sequences=False, stateful=True))
    model.add(Dense(conf['rnn_output_size'], activation='softmax'))
    model.load_weights(model_weight)
    logger.info(f"Loaded model_weight from {model_weight}")

    return model


def evaluate_node(new_compound, generated_dict, reward_calculator, conf, logger, gids):
    node_index = []
    valid_compound = []
    generated_ids = []
    filter_check_list = []
    true_values = []

    valid_conf_list = []
    valid_mol_list = []
    valid_filter_check_value_list = []
    dup_compound_info = {}

    #check valid smiles
    for i in range(len(new_compound)):
        mol = Chem.MolFromSmiles(new_compound[i])
        if mol is None:
            continue#如果mol为空，说明无法从给定的smiles创建有效的分子对象，那么代码就会跳过当前的循环迭代，继续处理下一个化合物
        _mol = copy.deepcopy(mol)  #对mol创建一个拷贝，防止被原地修改# Chem.SanitizeMol() modifies `mol` in place
        
        if Chem.SanitizeMol(_mol, catchErrors=True).name != 'SANITIZE_NONE':#如果这个式子成立，说明分子在处理过程出现了问题，当前循环会被跳过
            continue

        #Neutralize
        if conf['neutralization']:
            if conf['neutralization_strategy'] == 'Uncharger':#如果条件成立，则代表选择了一种中性化策略叫uncharger
                un = rdMolStandardize.Uncharger()#处理分子化学中电荷的工具
                un.uncharge(mol)
            elif conf['neutralization_strategy'] == 'nocharge':#另一个中性化策略
                neutralize_atoms(mol)#中性化分子中的原子
            new_compound[i] = Chem.MolToSmiles(mol)#将经过中性化处理的分子转化为smiles

        if new_compound[i] in generated_dict:
            dup_compound_info[i] = { #储存化合物的信息
                'valid_compound': new_compound[i],
                'objective_values': generated_dict[new_compound[i]][0], 
                'generated_id': gids[i],
                'filter_check': generated_dict[new_compound[i]][1],
                'true_values': generated_dict[new_compound[i]][2]}
            continue

        if has_passed_through_filters(new_compound[i], conf):#如果分子通过了所有的fliter
            filter_check_value = 1
            filter_check_list.append(filter_check_value)
        else:
            if conf['include_filter_result_in_reward']:#是否把 分子是否通过fliter当作reward的评价标准
                filter_check_value = 0
                filter_check_list.append(filter_check_value)
            else:
                continue

        _conf = copy.deepcopy(conf)
        _conf['gid'] = gids[i]
        node_index.append(i)
        valid_compound.append(new_compound[i])#能走到这里的都是有效并且通过fliter的分子了
        generated_ids.append(gids[i])

        valid_conf_list.append(_conf)
        valid_mol_list.append(mol)
        valid_filter_check_value_list.append(filter_check_value)

    if len(valid_mol_list) == 0: #如果没有有效分子
        return [], [], [], [],[],[]
    
    #calculation rewards of valid molecules计算奖励值
    def _get_objective_values(mol, conf):
        return [f(mol) for f in reward_calculator.get_objective_functions(conf)]
    def _get_true_values(mol, conf):
        return [f(mol) for f in reward_calculator.get_true_objective_functions(conf)]

    if conf['leaf_parallel']:
        if conf['qsub_parallel']:
            if len(valid_mol_list) > 0:
                values_list = run_qsub_parallel(valid_mol_list, reward_calculator, valid_conf_list)#进行分子模拟并返回目标列表值
        else:
            # standard parallelization标准并行化
            values_list = joblib.Parallel(n_jobs=conf['leaf_parallel_num'])(
                joblib.delayed(_get_objective_values)(m, c) for m, c in zip(valid_mol_list, valid_conf_list))
    elif conf['batch_reward_calculation']:
        values_list = [f(valid_mol_list, valid_conf_list) for f in reward_calculator.get_batch_objective_functions()]
        values_list = np.array(values_list).T.tolist()
    else:
        values_list = [_get_objective_values(m, c) for m, c in zip(valid_mol_list, valid_conf_list)]
        true_values =  [_get_true_values(m, c) for m, c in zip(valid_mol_list, valid_conf_list)]
        #假设在这里得到一个val_score=[]里面是该分子对每个goal的得分
    #record values and other data
    for i in range(len(valid_mol_list)):
        values = values_list[i]
        score = true_values[i]
        filter_check_value = valid_filter_check_value_list[i]
        generated_dict[valid_compound[i]] = [values, filter_check_value,score]

    # add duplicate compounds' data if duplicated compounds are generated
    for k, v in sorted(dup_compound_info.items()):
        node_index.append(k)
        valid_compound.append(v['valid_compound'])
        generated_ids.append(v['generated_id'])
        values_list.append(v['objective_values'])
        filter_check_list.append(v['filter_check'])
        true_values.append(v['true_values'])

    logger.info(f"Valid SMILES ratio: {len(valid_compound)/len(new_compound)}")

    return node_index, values_list, valid_compound, generated_ids, filter_check_list, true_values
