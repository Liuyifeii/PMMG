import os
import sys
import time
import math
import numpy as np
import pandas as pd
import pickle
import random
from math import log, sqrt
from chemtsv2.utils import chem_kn_simulation, build_smiles_from_tokens,\
    evaluate_node, node_to_add, expanded_node, back_propagation

goals_num = 8
class State:
    def __init__(self, position=['&'], parent=None,n_obj=8):
        self.position = position
        self.visits = 0
        self.total_reward = np.zeros(n_obj)
        #self.total_reward = 0
        self.parent_node = parent
        self.child_nodes = []   
 
    def clone(self, include_visit=False, include_total_reward=False, include_parent_node=False, include_child_node=False):#设置里面的某些值是否被复制
        st = State()
        st.position = self.position[:]
        st.visits = self.visits if include_visit else 0
        st.total_reward = self.total_reward if include_total_reward else [0]*goals_num
        st.parent_node = self.parent_node if include_parent_node else None
        st.child_nodes = self.child_nodes if include_child_node else []
        self.score = []
        return st
    



    def add_position(self, m):
        self.position.append(m)


class Node:
    def __init__(self, policy_evaluator, position=None, state=None, conf=None, n_obj=8):
        self.position = position
        self.state = state
        #self.reward = 0
        #self.reward = np.zeros(n_obj)
        self.policy_evaluator = policy_evaluator
        self.conf = conf

    def select_node(self, logger):#选出得分最高的节点（从得分最高的节点中随机选择一个节点）#选择节点的算法这部分需要改，要结合pareto


        best_d = dict()
        for i in range(len(self.state.child_nodes)):
            n = self.state.child_nodes[i]
            print('n.state.total_reward is here')
            print(n.state.total_reward)
            print(n.state.visits)
            exploit = n.state.total_reward / n.state.visits
            explore = sqrt(2 * log(n.state.parent_node.state.visits) / n.state.visits)
            # explore = math.sqrt(2.0 * math.log(self.state.visits) + 0.5 * math.log(len(self.state.total_reward)) / float(n.state.visits))
            print(exploit)
            #print(explore)
            score = exploit + explore*0.2#可以调整系数
            best_d[n] = score
        first_run = True
        last_len = len(best_d)
        while len(best_d) != last_len or first_run:
            first_run = False
            last_len = len(best_d)
            keys = list(best_d.keys())
            keys_to_delete = []
            for i in range(len(keys)-1):
                node1 = keys[i]
                score1 = np.array(best_d[node1])
                for j in range(i, len(keys)):
                    node2 = keys[j]
                    score2 = np.array(best_d[node2])
                    if (score1 > score2).all():
                        keys_to_delete.append(node2)
                    elif (score1 < score2).all():
                        keys_to_delete.append(node1)
                    else:
                        pass
            for k in keys_to_delete:
                best_d.pop(k, None)
        best_children = list(best_d.keys())
        return random.choice(best_children)


    def add_node(self, m, state, policy_evaluator):#添加新的节点
        state.parent_node = self#将子节点变为父节点
        node = Node(policy_evaluator, position=m, state=state, conf=self.conf)
        self.state.child_nodes.append(node)

    def simulation(self):
        raise SystemExit("[ERROR] Do NOT use this method")

    def update(self, reward):
        self.state.visits += 1
        self.state.total_reward += np.array(reward,dtype='float64')#将reward的值累加到对象的total_reward变量中



class MCTS:
    def __init__(self, root_state, conf, tokens, model, reward_calculator, policy_evaluator, logger):
        self.start_time = time.time()
        self.rootnode = Node(policy_evaluator, state=root_state, conf=conf)
        self.conf = conf
        self.tokens = tokens
        self.model = model
        self.reward_calculator = reward_calculator
        self.policy_evaluator = policy_evaluator
        self.logger = logger

        self.valid_smiles_list = []
        self.depth_list = []
        self.objective_values_list = []
        self.true_values_list = []
        self.reward_values_list = []
        self.elapsed_time_list = []
        self.generated_dict = {}  # dictionary of generated compounds
        self.generated_id_list = []
        self.filter_check_list = []
        self.total_valid_num = 0
        self.tmp_smiles = []
        self.tmp_values = []
        
        if conf['batch_reward_calculation']:
            self.obj_column_names = [f.__name__ for f in self.reward_calculator.get_batch_objective_functions()]
        else:
            self.obj_column_names = [f.__name__ for f in self.reward_calculator.get_objective_functions(self.conf)]
        self.output_path = os.path.join(conf['output_dir'], f"result_C{conf['c_val']}.csv")
        if os.path.exists(self.output_path) and not conf['restart']:
            sys.exit(f"[ERROR] {self.output_path} already exists. Please specify a different file name.")

        self.gid = 0
        self.loop_counter_for_selection = 0
        self.loop_counter_for_expansion = 0
        self.expanded_before = {}

        if conf['threshold_type'] == "time":
            self.threshold = time.time() + 3600 * conf['hours']
        elif conf['threshold_type'] == "generation_num":
            self.threshold = conf['generation_num']
        else:
            sys.exit("[ERROR] Specify 'threshold_type': [time, generation_num]")

    def flush(self):


        df = pd.DataFrame({
            "generated_id": self.generated_id_list,#生成的ID
            "smiles": self.valid_smiles_list,#smiles字符串
            "reward": self.reward_values_list,#奖励值
            "score":self.true_values_list,
            "depth": self.depth_list,#深度
            "elapsed_time": self.elapsed_time_list,#经过的时间
            "is_through_filter": self.filter_check_list,#是否通过过滤器检查
        })
        a = [sublist1 + sublist2 for sublist1, sublist2 in zip(self.objective_values_list,self.true_values_list)]
        score_column_names = ['true_EGFR', 'true_ERBB2','true_Solubility', 'true_Permeability', 'true_Metabolic_stability', 'true_QED','true_Toxicity', 'true_SAScore']


        df_obj = pd.DataFrame(a, columns=self.obj_column_names + score_column_names)
        df = pd.concat([df, df_obj], axis=1)
        if os.path.exists(self.output_path):
            df.to_csv(self.output_path, mode='a', index=False, header=False)
        else:
            df.to_csv(self.output_path, mode='w', index=False)


        self.logger.info(f"save results at {self.output_path}")

        self.generated_id_list.clear()
        self.valid_smiles_list.clear()
        self.reward_values_list.clear()
        self.depth_list.clear()
        self.elapsed_time_list.clear()
        self.filter_check_list.clear()
        self.objective_values_list.clear()
        self.true_values_list.clear

    def dominate(self,v1,v2):
        dominate_score = 0
        if v1 in self.tmp_smiles and v2 in self.valid_smiles_list:
            index1 = self.tmp_smiles.index(v1)
            index2 = self.valid_smiles_list.index(v2)
            value1 = self.tmp_values[index1]
            value2 = self.objective_values_list[index2]
            #goals_num = goals_num
            for i in range (goals_num):
                if i < 5:
                    if value1[i] > value2[i]:#根据对应goal期望的数值大还是小，可以选择＞还是＜
                        dominate_score = dominate_score + 1
                elif i >= 5:
                    if value1[i] > value2[i]:
                        dominate_score = dominate_score + 1
            if dominate_score == goals_num:
                return True
            else:
                return False
        elif v1 in self.valid_smiles_list and v2 in self.tmp_smiles:
            index1 = self.valid_smiles_list.index(v1)
            index2 = self.tmp_smiles.index(v2)
            value1 = self.objective_values_list[index1]
            value2 = self.tmp_values[index2]
           # goals_num = goals_num
            for i in range (0,goals_num):
                if i < 5:
                    if value1[i] > value2[i]:#根据对应goal期望的数值大还是小，可以选择＞还是＜
                        dominate_score = dominate_score + 1
                elif i >= 5:
                    if value1[i] > value2[i]:
                        dominate_score = dominate_score + 1
            if dominate_score == goals_num:
                return True
            else:
                return False
    def remove_duplicates(self,lst):
        unique_elements = []
        for item in lst:
            if item not in unique_elements:
                unique_elements.append(item)
        lst.clear()
        lst.extend(unique_elements)

    def search(self):
        """initialization of search tree"""
        slection = 1
        expansion = 1
        back = 1
        ckpt_path = os.path.join(self.conf['output_dir'], self.conf['checkpoint_file'])
        if self.conf['restart'] and os.path.exists(ckpt_path):
            self.logger.info(f"Load the checkpoint file from {ckpt_path}")
            self.load_checkpoint()
        
        while (time.time() if self.conf['threshold_type']=="time" else self.total_valid_num) <= self.threshold:#小于设定的阈值的时候就进行以下步骤
            node = self.rootnode 
            state = node.state.clone() 

            """selection step"""
            slection = slection + 1
            print("'slection' 等于", slection)

            node_pool = []
            while node.state.child_nodes != []:
                node = node.select_node(self.logger)
                state.add_position(node.position)
            self.logger.info(f"state position: {state.position}")

            self.logger.debug(f"infinite loop counter (selection): {self.loop_counter_for_selection}")
            if node.position == '\n':
                back_propagation(node, reward=[0]*goals_num)
                self.loop_counter_for_selection += 1
                if self.loop_counter_for_selection > self.conf['infinite_loop_threshold_for_selection']:#如果循环次数超过的设定的无限循环次数，则代表发生了无限循环
                    self.flush()
                    sys.exit('[WARN] Infinite loop is detected in the selection step. Change hyperparameters or RNN model.')
                continue
            else:
                self.loop_counter_for_selection = 0

            

            """expansion step"""
            expansion = expansion + 1 
            print("'expansion' 等于", expansion)
            expanded = expanded_node(self.model, state.position, self.tokens, self.logger, threshold=self.conf['expansion_threshold'])
            self.logger.debug(f"infinite loop counter (expansion): {self.loop_counter_for_expansion}")
            if set(expanded) == self.expanded_before:
                self.loop_counter_for_expansion += 1
                if self.loop_counter_for_expansion > self.conf['infinite_loop_threshold_for_expansion']:
                    self.flush()
                    sys.exit('[WARN] Infinite loop is detected in the expansion step. Change hyperparameters or RNN model.')
            else:
                self.loop_counter_for_expansion = 0
            self.expanded_before = set(expanded)

            new_compound = []
            nodeadded = []
            for _ in range(self.conf['simulation_num']):
                nodeadded_tmp = node_to_add(expanded, self.tokens, self.logger)
                nodeadded.extend(nodeadded_tmp)#将nodeadded_tmp中的元素追加到nodeadded中
                for n in nodeadded_tmp:
                    position_tmp = state.position + [n]#将n加到列表末尾
                    all_posible = chem_kn_simulation(self.model, position_tmp, self.tokens, self.conf)

                    new_compound.append(build_smiles_from_tokens(all_posible, self.tokens))#把一个个字符连接成smiles字符串
            self.remove_duplicates(new_compound)#去除重复的分子
            _gids = list(range(self.gid, self.gid+len(new_compound)))
            self.gid += len(new_compound)
            self.logger.debug(f"nodeadded {nodeadded}")
            self.logger.info(f"new compound {new_compound}")
            self.logger.debug(f"generated_dict {self.generated_dict}") 
            if self.conf["debug"]:
                self.logger.debug('\n' + '\n'.join([f"lastcomp {comp[-1]} ... " + str(comp[-1] == '\n') for comp in new_compound]))

            print('new_compound is here')
            print(new_compound)            
            print(self.generated_dict)
            print(self.reward_calculator)
            print(self.conf)
            print(self.logger)            
            print(_gids)
            node_index, objective_values, valid_smiles, generated_id_list, filter_check_list ,true_values = evaluate_node(new_compound, self.generated_dict, self.reward_calculator, self.conf, self.logger, _gids)

            """pareto select"""
            if len(valid_smiles) > 1:#生成的分子先自我比对删除被支配分子
                tmp = valid_smiles
                tmp_values = objective_values
                tmp_back = valid_smiles
                tmp_values_back = objective_values
                tmp_true_values = []
                tmp_smiles_to_delete = []
                tmp_valid_smiles = []
                tmp_node_index = []
                tmp_objective_values = []
                tmp_generated_id_list = []
                tmp_filter_check_list = []
                for smiles3 in tmp:
                    for smiles4 in tmp_back:
                        index_tmp3 = tmp.index(smiles3)
                        index_tmp4 = tmp_back.index(smiles4)
                        values3 = tmp_values[index_tmp3]
                        values4 = tmp_values_back[index_tmp4]
                        score = 0
                        n = goals_num
                        for i in range(n):
                            if values3[i] > values4[i]:
                                score = score + 1 
                        if score == n:
                            tmp_smiles_to_delete.append(smiles4)
                for s in tmp:
                    if s not in tmp_smiles_to_delete:
                        index4 = tmp.index(s)
                        tmp_valid_smiles.append(s)
                        tmp_node_index.append(node_index[index4])
                        tmp_objective_values.append(objective_values[index4])
                        tmp_generated_id_list.append(generated_id_list[index4])
                        tmp_filter_check_list.append(filter_check_list[index4])
                        tmp_true_values.append(true_values[index4])
                        
                node_index = tmp_node_index
                objective_values = tmp_objective_values
                valid_smiles = tmp_valid_smiles
                generated_id_list = tmp_generated_id_list
                filter_check_list = tmp_filter_check_list
                true_values = tmp_true_values

                
            
            self.tmp_smiles = valid_smiles
            self.tmp_values = objective_values
            a = valid_smiles
            valid_smiles_list_to_delete = [] 
            valid_smiles_to_delete = []
            if self.valid_smiles_list != [] and valid_smiles != []:
                for smiles1 in valid_smiles:
                    for smiles2 in self.valid_smiles_list:
                        
                        if self.dominate(smiles1,smiles2):
                            valid_smiles_list_to_delete.append(smiles2)
                    

                        elif self.dominate(smiles2,smiles1):
                            valid_smiles_to_delete.append(smiles1)
                for smiles2 in valid_smiles_list_to_delete:
                    if smiles2 in self.valid_smiles_list:
                        index2 = self.valid_smiles_list.index(smiles2)
                        self.valid_smiles_list.pop(index2)
                        self.objective_values_list.pop(index2)
                        self.generated_id_list.pop(index2)
                        self.filter_check_list.pop(index2)
                        self.depth_list.pop(index2)
                        self.elapsed_time_list.pop(index2)
                        self.reward_values_list.pop(index2)
                        self.true_values_list.pop(index2)
                for smiles1 in valid_smiles_to_delete:
                    if smiles1 in valid_smiles:
                        index1 = valid_smiles.index(smiles1)
                        valid_smiles.pop(index1)
                        node_index.pop(index1)
                        objective_values.pop(index1)
                        generated_id_list.pop(index1)
                        filter_check_list.pop(index1)
                        true_values.pop(index1)

            #删除分子以后还要删除其他列表里对应的被删除分子的数值

            if len(valid_smiles) == 0:
                back_propagation(node, reward=[0]*goals_num)
                continue
            v_to_add = []
            v_to_delete = []
            #self.total_valid_num += valid_num#总有效分子数
            for s in valid_smiles:#去除重复分子
                if s not in self.valid_smiles_list:
                    v_to_add.append(s)
                    index_s = valid_smiles.index(s)
                    self.valid_smiles_list.append(s)
                    self.objective_values_list.append(objective_values[index_s])
                    self.generated_id_list.append(generated_id_list[index_s])
                    self.filter_check_list.append(filter_check_list[index_s])
                    self.true_values_list.append(true_values[index_s])
                else:
                    v_to_delete.append(s)
            for v in v_to_delete:
                index_x = valid_smiles.index(v)
                valid_smiles.pop(index_x)
                node_index.pop(index_x)
                objective_values.pop(index_x)
                true_values.pop(index_x)
            valid_num = len(v_to_add)#有小分子数
            #self.valid_smiles_list.extend(valid_smiles)
            self.total_valid_num = len(self.valid_smiles_list)#总有效分子数
            depth = len(state.position)
            self.depth_list.extend([depth for _ in range(valid_num)])
            elapsed_time = round(time.time()-self.start_time, 1)
            self.elapsed_time_list.extend([elapsed_time for _ in range(valid_num)])
            #self.objective_values_list.extend(objective_values)
            #self.generated_id_list.extend(generated_id_list)
            #self.filter_check_list.extend(filter_check_list)

            self.logger.info(f"Number of valid SMILES: {self.total_valid_num}")
            self.logger.debug(f"node {node_index} objective_values {objective_values} valid smiles {valid_smiles} time {elapsed_time}")
            
            def dominate_score(v1, v2):
     
                reward = [1.0]*goals_num
                for i in range(goals_num):
                    if i == 0 :
                        if v1[i] < v2[i]:
                            reward[i] = 0.0
                    if i == 1:
                        if v1[i] < v2[i]:
                            reward[i] = 0.0
                    if i == 2:
                        if v1[i] < v2[i]:
                            reward[i] = 0.0
                    if i == 3:
                        if v1[i] < v2[i]:
                            reward[i] = 0.0
                    if i == 4:
                        if v1[i] < v2[i]:
                            reward[i] = 0.0
                    if i == 5:
                        if v1[i] < v2[i]:
                            reward[i] = 0.0
                    if i == 6:
                        if v1[i] < v2[i]:
                            reward[i] = 0.0
                    if i == 7:
                        if v1[i] < v2[i]:
                            reward[i] = 0.0   
                    if i == 8:
                        if v1[i] < v2[i]:
                            reward[i] = 0.0                    
                return np.array(reward)


            re_list = []
            atom_checked = []
            for i in range(len(node_index)):
                m = node_index[i]
                atom = nodeadded[m]
                state_clone = state.clone(include_visit=True, include_total_reward=True)

                if atom not in atom_checked: 
                    node.add_node(atom, state_clone, self.policy_evaluator)
                    node_pool.append(node.state.child_nodes[len(atom_checked)])
                    atom_checked.append(atom)
                else:
                    node_pool.append(node.state.child_nodes[atom_checked.index(atom)])

                if self.conf["debug"]:
                    self.logger.debug('\n' + '\n'.join([f"Child node position ... {c.position}" for c in node.state.child_nodes]))

                re = [0]*goals_num if atom == '\n' else self.reward_calculator.calc_reward_from_objective_values(values=objective_values[i], conf=self.conf)
                total_reward = np.zeros(goals_num)
                if atom == '\n':
                    dominate_re = re
                else:
                    for i in range(len(self.valid_smiles_list)):
                        re1 = self.objective_values_list[i]
                        win = dominate_score(re,re1)
                        total_reward += win
                    n = len(self.valid_smiles_list)
                    print('n and total_reward is here')
                    print(n)
                    print(total_reward)
                    dominate_re = total_reward/n




                if self.conf['include_filter_result_in_reward']:
                    dominate_re *= filter_check_list[i]#re乘以filter_check_list[i]的值
                    
                re_list.append(dominate_re)
                self.logger.debug(f"atom: {atom} re_list: {re_list}")
            self.reward_values_list.extend(re_list)

            """backpropation step"""
            back = back +1
            print("'back' 等于", back)
            for i in range(len(node_pool)):
                node = node_pool[i]
                back_propagation(node, reward=re_list[i])

            if self.conf['debug']:
                self.logger.debug('\n' + '\n'.join([f"child position: {c.position}, total_reward: {c.state.total_reward}, visits: {c.state.visits}" for c in node_pool]))

            if len(self.valid_smiles_list) > self.conf['flush_threshold'] and self.conf['flush_threshold'] != -1:
                self.flush()
            
            """save checkpoint file"""
            if self.conf['save_checkpoint']:
                self.save_checkpoint()


        if len(self.valid_smiles_list) > 0:
            self.flush()
            
    def load_checkpoint(self):#加载ckpt文件
        ckpt_path = os.path.join(self.conf['output_dir'], self.conf['checkpoint_file'])
        with open(ckpt_path, mode='rb') as f:
            cp_obj = pickle.load(f)
        self.gid = cp_obj['gid']
        self.loop_counter_for_selection = cp_obj['loop_counter_for_selection']
        self.loop_counter_for_expansion = cp_obj['loop_counter_for_expansion']
        self.expanded_before = cp_obj['expanded_before']        
        self.start_time = cp_obj['start_time']
        self.rootnode = cp_obj['rootnode']
        self.conf = cp_obj['conf']
        self.generated_dict = cp_obj['generated_dict']
        self.total_valid_num = cp_obj['total_valid_num']


    def save_checkpoint(self):
        ckpt_fname = self.conf['checkpoint_file']
        ckpt_path = os.path.join(self.conf['output_dir'], ckpt_fname)
        stem, ext = ckpt_fname.rsplit('.', 1)
        # To keep the three most recent checkpoint files.
        ckpt1_path = os.path.join(self.conf['output_dir'], f'{stem}2.{ext}')
        ckpt2_path = os.path.join(self.conf['output_dir'], f'{stem}3.{ext}')
        if os.path.exists(ckpt1_path):
            os.rename(ckpt1_path, ckpt2_path)
        if os.path.exists(ckpt_path):
            os.rename(ckpt_path, ckpt1_path)

        cp_obj = {
            'gid': self.gid,
            'loop_counter_for_selection': self.loop_counter_for_selection,
            'loop_counter_for_expansion': self.loop_counter_for_expansion,
            'expanded_before': self.expanded_before,
            'start_time': self.start_time, 
            'conf': self.conf, 
            'rootnode': self.rootnode,
            'generated_dict': self.generated_dict,
            'total_valid_num': self.total_valid_num,
        }
        
        with open(ckpt_path, mode='wb') as f:
            pickle.dump(cp_obj, f)
        self.flush()
            