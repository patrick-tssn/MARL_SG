import numpy as np
import torch
from du_rl.vrd_utils import calc_cum_discounted_reward_credit, calc_reinforce_loss, rouge_n
import copy

# parallel
def RR_p(x):
    return RR(*x)

def RR(vrd_agent, text_agent, rl_optimizer, tokenizer, vrd_baseline, text_baseline, \
     vid_lst, subid_lst, vrd_start_lsts, text_start_lsts, ref_dct, ref_lsts, video_loader, text_loader, Knowledge_graph, \
        vrd_id2rel, text_id2rel, vrd_id2ent, text_id2ent, vrd_option, text_option, args, rr_itr):


    # RR
    if rr_itr != 0:
        vrd_all_rr_loss = []
        vrd_all_rr_logits = []
        vrd_rr_reward = []
        vrd_all_rr_traj = []

        text_all_rr_loss = []
        text_all_rr_logits = []
        text_rr_reward = []
        text_all_rr_traj = []

        ref_str_rr_lst = []
        ref_trip_rr_lst = []

        for i in range(len(vid_lst)):
            vid = vid_lst[i]
            subid = subid_lst[i]
            # ref = ref_lsts[i] # string 
            ori_bsz = min(len(vrd_start_lsts[i][:args.top_n]), len(text_start_lsts[i][:args.top_n]))
            bsz = ori_bsz * vrd_option.train_rollouts
            ref_str_rr_lst.append(ref_dct[subid]) # string 

            # vrd_agent init
            # 1. init Graph
            vrd_trip_data = video_loader.get_graph_data(vid)
            vrd_train_graph = Knowledge_graph(vrd_option, video_loader, vrd_trip_data)
            vrd_agent.set_graph(vrd_train_graph)
            # 2. init start entities
            vrd_start_ent_str = [x[0] for x in vrd_start_lsts[i][:ori_bsz]] # [ent1, ent2, ...]
            vrd_start_rel_str = [x[1] for x in vrd_start_lsts[i][:ori_bsz]] # [rel1, rel2, ...]
            vrd_start_entity_id = [video_loader.entity2num[x] for x in vrd_start_ent_str]
            vrd_start_relation_id = [video_loader.relation2num[x] for x in vrd_start_rel_str]
            vrd_start_entity_id = np.repeat(vrd_start_entity_id, vrd_option.train_rollouts)
            vrd_start_relation_id = np.repeat(vrd_start_relation_id, vrd_option.train_rollouts)
            vrd_start_entity = torch.tensor(vrd_start_entity_id).to(args.device)
            vrd_relation = torch.tensor(vrd_start_relation_id).to(args.device)
            vrd_prev_relation = vrd_agent.get_dummy_start_relation(bsz).to(args.device)
            vrd_current_entity = vrd_start_entity
            vrd_traj = []
            for x in vrd_start_ent_str:
                for j in range(vrd_option.train_rollouts):
                    vrd_traj.append([x])
            vrd_state = torch.zeros(1, 2, bsz, vrd_agent.m * vrd_option.state_embed_size).to(args.device)
            vrd_log_probs = np.zeros((bsz, )) * 1.0

            # text_agent init
            # 1. init graph
            text_trip_data = text_loader.get_graph_data(subid)
            text_train_graph = Knowledge_graph(text_option, text_loader, text_trip_data)
            text_agent.set_graph(text_train_graph)
            # 2. init start entities
            text_start_ent_str = [x[0] for x in text_start_lsts[i][:ori_bsz]] # [ent1, ent2, ...]
            text_start_rel_str = [x[1] for x in text_start_lsts[i][:ori_bsz]] # [rel1, rel2, ...]
            text_start_entity_id = [text_loader.entity2num[x] for x in text_start_ent_str]
            text_start_relation_id = [text_loader.relation2num[x] for x in text_start_rel_str]
            text_start_entity_id = np.repeat(text_start_entity_id, text_option.train_rollouts)
            text_start_relation_id = np.repeat(text_start_relation_id, text_option.train_rollouts)
            text_start_entity = torch.tensor(text_start_entity_id).to(args.device)
            text_relation = torch.tensor(text_start_relation_id).to(args.device)
            text_prev_relation = text_agent.get_dummy_start_relation(bsz).to(args.device)
            text_current_entity = text_start_entity
            text_traj = []
            for x in text_start_ent_str:
                for j in range(vrd_option.train_rollouts):
                    text_traj.append([x])
            text_state = torch.zeros(1, 2, bsz, text_agent.m * text_option.state_embed_size).to(args.device)
            text_log_probs = np.zeros((bsz, )) * 1.0
            
            range_arr = torch.arange(bsz).to(args.device)
            
            
        
            # start grounding on graph 
            for step in range(vrd_option.max_step_length):
                vrd_loss, vrd_state, vrd_logits, vrd_action_idx, vrd_next_entity, vrd_chosen_relation= \
                    vrd_agent.step(vrd_state, vrd_prev_relation, vrd_current_entity, vrd_relation, range_arr, text_state)
                # loss [B] logits [B, max_out] action_id [B]
                text_loss, text_state, text_logits, text_action_idx, text_next_entity, text_chosen_relation= \
                    text_agent.step(text_state, text_prev_relation, text_current_entity, text_relation, range_arr, vrd_state)
                # loss [B] logits [B, max_out] action_id [B]

                # vrd_loss = torch.mean(vrd_loss, dim=0).unsqueeze(0)
                # vrd_logits = torch.mean(vrd_logits, dim=0).unsqueeze(0)
                # text_loss = torch.mean(text_loss, dim=0).unsqueeze(0)
                # text_logits = torch.mean(text_logits, dim=0).unsqueeze(0)

                vrd_log_probs += vrd_logits.clone().detach().cpu().numpy()[np.arange(vrd_log_probs.shape[0]), vrd_action_idx.cpu().numpy()]
                text_log_probs += text_logits.clone().detach().cpu().numpy()[np.arange(text_log_probs.shape[0]), text_action_idx.cpu().numpy()] # [B*num_rollouts]

                if i == 0:
                    vrd_all_rr_loss.append(vrd_loss) # [B, 1]
                    vrd_all_rr_logits.append(vrd_logits) # [B, max_out]
                    text_all_rr_loss.append(text_loss) # [B, 1]
                    text_all_rr_logits.append(text_logits) # [B, max_out]
                else:
                    vrd_all_rr_loss[step] = torch.cat([vrd_all_rr_loss[step], vrd_loss], dim=0)
                    vrd_all_rr_logits[step] = torch.cat([vrd_all_rr_logits[step], vrd_logits], dim=0)
                    text_all_rr_loss[step] = torch.cat([text_all_rr_loss[step], text_loss], dim=0)
                    text_all_rr_logits[step] = torch.cat([text_all_rr_logits[step], text_logits], dim=0)
                

                # all_action_id.append(action_id)
                vrd_prev_relation = vrd_chosen_relation
                vrd_current_entity = vrd_next_entity
                for j in range(bsz):
                    vrd_traj[j].append(vrd_id2rel[(int)(vrd_chosen_relation[j])])
                    vrd_traj[j].append(vrd_id2ent[(int)(vrd_next_entity[j])])    

                # all_action_id.append(action_id)
                text_prev_relation = text_chosen_relation
                text_current_entity = text_next_entity
                for j in range(bsz):
                    text_traj[j].append(text_id2rel[(int)(text_chosen_relation[j])])
                    text_traj[j].append(text_id2ent[(int)(text_next_entity[j])])   

            # save all traj
            text_all_rr_traj += text_traj
            vrd_all_rr_traj += vrd_traj
            
            # select traj by log prob.
            # # merge batch
            # sorted_vrd_indx = np.argsort(-vrd_log_probs)
            # vrd_traj_lst = vrd_traj[sorted_vrd_indx[0]]
            # if vrd_traj_lst == []:
            #     vrd_traj_lst.append('<pad>')
            # split batch
            vrd_log_probs = np.reshape(vrd_log_probs, (ori_bsz, vrd_option.train_rollouts))
            sorted_vrd_indx = np.argsort(-vrd_log_probs) # [ori_B, rollouts]
            vrd_traj_lst = []
            for j in range(ori_bsz):
                vrd_traj_lst += vrd_traj[j*vrd_option.train_rollouts + sorted_vrd_indx[j][0]]
            if vrd_traj_lst == []:
                vrd_traj_lst.append('<pad>')
            
            # # merge batch
            # sorted_text_indx = np.argsort(-text_log_probs)
            # text_traj_lst = text_traj[sorted_text_indx[0]]
            # if text_traj_lst == []:
            #     text_traj_lst.append('<pad>')
            # split batch
            text_log_probs = np.reshape(text_log_probs, (ori_bsz, text_option.train_rollouts))
            sorted_text_indx = np.argsort(-text_log_probs) # [ori_B, rollouts]
            text_traj_lst = []
            for j in range(ori_bsz):
                text_traj_lst += text_traj[j*text_option.train_rollouts + sorted_text_indx[j][0]]
            if text_traj_lst == []:
                text_traj_lst.append('<pad>')
            

            # cosine anneal
            if ref_lsts[i] == []:
                ref = ['<pad>', '<pad>', '<pad>']
            else:
                ref = ref_lsts[i][np.random.choice(len(ref_lsts[i]), 1)[0]]
            # ref_lst.append(ref)
            ref_trip_rr_lst.append(ref)

        ft_index = np.random.choice(range(len(ref_trip_rr_lst)), 1, replace=False)[0]

        # optimize rl agent
        if args.reward == 'rouge':
            for j, traj in enumerate(vrd_all_rr_traj):
                
                # r1: 0 
                if rr_itr == 0:
                    vrd_rr_reward.append(0)

                # r2: 1 
                elif rr_itr == 1:
                    if ref_trip_rr_lst[j//bsz] == ['<pad>'] * len(ref_trip_rr_lst[j//bsz]) or traj == ['<pad>'] * len(traj):
                    # if traj == ['<pad>'] * len(traj):
                        vrd_rr_reward.append(0)
                    else:
                        # vrd_rr_reward.append(bleu_corpus(' '.join(traj), ref))
                        # gold trip
                        # vrd_rr_reward.append(rouge_n(tokenizer.tokenize(' '.join(ref_trip_rr_lst[j//bsz]).replace('<pad>', '')), tokenizer.tokenize(' '.join(traj).replace('<pad>', ''))))
                        vrd_rr_reward.append(rouge_n(' '.join(ref_trip_rr_lst[j//bsz]).replace('<pad>', '').split(), ' '.join(traj).replace('<pad>', '').split()))
                        # # gold str
                        # vrd_rr_reward.append(rouge_n(tokenizer.tokenize(ref_str_rr_lst[j//bsz]), tokenizer.tokenize(' '.join(traj).replace('<pad>', ''))))

                # r3: random weight
                elif rr_itr == 2:
                    # trip gold
                    if ref_trip_rr_lst[j//bsz] == ['<pad>'] * len(ref_trip_rr_lst[j//bsz]) or traj == ['<pad>'] * len(traj):
                        vrd_rr_reward.append(0)
                    else:
                        # vrd_rr_reward.append(bleu_corpus(' '.join(traj), ref))
                        # gold trip
                        vrd_rr_reward.append(rouge_n(tokenizer.tokenize(' '.join(ref_trip_rr_lst[j//bsz]).replace('<pad>', '')), tokenizer.tokenize(' '.join(traj).replace('<pad>', '')))*np.random.random())
                    # # str gold
                    # if ref_str_rr_lst[j//bsz] == '' or traj == ['<pad>'] * len(traj):
                    #     vrd_rr_reward.append(0)
                    # else:
                    #     vrd_rr_reward.append(rouge_n(tokenizer.tokenize(ref_str_rr_lst[j//bsz]), tokenizer.tokenize(' '.join(traj).replace('<pad>', ''))))

                # r4: random reward 4
                elif rr_itr == 3:
                    if ref_trip_rr_lst[-1] == ['<pad>'] * len(ref_trip_rr_lst[-1]) or traj == ['<pad>'] * len(traj):
                        vrd_rr_reward.append(0)
                    else:
                        # vrd_rr_reward.append(bleu_corpus(' '.join(traj), ref))
                        # gold trip
                        # vrd_rr_reward.append(rouge_n(tokenizer.tokenize(' '.join(ref_trip_rr_lst[-1]).replace('<pad>', '')), tokenizer.tokenize(' '.join(traj).replace('<pad>', ''))))
                        vrd_rr_reward.append(rouge_n(' '.join(ref_trip_rr_lst[-1]).replace('<pad>', '').split(), ' '.join(traj).replace('<pad>', '').split()))
                        # # gold str
                        # vrd_rr_reward.append(rouge_n(tokenizer.tokenize(ref_str_rr_lst[-1]), tokenizer.tokenize(' '.join(traj).replace('<pad>', ''))))
                    # # str gold
                    # if ref_str_rr_lst[-1] == '' or traj == ['<pad>'] * len(traj):
                    #     vrd_rr_reward.append(0)
                    # else:
                    #     vrd_rr_reward.append(rouge_n(tokenizer.tokenize(ref_str_rr_lst[-1]), tokenizer.tokenize(' '.join(traj).replace('<pad>', ''))))
                
                # r5: random weight
                elif rr_itr == 4:
                    if ref_trip_rr_lst[j//bsz] == ['<pad>'] * len(ref_trip_rr_lst[j//bsz]) or traj == ['<pad>'] * len(traj):
                        vrd_rr_reward.append(0)
                    else:
                        # vrd_rr_reward.append(bleu_corpus(' '.join(traj), ref))
                        # gold trip
                        vrd_rr_reward.append(rouge_n(tokenizer.tokenize(' '.join(ref_trip_rr_lst[j//bsz]).replace('<pad>', '')), tokenizer.tokenize(' '.join(traj).replace('<pad>', '')))*3)
                
                # r6: random index
                elif rr_itr == 5:
                    if ref_trip_rr_lst[ft_index] == ['<pad>'] * len(ref_trip_rr_lst[ft_index]) or traj == ['<pad>'] * len(traj):
                        vrd_rr_reward.append(0)
                    else:
                        # vrd_rr_reward.append(bleu_corpus(' '.join(traj), ref))
                        # gold trip
                        vrd_rr_reward.append(rouge_n(tokenizer.tokenize(' '.join(ref_trip_rr_lst[ft_index]).replace('<pad>', '')), tokenizer.tokenize(' '.join(traj).replace('<pad>', ''))))
                    # if ref_trip_rr_lst[j//bsz] == ['<pad>'] * len(ref_trip_rr_lst[j//bsz]) or traj == ['<pad>'] * len(traj):
                    #     vrd_rr_reward.append(0)
                    # else:
                    #     # vrd_rr_reward.append(bleu_corpus(' '.join(traj), ref))
                    #     # gold trip
                    #     vrd_rr_reward.append(rouge_n(tokenizer.tokenize(' '.join(ref_trip_rr_lst[j//bsz]).replace('<pad>', '')), tokenizer.tokenize(' '.join(traj).replace('<pad>', ''))))
                
                # r7: random reward 2
                elif rr_itr == 6:
                    if j % 4 != 0:
                        if ref_trip_rr_lst[j//bsz] == ['<pad>'] * len(ref_trip_rr_lst[j//bsz]) or traj == ['<pad>'] * len(traj):
                            vrd_rr_reward.append(0)
                        else:
                            # vrd_rr_reward.append(bleu_corpus(' '.join(traj), ref))
                            # gold trip
                            vrd_rr_reward.append(rouge_n(tokenizer.tokenize(' '.join(ref_trip_rr_lst[j//bsz]).replace('<pad>', '')), tokenizer.tokenize(' '.join(traj).replace('<pad>', ''))))
                    else:
                        if ref_trip_rr_lst[ft_index] == ['<pad>'] * len(ref_trip_rr_lst[ft_index]) or traj == ['<pad>'] * len(traj):
                            vrd_rr_reward.append(0)
                        else:
                            # vrd_rr_reward.append(bleu_corpus(' '.join(traj), ref))
                            # gold trip
                            vrd_rr_reward.append(rouge_n(tokenizer.tokenize(' '.join(ref_trip_rr_lst[ft_index]).replace('<pad>', '')), tokenizer.tokenize(' '.join(traj).replace('<pad>', ''))))
                    # # str gold
                    # if ref_str_rr_lst[-1] == '' or traj == ['<pad>'] * len(traj):
                    #     vrd_rr_reward.append(0)
                    # else:
                    #     vrd_rr_reward.append(rouge_n(tokenizer.tokenize(ref_str_rr_lst[-1]), tokenizer.tokenize(' '.join(traj).replace('<pad>', ''))))
                elif rr_itr == 7:
                    if j // bsz == len(ref_trip_rr_lst)-1: # trian_batch_size is 4
                        if ref_trip_rr_lst[j//bsz] == ['<pad>'] * len(ref_trip_rr_lst[j//bsz]) or traj == ['<pad>'] * len(traj):
                            vrd_rr_reward.append(0)
                        else:
                            # text_rr_reward.append(bleu_corpus(' '.join(traj), ref))
                            # gold trip
                            vrd_rr_reward.append(rouge_n(tokenizer.tokenize(' '.join(ref_trip_rr_lst[j//bsz]).replace('<pad>', '')), tokenizer.tokenize(' '.join(traj).replace('<pad>', ''))))
                    else:
                        vrd_rr_reward.append(0) 
                else:
                    raise ValueError('INVALID RR ITR!')
                
            for j, traj in enumerate(text_all_rr_traj): # bsz * num_rollouts
                # r1: 0 
                if rr_itr == 0:
                    text_rr_reward.append(0)

                # r2: standard reward
                elif rr_itr == 1:
                    if ref_trip_rr_lst[j//bsz] == ['<pad>'] * len(ref_trip_rr_lst[j//bsz]) or traj == ['<pad>'] * len(traj):
                    # if traj == ['<pad>'] * len(traj):
                        text_rr_reward.append(0)
                    else:
                        # text_rr_reward.append(bleu_corpus(' '.join(traj), ref))
                        # gold trip
                        # text_rr_reward.append(rouge_n(tokenizer.tokenize(' '.join(ref_trip_rr_lst[j//bsz]).replace('<pad>', '')), tokenizer.tokenize(' '.join(traj).replace('<pad>', ''))))
                        text_rr_reward.append(rouge_n(' '.join(ref_trip_rr_lst[j//bsz]).replace('<pad>', '').split(), ' '.join(traj).replace('<pad>', '').split()))
                        # # gold str
                        # text_rr_reward.append(rouge_n(tokenizer.tokenize(ref_str_rr_lst[j//bsz]), tokenizer.tokenize(' '.join(traj).replace('<pad>', ''))))

                # r3: random weight
                elif rr_itr == 2:
                    if ref_trip_rr_lst[j//bsz] == ['<pad>'] * len(ref_trip_rr_lst[j//bsz]) or traj == ['<pad>'] * len(traj):
                        text_rr_reward.append(0)
                    else:
                        # text_rr_reward.append(bleu_corpus(' '.join(traj), ref))
                        # gold trip
                        text_rr_reward.append(rouge_n(tokenizer.tokenize(' '.join(ref_trip_rr_lst[j//bsz]).replace('<pad>', '')), tokenizer.tokenize(' '.join(traj).replace('<pad>', '')))*np.random.random())
                    # # str gold
                    # if ref_str_rr_lst[j//bsz] == '' or traj == ['<pad>'] * len(traj):
                    #     text_rr_reward.append(0)
                    # else:
                    #     text_rr_reward.append(rouge_n(tokenizer.tokenize(ref_str_rr_lst[j//bsz]), tokenizer.tokenize(' '.join(traj).replace('<pad>', ''))))

                # r3: last index
                elif rr_itr == 3:
                    if ref_trip_rr_lst[-1] == ['<pad>'] * len(ref_trip_rr_lst[-1]) or traj == ['<pad>'] * len(traj):
                        text_rr_reward.append(0)
                    else:
                        # text_rr_reward.append(bleu_corpus(' '.join(traj), ref))
                        # gold trip
                        # text_rr_reward.append(rouge_n(tokenizer.tokenize(' '.join(ref_trip_rr_lst[-1]).replace('<pad>', '')), tokenizer.tokenize(' '.join(traj).replace('<pad>', ''))))
                        text_rr_reward.append(rouge_n(' '.join(ref_trip_rr_lst[-1]).replace('<pad>', ''), ' '.join(traj).replace('<pad>', '')))
                        # # gold str
                        # text_rr_reward.append(rouge_n(tokenizer.tokenize(ref_str_rr_lst[-1]), tokenizer.tokenize(' '.join(traj).replace('<pad>', ''))))
                    # # str gold
                    # if ref_str_rr_lst[-1] == '' or traj == ['<pad>'] * len(traj):
                    #     text_rr_reward.append(0)
                    # else:
                    #     text_rr_reward.append(rouge_n(tokenizer.tokenize(ref_str_rr_lst[-1]), tokenizer.tokenize(' '.join(traj).replace('<pad>', ''))))

                # r4: large standard reward
                elif rr_itr == 4: 
                    if ref_trip_rr_lst[j//bsz] == ['<pad>'] * len(ref_trip_rr_lst[j//bsz]) or traj == ['<pad>'] * len(traj):
                        text_rr_reward.append(0)
                    else:
                        # text_rr_reward.append(bleu_corpus(' '.join(traj), ref))
                        # gold trip
                        text_rr_reward.append(rouge_n(tokenizer.tokenize(' '.join(ref_trip_rr_lst[j//bsz]).replace('<pad>', '')), tokenizer.tokenize(' '.join(traj).replace('<pad>', '')))*3)

                # r5: random index
                elif rr_itr == 5:
                    if ref_trip_rr_lst[ft_index] == ['<pad>'] * len(ref_trip_rr_lst[ft_index]) or traj == ['<pad>'] * len(traj):
                        text_rr_reward.append(0)
                    else:
                        # text_rr_reward.append(bleu_corpus(' '.join(traj), ref))
                        # gold trip
                        text_rr_reward.append(rouge_n(tokenizer.tokenize(' '.join(ref_trip_rr_lst[ft_index]).replace('<pad>', '')), tokenizer.tokenize(' '.join(traj).replace('<pad>', ''))))
                
                # r4: random reward 2
                elif rr_itr == 6:
                    if j % 4 != 0: # trian_batch_size is 4
                        if ref_trip_rr_lst[j//bsz] == ['<pad>'] * len(ref_trip_rr_lst[j//bsz]) or traj == ['<pad>'] * len(traj):
                            text_rr_reward.append(0)
                        else:
                            # text_rr_reward.append(bleu_corpus(' '.join(traj), ref))
                            # gold trip
                            text_rr_reward.append(rouge_n(tokenizer.tokenize(' '.join(ref_trip_rr_lst[j//bsz]).replace('<pad>', '')), tokenizer.tokenize(' '.join(traj).replace('<pad>', ''))))
                    else:
                        if ref_trip_rr_lst[ft_index] == ['<pad>'] * len(ref_trip_rr_lst[ft_index]) or traj == ['<pad>'] * len(traj):
                            text_rr_reward.append(0)
                        else:
                            # text_rr_reward.append(bleu_corpus(' '.join(traj), ref))
                            # gold trip
                            text_rr_reward.append(rouge_n(tokenizer.tokenize(' '.join(ref_trip_rr_lst[ft_index]).replace('<pad>', '')), tokenizer.tokenize(' '.join(traj).replace('<pad>', ''))))
                    # # str gold
                    # if ref_str_rr_lst[-1] == '' or traj == ['<pad>'] * len(traj):
                    #     text_rr_reward.append(0)
                    # else:
                    #     text_rr_reward.append(rouge_n(tokenizer.tokenize(ref_str_rr_lst[-1]), tokenizer.tokenize(' '.join(traj).replace('<pad>', ''))))
                elif rr_itr == 7:
                    if j // bsz == len(ref_trip_rr_lst)-1: # trian_batch_size is 4
                        if ref_trip_rr_lst[j//bsz] == ['<pad>'] * len(ref_trip_rr_lst[j//bsz]) or traj == ['<pad>'] * len(traj):
                            text_rr_reward.append(0)
                        else:
                            # text_rr_reward.append(bleu_corpus(' '.join(traj), ref))
                            # gold trip
                            text_rr_reward.append(rouge_n(tokenizer.tokenize(' '.join(ref_trip_rr_lst[j//bsz]).replace('<pad>', '')), tokenizer.tokenize(' '.join(traj).replace('<pad>', ''))))
                    else:
                        text_rr_reward.append(0)
                        

                else:
                    raise ValueError('INVALID RR ITR!')

        else:
            raise ValueError('INVALID REWARD TYPE!')

        # select best traj for constructing input of generator

        vrd_rr_reward = torch.tensor(vrd_rr_reward).to(args.device)
        text_rr_reward = torch.tensor(text_rr_reward).to(args.device)
        vrd_cum_discounted_reward = calc_cum_discounted_reward_credit(vrd_option, vrd_rr_reward, text_rr_reward*0)
        vrd_reinforce_loss = calc_reinforce_loss(vrd_baseline, vrd_all_rr_loss, vrd_all_rr_logits, vrd_cum_discounted_reward, vrd_option.beta)
        # vrd_baseline.update(torch.mean(vrd_cum_discounted_reward)) # baseline 使用线性插值
        text_cum_discounted_reward = calc_cum_discounted_reward_credit(text_option, text_rr_reward, vrd_rr_reward*0)
        text_reinforce_loss = calc_reinforce_loss(text_baseline, text_all_rr_loss, text_all_rr_logits, text_cum_discounted_reward, text_option.beta)
        # text_baseline.update(torch.mean(text_cum_discounted_reward))

        rl_optimizer.zero_grad()
        reinforce_loss = vrd_reinforce_loss + text_reinforce_loss
        reinforce_loss.backward()
        torch.nn.utils.clip_grad_norm_(vrd_agent.parameters(), max_norm=vrd_option.grad_clip_norm, norm_type=2)
        torch.nn.utils.clip_grad_norm_(text_agent.parameters(), max_norm=text_option.grad_clip_norm, norm_type=2)
        rl_optimizer.step()

    # evaluation
    vrd_eval_logit_lst = []
    text_eval_logit_lst = []
    vrd_eval_traj_lst = []
    text_eval_traj_lst = []
    vrd_eval_reward = []
    text_eval_reward = []
    ref_trip_eval_lst = []
    ref_str_eval_lst = []
    eval_reward = []
    for i in range(len(vid_lst)):
        vid = vid_lst[i]
        subid = subid_lst[i]
        ref_str_eval_lst.append(ref_dct[subid]) # string 
        ori_bsz = min(len(vrd_start_lsts[i][:args.top_n]), len(text_start_lsts[i][:args.top_n]))
        bsz = ori_bsz * vrd_option.train_rollouts

        # vrd_agent init
        # 1. init Graph
        vrd_trip_data = video_loader.get_graph_data(vid)
        vrd_train_graph = Knowledge_graph(vrd_option, video_loader, vrd_trip_data)
        vrd_agent.set_graph(vrd_train_graph)
        # 2. init start entities
        vrd_start_ent_str = [x[0] for x in vrd_start_lsts[i][:ori_bsz]] # [ent1, ent2, ...]
        vrd_start_rel_str = [x[1] for x in vrd_start_lsts[i][:ori_bsz]] # [rel1, rel2, ...]
        vrd_start_entity_id = [video_loader.entity2num[x] for x in vrd_start_ent_str]
        vrd_start_relation_id = [video_loader.relation2num[x] for x in vrd_start_rel_str]
        vrd_start_entity_id = np.repeat(vrd_start_entity_id, vrd_option.train_rollouts)
        vrd_start_relation_id = np.repeat(vrd_start_relation_id, vrd_option.train_rollouts)
        vrd_start_entity = torch.tensor(vrd_start_entity_id).to(args.device)
        vrd_relation = torch.tensor(vrd_start_relation_id).to(args.device)
        vrd_prev_relation = vrd_agent.get_dummy_start_relation(bsz).to(args.device)
        vrd_current_entity = vrd_start_entity
        vrd_traj = []
        for x in vrd_start_ent_str:
            for j in range(vrd_option.train_rollouts):
                vrd_traj.append([x])
        vrd_state = torch.zeros(1, 2, bsz, vrd_agent.m * vrd_option.state_embed_size).to(args.device)
        vrd_log_probs = np.zeros((bsz, )) * 1.0

        # text_agent init
        # 1. init graph
        text_trip_data = text_loader.get_graph_data(subid)
        text_train_graph = Knowledge_graph(text_option, text_loader, text_trip_data)
        text_agent.set_graph(text_train_graph)
        # 2. init start entities
        text_start_ent_str = [x[0] for x in text_start_lsts[i][:ori_bsz]] # [ent1, ent2, ...]
        text_start_rel_str = [x[1] for x in text_start_lsts[i][:ori_bsz]] # [rel1, rel2, ...]
        text_start_entity_id = [text_loader.entity2num[x] for x in text_start_ent_str]
        text_start_relation_id = [text_loader.relation2num[x] for x in text_start_rel_str]
        text_start_entity_id = np.repeat(text_start_entity_id, text_option.train_rollouts)
        text_start_relation_id = np.repeat(text_start_relation_id, text_option.train_rollouts)
        text_start_entity = torch.tensor(text_start_entity_id).to(args.device)
        text_relation = torch.tensor(text_start_relation_id).to(args.device)
        text_prev_relation = text_agent.get_dummy_start_relation(bsz).to(args.device)
        text_current_entity = text_start_entity
        text_traj = []
        for x in text_start_ent_str:
            for j in range(vrd_option.train_rollouts):
                text_traj.append([x])
        text_state = torch.zeros(1, 2, bsz, text_agent.m * text_option.state_embed_size).to(args.device)
        text_log_probs = np.zeros((bsz, )) * 1.0
        
        range_arr = torch.arange(bsz).to(args.device)
        
        # start grounding on graph 
        for step in range(vrd_option.max_step_length):
            vrd_loss, vrd_state, vrd_logits, vrd_action_idx, vrd_next_entity, vrd_chosen_relation= \
                vrd_agent.step(vrd_state, vrd_prev_relation, vrd_current_entity, vrd_relation, range_arr, text_state)
            # loss [B] logits [B, max_out] action_id [B]
            text_loss, text_state, text_logits, text_action_idx, text_next_entity, text_chosen_relation= \
                text_agent.step(text_state, text_prev_relation, text_current_entity, text_relation, range_arr, vrd_state)
            # loss [B] logits [B, max_out] action_id [B]

            vrd_log_probs += vrd_logits.clone().detach().cpu().numpy()[np.arange(vrd_log_probs.shape[0]), vrd_action_idx.cpu().numpy()]
            text_log_probs += text_logits.clone().detach().cpu().numpy()[np.arange(text_log_probs.shape[0]), text_action_idx.cpu().numpy()] # [B*num_rollouts]

            if i == 0:
                vrd_eval_logit_lst.append(vrd_logits) # [B, max_out]
                text_eval_logit_lst.append(text_logits) # [B, max_out]
            else:
                vrd_eval_logit_lst[step] = torch.cat([vrd_eval_logit_lst[step], vrd_logits], dim=0)
                text_eval_logit_lst[step] = torch.cat([text_eval_logit_lst[step], text_logits], dim=0)
            

            # all_action_id.append(action_id)
            vrd_prev_relation = vrd_chosen_relation
            vrd_current_entity = vrd_next_entity
            for j in range(bsz):
                vrd_traj[j].append(vrd_id2rel[(int)(vrd_chosen_relation[j])])
                vrd_traj[j].append(vrd_id2ent[(int)(vrd_next_entity[j])])    

            # all_action_id.append(action_id)
            text_prev_relation = text_chosen_relation
            text_current_entity = text_next_entity
            for j in range(bsz):
                text_traj[j].append(text_id2rel[(int)(text_chosen_relation[j])])
                text_traj[j].append(text_id2ent[(int)(text_next_entity[j])])   

        
        vrd_log_probs = np.reshape(vrd_log_probs, (ori_bsz, vrd_option.train_rollouts))
        sorted_vrd_indx = np.argsort(-vrd_log_probs) # [ori_B, rollouts]
        vrd_traj_lst = []
        for j in range(ori_bsz):
            vrd_traj_lst += vrd_traj[j*vrd_option.train_rollouts + sorted_vrd_indx[j][0]]
        if vrd_traj_lst == []:
            vrd_traj_lst.append('<pad>')
        vrd_eval_traj_lst.append(vrd_traj_lst)
        
        
        text_log_probs = np.reshape(text_log_probs, (ori_bsz, text_option.train_rollouts))
        sorted_text_indx = np.argsort(-text_log_probs) # [ori_B, rollouts]
        text_traj_lst = []
        for j in range(ori_bsz):
            text_traj_lst += text_traj[j*text_option.train_rollouts + sorted_text_indx[j][0]]
        if text_traj_lst == []:
            text_traj_lst.append('<pad>')
        text_eval_traj_lst.append(text_traj_lst)

        # cosine anneal
        if ref_lsts[i] == []:
            ref = ['<pad>', '<pad>', '<pad>']
        else:
            ref = ref_lsts[i][np.random.choice(len(ref_lsts[i]), 1)[0]]
        # ref_lst.append(ref)
        ref_trip_eval_lst.append(ref)

    # eval. by rouge
    # joint eval.
    assert len(vrd_eval_traj_lst) == len(text_eval_traj_lst)
    for j in range(len(vrd_eval_traj_lst)):
        if ref_trip_eval_lst[j] == ['<pad>'] * len(ref_trip_eval_lst[j]) or vrd_eval_traj_lst[j] == ['<pad>'] * len(vrd_eval_traj_lst[j]) or text_eval_traj_lst[j] == ['<pad>'] * len(text_eval_traj_lst[j]):
            eval_reward.append(0)
        else:
            # vrd_rr_reward.append(bleu_corpus(' '.join(traj), ref))
            # gold trip
            # eval_reward.append(rouge_n(tokenizer.tokenize(' '.join(ref_trip_eval_lst[j]).replace('<pad>', '')), tokenizer.tokenize(' '.join(vrd_eval_traj_lst[j] + text_eval_traj_lst[j]).replace('<pad>', ''))))
            eval_reward.append(rouge_n(' '.join(ref_trip_eval_lst[j]).replace('<pad>', ' ').split(), ' '.join(vrd_eval_traj_lst[j] + text_eval_traj_lst[j]).replace('<pad>', ' ').split()))
            # # gold str
            # eval_reward.append(rouge_n(tokenizer.tokenize(ref_str_eval_lst[j]), tokenizer.tokenize(' '.join(vrd_eval_traj_lst[j] + text_eval_traj_lst[j]).replace('<pad>', ''))))
            # print('debug')

    # assert args.reward == 'rouge'
    # for j, traj in enumerate(vrd_eval_traj_lst):
    #     if ref_trip_eval_lst[j//bsz] == ['<pad>'] * len(ref_trip_eval_lst[j//bsz]) or traj == ['<pad>'] * len(traj):
    #         vrd_eval_reward.append(0)
    #     else:
    #         # vrd_rr_reward.append(bleu_corpus(' '.join(traj), ref))
    #         # gold trip
    #         vrd_eval_reward.append(rouge_n(tokenizer.tokenize(' '.join(ref_trip_eval_lst[j//bsz]).replace('<pad>', '')), tokenizer.tokenize(' '.join(traj).replace('<pad>', ''))))
        
    # for j, traj in enumerate(text_eval_traj_lst):
        
    #     if ref_trip_eval_lst[j//bsz] == ['<pad>'] * len(ref_trip_eval_lst[j//bsz]) or traj == ['<pad>'] * len(traj):
    #         text_eval_reward.append(0)
    #     else:
    #         # text_rr_reward.append(bleu_corpus(' '.join(traj), ref))
    #         # gold trip
    #         text_eval_reward.append(rouge_n(tokenizer.tokenize(' '.join(ref_trip_eval_lst[j//bsz]).replace('<pad>', '')), tokenizer.tokenize(' '.join(traj).replace('<pad>', ''))))
    
    if rr_itr == 0:
        
        vrd_cum_discounted_reward = torch.zeros((4, 2), device=args.device)
        text_cum_discounted_reward = torch.zeros((4, 2), device=args.device)

        # vrd_eval_reward = torch.zeros((4, 2), device=args.device)
        # text_eval_reward = torch.zeros((4, 2), device=args.device)
        # vrd_cum_discounted_reward = calc_cum_discounted_reward_credit(vrd_option, vrd_eval_reward, 0)
        # text_cum_discounted_reward = calc_cum_discounted_reward_credit(text_option, text_eval_reward, 0)
    
    # save param and reward
    return copy.deepcopy(vrd_agent.state_dict()), copy.deepcopy(text_agent.state_dict()), copy.deepcopy(rl_optimizer.state_dict()), eval_reward, vrd_eval_traj_lst, text_eval_traj_lst, torch.mean(vrd_cum_discounted_reward), torch.mean(text_cum_discounted_reward)
    

