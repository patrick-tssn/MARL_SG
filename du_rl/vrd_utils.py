import sys
sys.path.append('nltk_data')

import torch
import numpy as np
import nltk.stem as ns

def calc_cum_discounted_reward(option, rewards):
    running_add = torch.zeros([rewards.size(0)])
    cum_disc_reward = torch.zeros([rewards.size(0), option.max_step_length])

    running_add = running_add.to(option.device)
    cum_disc_reward = cum_disc_reward.to(option.device)

    cum_disc_reward[:, option.max_step_length - 1] = rewards
    for t in reversed(range(option.max_step_length)):
        running_add = option.gamma * running_add + cum_disc_reward[:, t] # MC 计算回报的折扣因子
        cum_disc_reward[:, t] = running_add
    return cum_disc_reward
    # [B, 1] -> [B, T] e.g[gamma^2R, gammaR, R]
def calc_cum_discounted_reward_credit(option, main_rewards, extra_rewards):
    num_instances = main_rewards.size(0)
    running_add = torch.zeros([num_instances]).to(option.device)  # [original batch_size * num_rollout]
    cum_disc_reward = torch.zeros([num_instances, option.max_step_length]).to(option.device)  # [original batch_size * num_rollout, T]
    cum_disc_reward[:,option.max_step_length - 1] = main_rewards  # set the last time step to the reward received at the last state

    for t in reversed(range(1, option.max_step_length)):
        running_add = option.gamma * running_add + cum_disc_reward[:, t] + extra_rewards # approx_credits[t].to(self.device) * cluster_rewards
        cum_disc_reward[:, t-1] = running_add

    return cum_disc_reward

def entropy_reg_loss(all_logits):
    all_logits = torch.stack(all_logits, dim=2)  # [B, MAX_NUM_ACTIONS, T]
    entropy_loss = - torch.mean(torch.sum(torch.mul(torch.exp(all_logits), all_logits), dim=1))  # scalar
    return entropy_loss
    # 

def calc_reinforce_loss(baseline, all_loss, all_logits, cum_discounted_reward, decaying_beta):

    loss = torch.stack(all_loss, dim=1)  # [B, T]
    base_value = baseline.get_baseline_value()
    final_reward = cum_discounted_reward - base_value # cum_discounted_reward

    reward_mean = torch.mean(final_reward)
    reward_std = torch.std(final_reward) + 1e-6
    final_reward = torch.div(final_reward - reward_mean, reward_std) # (x-u)/n

    loss = torch.mul(loss, final_reward)  # [B, T] negative 
    entropy_loss = decaying_beta * entropy_reg_loss(all_logits) # 增加选择的多样性

    total_loss = torch.mean(loss) - entropy_loss  # scalar

    return total_loss


#对每个句子的所有词向量取均值，来生成一个句子的vector
#sentence是输入的句子，size是词向量维度，w2v_model是训练好的词向量模型
def build_sentence_vector(sentence,size,w2v_model):
    vec=np.zeros(size).reshape((size))
    count=0
    for word in sentence:
        try:
            vec+=w2v_model[word].reshape((size))
            count+=1
        except KeyError:
            continue
    if count!=0:
        vec/=count
    return vec
 
#计算两个句向量的余弦相似性值
def cosine_similarity(vec1, vec2):
    a= np.array(vec1)
    b= np.array(vec2)
    cos1 = np.sum(a * b)
    if cos1 == 0.:
        return cos1
    cos21 = np.sqrt(sum(a ** 2))
    cos22 = np.sqrt(sum(b ** 2))
    cosine_value = cos1 / float(cos21 * cos22)
    return cosine_value
 
#输入两个句子，计算两个句子的余弦相似性
def compute_cosine_similarity(sents_1, sents_2, w2v_model):
    lemmatizer = ns.WordNetLemmatizer()
    lem_sent1 = [lemmatizer.lemmatize(w) for w in sents_1]
    lem_sent2 = [lemmatizer.lemmatize(w) for w in sents_2]
    size=300
    vec1=build_sentence_vector(lem_sent1,size,w2v_model)
    vec2=build_sentence_vector(lem_sent2,size,w2v_model)
    similarity = cosine_similarity(vec1, vec2)
    return similarity

def padding(seq, pad_token):
    max_len = max([i.size(0) for i in seq])
    if len(seq[0].size()) == 1:
        result = torch.ones((len(seq), max_len)).long() * pad_token
    else:
        result = torch.ones((len(seq), max_len, seq[0].size(-1))).float()
    for i in range(len(seq)):
        result[i, :seq[i].size(0)] = seq[i]
    return result

def rouge_n(gold, pred, ignore=None):
    hit_n = 0
    if ignore is None:
        for token in gold:
            if token in pred:
                hit_n += 1
        return hit_n / len(gold)
    else:
        sum_len = 0
        for token in gold:
            if token == ignore:
                break
            else:
                if token in pred:
                    hit_n += 1
            sum_len += 1
        return hit_n / sum_len

from nltk.translate import bleu_score as nltkbleu

def bleu_corpus(hypothesis, references):
    from nltk.translate.bleu_score import corpus_bleu
    b1 = corpus_bleu(references, hypothesis, weights=(1.0/1.0,), smoothing_function=nltkbleu.SmoothingFunction(epsilon=1e-12).method1)
    b2 = corpus_bleu(references, hypothesis, weights=(1.0/2.0, 1.0/2.0), smoothing_function=nltkbleu.SmoothingFunction(epsilon=1e-12).method1)
    b3 = corpus_bleu(references, hypothesis, weights=(1.0/3.0, 1.0/3.0, 1.0/3.0), smoothing_function=nltkbleu.SmoothingFunction(epsilon=1e-12).method1)
    b4 = corpus_bleu(references, hypothesis, weights=(1.0/4.0, 1.0/4.0, 1.0/4.0, 1.0/4.0), smoothing_function=nltkbleu.SmoothingFunction(epsilon=1e-12).method1)
    # return (b1, b2, b3, b4)
    return b1

def bleu_metric(hypothesis, references):
    assert type(hypothesis) == type(references)
    if type(hypothesis) is str:
        lemmatizer = ns.WordNetLemmatizer()
        hypothesis = [lemmatizer.lemmatize(w.lower()) for w in hypothesis.split()]
        references = [lemmatizer.lemmatize(w.lower()) for w in references.split()]
        hypothesis = [' '.join(hypothesis)]
        references = [' '.join(references)]
    # elif type(hypothesis) is list:
    return bleu_corpus(hypothesis, references)

if __name__ == '__main__':
    # print(bleu_corpus(['test'], ['tests']))
    print(bleu_corpus([1, 2], [2, 3, 4]))