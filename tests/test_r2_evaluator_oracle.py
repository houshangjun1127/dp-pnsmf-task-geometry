"""Independent scalar Python oracle, synthetic edge cases only; no training."""
from pathlib import Path
import sys, json, math, unittest, itertools
import numpy as np
import torch
from src.evaluation.full_catalog_stable import evaluate_factors

def oracle(users,items,targets,train,ids,k):
    records=[]
    for ui,user in enumerate(users):
        if not targets[ui]: continue
        candidates=[]
        for ii,item in enumerate(items):
            if ids[ii] in train[ui]: continue
            score=sum(float(a)*float(b) for a,b in zip(user,item))
            candidates.append((score,int(ids[ii])))
        ranked=[item for score,item in sorted(candidates,key=lambda x:(-x[0],x[1]))[:k]]
        hit=[int(item in targets[ui]) for item in ranked]
        dcg=sum(h/math.log2(pos+2) for pos,h in enumerate(hit))
        idcg=sum(1/math.log2(pos+2) for pos in range(min(k,len(targets[ui]))))
        records.append({'ranked':ranked,'ndcg':dcg/idcg,'recall':sum(hit)/len(targets[ui])})
    return {'ndcg':sum(r['ndcg'] for r in records)/len(records),'recall':sum(r['recall'] for r in records)/len(records),'records':records}

class EvaluationOracleTests(unittest.TestCase):
    def check(self,u,v,t,tr,ids,k,batch=2):
        out=evaluate_factors(np.asarray(u),np.asarray(v),[np.array(x,dtype=np.int64) for x in t],
            [np.array(x,dtype=np.int64) for x in tr],item_ids=ids,cutoff=k,batch_size=batch,include_rankings=True)
        ref=oracle(u,v,t,tr,ids,k)
        self.assertAlmostEqual(out[f'ndcg@{k}'],ref['ndcg'],places=14)
        self.assertAlmostEqual(out[f'recall@{k}'],ref['recall'],places=14)
        self.assertEqual([r['original_item_ids'] for r in out['rankings']],[r['ranked'] for r in ref['records']])
        return out

    def test_zero_factors_and_empty_train_original_id_order(self):
        r=self.check([[0]],[[3],[0],[1],[-1]],[[10,30]],[[]],np.array([40,10,30,20]),2)
        self.assertEqual(r['rankings'][0]['original_item_ids'],[10,20])
        self.assertEqual(r['recall@2'],.5)
        self.assertEqual(r['diagnostics']['topK_boundary_tie_users'],1)

    def test_positive_ties_and_rare_cold_zero_vectors(self):
        r=self.check([[1]],[[2],[2],[0],[0],[-1]],[[1,4]],[[2]],np.arange(1,6),3)
        self.assertEqual(r['rankings'][0]['original_item_ids'],[1,3,4])
        self.assertEqual(r['recall@3'],1.)

    def test_full_targets_denominator_not_capped(self):
        r=self.check([[1]],[[100] for _ in range(10)]+[[50] for _ in range(20)]+[[-100] for _ in range(30)],
            [list(range(1,11))+list(range(31,61))],[[]],np.arange(1,61),20)
        self.assertEqual(r['recall@20'],.25) # 10 hits / 40 targets, not .5

    def test_mask_all_original_train_not_cap_or_validation(self):
        r=self.check([[1]],[[5],[4],[3],[2],[1]],[[4]],[[1,2,3]],np.arange(1,6),2)
        self.assertEqual(r['rankings'][0]['original_item_ids'],[4,5])

    def test_candidate_shortage_keeps_user_never_fills_masked(self):
        r=self.check([[1],[1]],[[5],[4],[3],[2]],[[4],[2]],[[1,2,3],[]],np.arange(1,5),3)
        self.assertEqual(r['users'],2)
        self.assertEqual(r['diagnostics']['candidate_shortage_users'],[0])
        self.assertEqual(r['rankings'][0]['original_item_ids'],[4])

    def test_empty_test_user_omitted_not_empty_train(self):
        r=self.check([[1],[0]],[[1],[0],[-1]],[[],[2]],[[],[]],np.arange(1,4),2)
        self.assertEqual(r['users'],1)
        self.assertEqual(r['diagnostics']['test_empty_users'],1)

    def test_item_row_permutations_same_public_id_order(self):
        base=None
        ids=[8,2,15,1]
        items=[[1],[0],[1],[0]]
        for perm in itertools.permutations(range(4)):
            r=self.check([[1]],[items[i] for i in perm],[[1,15]],[[]],np.array([ids[i] for i in perm]),3)
            ranked=r['rankings'][0]['original_item_ids']
            if base is None: base=ranked
            self.assertEqual(ranked,base)

    def test_close_unequal_scores_not_merged(self):
        r=self.check([[1]],[[1],[1+1e-12],[0]],[[2]],[[]],np.arange(1,4),2)
        self.assertEqual(r['rankings'][0]['original_item_ids'],[2,1])

    def test_exhaustive_tie_score_patterns_against_oracle(self):
        for vals in itertools.product([-1,0,1],repeat=4):
            self.check([[1]],[[x] for x in vals],[[1,3]],[[2]],np.array([1,2,3,4]),2)

    def test_block_sizes_same_macro_and_rankings(self):
        outputs=[]
        for b in [1,2,8]:
            outputs.append(self.check([[1],[0],[2]],[[1],[1],[0],[-1]],[[2],[3],[1,4]],[[],[1],[]],np.arange(1,5),2,b))
        self.assertEqual(outputs[0]['rankings'],outputs[1]['rankings'])
        self.assertEqual(outputs[0]['rankings'],outputs[2]['rankings'])

    def test_duplicate_targets_and_train_rejected(self):
        for t,tr in [([[1,1]],[[]]),([[2]],[[1,1]])]:
            with self.assertRaisesRegex(ValueError,'duplicate'):
                evaluate_factors([[1]],[[1],[0]],t,tr,cutoff=1)

    def test_overlap_and_out_of_catalog_rejected(self):
        with self.assertRaisesRegex(ValueError,'overlap'):
            evaluate_factors([[1]],[[1],[0]],[[1]],[[1]],cutoff=1)
        with self.assertRaisesRegex(ValueError,'outside'):
            evaluate_factors([[1]],[[1],[0]],[[3]],[[]],cutoff=1)

    def test_nan_inf_duplicate_mapping_empty_targets_rejected(self):
        for value in [float('nan'),float('inf')]:
            with self.assertRaisesRegex(ValueError,'NaN/Inf'):
                evaluate_factors([[value]],[[1],[0]],[[1]],[[]],cutoff=1)
        with self.assertRaisesRegex(ValueError,'duplicate'):
            evaluate_factors([[1]],[[1],[0]],[[1]],[[]],item_ids=[1,1],cutoff=1)
        with self.assertRaisesRegex(ValueError,'nonempty'):
            evaluate_factors([[1]],[[1],[0]],[[]],[[]],cutoff=1)

