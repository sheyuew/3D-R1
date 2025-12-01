# -*- coding: utf-8 -*-
import os, json, random, numpy as np, torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from eval_utils.evaluate_cold import evaluate

from dataset.scannet_base_dataset import ScanNetBaseDataset, DatasetConfig, BASE

SPECIAL_TOKENS = ["<think>", "</think>", "<answer>", "</answer>"]

class Dataset(ScanNetBaseDataset):
    def __init__(
        self,
        args,
        dataset_config: DatasetConfig,
        split_set="all",
        num_points=40000,
        use_color=False,
        use_normal=False,
        use_multiview=False,
        use_height=False,
        augment=False,
        use_additional_encoders=False,
        use_rl_training=False,
    ):
        super().__init__(
            args,
            dataset_config,
            split_set=split_set,
            num_points=num_points,
            use_color=use_color,
            use_normal=use_normal,
            use_multiview=use_multiview,
            use_height=use_height,
            augment=augment,
            use_random_cuboid=False,
            use_additional_encoders=use_additional_encoders,
        )
        self.use_rl_training=use_rl_training

        self.task_name = 'cold-start'
        self.split = split_set
        self.eval_func = evaluate
        self.max_prompts = 1
        assert split_set in ["train", "val"]
        ann_path = os.path.join(
            BASE,"data","Scene30K",f"Scene-30K.jsonl",
        )
        scene_path = os.path.join(BASE, "data", "scene_caption.json")
        self.scene_description = json.load(open(scene_path,'r'))
        self.annotations = [json.loads(l) for l in open(ann_path)]
        self.scan_names  = sorted({a["scene_id"] for a in self.annotations})
        print(f"[SceneR1Cold-QA-PC]: "
              f"{len(self.annotations)} Q&A  from {len(self.scan_names)} scans")

        self.tokenizer = AutoTokenizer.from_pretrained(args.vocab)

        self.tokenizer.padding_side = 'left'
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if not all(tok in self.tokenizer.vocab for tok in SPECIAL_TOKENS):
            self.tokenizer.add_tokens(SPECIAL_TOKENS, special_tokens=True)

        self.qtokenizer = AutoTokenizer.from_pretrained(args.qformer_vocab)
        
        if not all(tok in self.qtokenizer.vocab for tok in SPECIAL_TOKENS):
            self.qtokenizer.add_tokens(SPECIAL_TOKENS, special_tokens=True)

        self.cfg_q = dict(max_length=512,
                padding='max_length',
                truncation="longest_first", return_tensors="np")

        self.cfg = dict(
            max_length=args.max_des_len,
            padding='max_length',
            truncation="longest_first",
            return_tensors="np",
        )

    # ------------------------------------------------------------------
    def __len__(self): return len(self.annotations)

    def __getitem__(self, idx):
        sample   = self.annotations[idx]
        scene_id = sample["scene_id"]

        
        ret = self._get_scan_data(scene_id)       
        description = random.choice(self.scene_description[scene_id]['captions'])

        q, cot = sample["question"].strip(), sample["cot"].strip()

        source_txt  = f"given the 3D scene and description:{description}, think step by step and answer the following question:{q}."
        target_txt  = f"{source_txt} Output format:<think>...reasoning...</think><answer>..final answer...</answer> {cot} {self.tokenizer.eos_token}"
        enc_source  = self.tokenizer.batch_encode_plus([source_txt], **self.cfg)
        enc_target  = self.tokenizer.batch_encode_plus([target_txt], **self.cfg)


        qformer_ids = self.qtokenizer.batch_encode_plus([f"given the 3D scene and description:{description}, think step by step and answer the following question:{q}"], **self.cfg_q)

        box_query = np.zeros((self.max_prompts, 8, 3))
        box_mask = np.zeros((self.max_prompts,))
        click_query = np.zeros((self.max_prompts, 3))
        click_mask = np.zeros((self.max_prompts,))
        
        
        ret["box_query"]   = box_query.astype(np.float32)
        ret["box_mask"]    = box_mask.astype(np.float32)
        ret["click_query"] = click_query.astype(np.float32)
        ret["click_mask"]  = click_mask.astype(np.float32)

       
        ret["input_ids"]        = enc_target["input_ids"][0].astype(np.int64)
        ret["attention_mask"]   = enc_target["attention_mask"][0].astype(np.float32)
        
        # # Improved gradient mask computation to prevent extreme values
        # gradient_mask = enc_target['attention_mask'][0] - enc_source['attention_mask'].astype(np.float32)
        
        # # Ensure gradient mask is non-negative and properly normalized
        # gradient_mask = np.maximum(gradient_mask, 0.0)
        # ret["gradient_mask"] = gradient_mask.astype(np.float32)
        # =================【修复mask】=================
        # 1. 获取 Source (Prompt) 和 Target (Prompt + Answer) 的真实有效长度
        len_source = int(enc_source['attention_mask'][0].sum())
        len_target = int(enc_target['attention_mask'][0].sum())
        len_total = len(ret["input_ids"]) 
        
        # 2. 初始化全 0 mask
        gradient_mask = np.zeros(len_total, dtype=np.float32)
        
        # 3. 计算 Answer 在左填充序列中的起始位置
        # 逻辑：总长度 - Target有效长度(包含Prompt+Answer) + Source有效长度(Prompt)
        # 剩下的右边部分就是 Answer
        start_answer_idx = len_total - len_target + len_source
        
        # 4. 将 Answer 部分设为 1
        if start_answer_idx < len_total:
            gradient_mask[start_answer_idx:] = 1.0
            
        ret["gradient_mask"] = gradient_mask.astype(np.float32)
        # ==========================================================

        # #=================【诊断探针 START】=================
        # # 只记录前 10 个样本，避免日志文件过大
        # if idx % 100 == 0:
        #     try:
        #         # 1. 获取关键长度信息
        #         len_source = int(enc_source['attention_mask'][0].sum())  # 提问的长度
        #         len_target = int(enc_target['attention_mask'][0].sum())  # 提问+回答的长度
        #         len_total = len(ret["input_ids"])                       # 总长度 (256)
                
        #         # 2. 推导：在“左填充”模式下，Answer 应该在哪里？
        #         # 结构应该是: [Pad, ..., Pad, Prompt, Answer]
        #         # 有效数据区长度 = len_target
        #         # 有效数据起始点 = len_total - len_target
        #         # Answer 起始点  = 有效数据起始点 + len_source
        #         expected_start = len_total - len_target + len_source
                
        #         # 3. 观测：现在的 Mask 实际上在哪里？
        #         mask_indices = np.where(gradient_mask > 0)[0]
        #         actual_start = mask_indices[0] if len(mask_indices) > 0 else -1
                
        #         # 4. 写入日志
        #         with open("/root/3D-R1/debug_mask_diagnosis.txt", "w") as f:
        #             f.write(f"\n>>> Sample {idx} Diagnosis <<<\n")
        #             f.write(f"Info: Total Len={len_total} | Valid Len={len_target} | Prompt Len={len_source}\n")
        #             f.write(f"Expect Answer Start: {expected_start}\n")
        #             f.write(f"Actual Mask Start:   {actual_start}\n")
                    
        #             # 自动判断结论
        #             if actual_start == -1:
        #                 f.write("RESULT: Mask is empty! (ERROR)\n")
        #             elif actual_start < expected_start - 2: # 允许1-2个token的误差
        #                 f.write(f"RESULT: ❌ Mask TOO EARLY! Diff = {expected_start - actual_start} tokens.\n")
        #                 f.write("       (Means mask is covering the PROMPT, confirming the bug.)\n")
        #             elif actual_start > expected_start + 2:
        #                 f.write("RESULT: ❌ Mask TOO LATE! (Unusual)\n")
        #             else:
        #                 f.write("RESULT: ✅ Mask looks correct.\n")
                        
        #             # 可视化末尾 50 个 Token 的 Mask 状态
        #             f.write(f"Mask (last 50): {gradient_mask[-50:].astype(int).tolist()}\n")
        #             f.flush()
        #             os.fsync(f.fileno())
        #     except Exception as e:
        #         pass
        # # =================【诊断探针 END】=================
        ret['scan_idx'] = np.array(idx).astype(np.int64)
        ret["instruction"] = enc_source['input_ids'][0].astype(np.int64)
        ret["instruction_mask"] = enc_source['attention_mask'][0].astype(np.float32)

        ret["qformer_input_ids"]      = qformer_ids["input_ids"][0].astype(np.int64)
        ret["qformer_attention_mask"] = qformer_ids["attention_mask"][0].astype(np.float32)

        
        return ret