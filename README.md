# COFormer: A Foundation Model for Solving Combinatorial Optimization Problems

COFormer is a unified model designed to simultaneously solve diverse combinatorial optimization problems (COPs).

## 1. Data Generation

1. **Generate Raw COP Data**

   Run the following script to generate raw data using an expert solver:
   ```bash
   ./data/gen_raw_data.py
   ```
   For example, set the parameters as:
   ```python
   scale = 20       
   cop_type = 'bp'
   ```
   The knapsack problem data will be saved in `./data/raw/bp/bp20`.

2. **Convert Raw Data**

   Convert the raw data into MDP trajectories by running:
   ```bash
   ./data/used/BP_V1/load_raw_data.py
   ```
   - For COFormer, supported MDP include: `BP_V1`, `TSP_V3`, `PCTSP_V3`, `CVRP_V3`, `OP_V4`, `SPCTSP_V3`, `FFSP_V1`, `OBP3D_V1`.
   - For the GATO baseline, supported MDP include: `BP_V2`, `TSP_V2`, `PCTSP_V2`, `CVRP_V2`, `OP_V2`, `SPCTSP_V1`, `FFSP_V3`.



## 2. Training

### 2.1 Dynamics Forward Stage

Train the dynamic prediction model by running:
```bash
./train/train_ddp.py
```
**Key Training Configuration**:

```python
args.is_obs_pretrain = True
args.init_from = None
args.exp_name = 'pretrain'
```

**Key Model Configuration:**

- **For COFormer:**
   ```python
   args.model = 'llama'
   args.use_prefix = True
   args.use_prompt = False
   ```
- **For GATO baseline:**
   ```python
   args.model = 'transformer_xl'
   args.use_prefix = False
   args.use_prompt = True
   ```

The best checkpoint will be saved at `./out/pretrain_xxx/sft/best/xxx.pt`.

### 2.2 Policy Generation Stage

Train the policy generation model by running:
```bash
./train/train_ddp.py
```
**Key Training Configuration**:

```python
args.is_obs_pretrain = False
args.init_from = f'{base_path}/out/pretrain_XXX/sft/best/xxx.pt'
args.exp_name = 'policy'
```
The best checkpoint will be saved at `./out/policy_xxx/sft/best/xxx.pt`.

### 2.3 Reinforcement Finetuning Stage

Further finetune the model's performance on specific COPs by running:
```bash
./rl/grpo.py
```
Set the checkpoint path:
```python
rl_args.sft_ckpt_path = f'{base_path}/out/policy_xxx/sft/best/xxx.pt'
```
The best checkpoint will be saved at `./out/policy_xxx/rl/best/xxx.pt`.



## 3. Evaluation

Evaluate the model performance by running:
```bash
./eval/evaluate_ddp.py
```
**Key Training Configuration**:

```python
eval_args.sft_ckpt_path = f'{base_path}/out/policy_xxx/sft/best/xxx.pt'
```