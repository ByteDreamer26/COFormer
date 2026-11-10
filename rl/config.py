from argparse import ArgumentParser

def str2bool(x):
    assert x == "True" or x == "False"
    return True if x == "True" else False

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--worker-num", type=int, default=0)
    parser.add_argument("--adj-range", type=str2bool, default=False)
    parser.add_argument("--ckpt-path", type=str, default=None,)
    parser.add_argument("--snapshot-path", type=str, default=None,)
    parser.add_argument("--data-base-path", type=str, default=None,)

    # check eval/train loss
    parser.add_argument("--data-num-bp", type=int, default=200, help="The bp episode num used for training. ")
    parser.add_argument("--data-num-tsp", type=int, default=200, help="The tsp episode num used for training. ")
    parser.add_argument("--data-num-atsp", type=int, default=200, help="The atsp episode num used for training. ")
    parser.add_argument("--data-num-op", type=int, default=200, help="The op episode num used for training. ")
    parser.add_argument("--data-num-cvrp", type=int, default=200, help="The cvrp episode num used for training. ")
    parser.add_argument("--data-num-pctsp", type=int, default=200, help="The pctsp episode num used for training. ")
    parser.add_argument("--data-num-spctsp", type=int, default=200, help="The spctsp episode num used for training. ")
    parser.add_argument("--data-num-ffsp", type=int, default=200, help="The ffsp episode num used for training. ")
    parser.add_argument("--data-num-mis", type=int, default=200, help="The mis episode num used for training. ")
    parser.add_argument("--data-num-obp3d", type=int, default=200, help="The obp3d episode num used for training. ")

    parser.add_argument("--batch-num", type=int, default=0,)
    parser.add_argument("--batch-size", type=int, default=100,)
    parser.add_argument("--eval-batch-num", type=int, default=0,)
    parser.add_argument("--eval-batch-size", type=int, default=100,)
    parser.add_argument("--test-batch-num", type=int, default=0,)
    parser.add_argument("--test-batch-size", type=int, default=100,)
    
    # check policy
    parser.add_argument(
        "--n-position", type=int, default=1024,
        help="The maximum sequence length that this model might ever be used with."
        "Typically set this to something large just in case (e.g., 512 or 1024 or 2048)",
    )

    parser.add_argument("--eval-envs", nargs='+', type=str, default=['Env_BP_V1', ])
    parser.add_argument("--problem-batch-size-per-gpu", type=int, default=240,)
    parser.add_argument("--problem-batch-num", type=int, default=20,)
    parser.add_argument("--total-problem-num", type=int, default=20, help="equals to (problem_batch_size_per_gpu * world_size * problem_batch_num)")

    parser.add_argument("--eval-problem-set", type=str, default='problem', choices={'problem', 'train_problem'})
    parser.add_argument("--use-kvcache", type=str2bool, default=False, nargs="?", const=True)
    parser.add_argument("--use-ddp-env", type=str2bool, default=False, nargs="?", const=True)
    parser.add_argument("--regen-times", type=int, default=16,)
    parser.add_argument("--use-default-policy-obj", type=str2bool, default=False)
    
    parser.add_argument(
        "--prompt-strategy", type=str, default="stochastic_subseq;moving_prompt",
        choices={"stochastic_timestep;moving_prompt", "stochastic_subseq;moving_prompt"},
    )

    return parser.parse_args()