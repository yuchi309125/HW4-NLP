import os

import torch

import transformers
from transformers import AutoTokenizer
from transformers import T5ForConditionalGeneration, T5Config, T5TokenizerFast
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
import wandb

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def setup_wandb(args):
    # Implement this if you wish to use wandb in your experiments
    pass

def initialize_model(args):
    model_name = "google-t5/t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if args.finetune:
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    else:
        # 從同樣架構但隨機初始化開始
        config = T5Config.from_pretrained(model_name)
        model = T5ForConditionalGeneration(config)

    model.tokenizer = tokenizer
    model.to(DEVICE)
    return model


def mkdir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass

def save_model(checkpoint_dir, model, best):
    """
    Save model checkpoint.
    best = True → save to best.pth
    best = False → save to last.pth
    """
    mkdir(checkpoint_dir)

    filename = "best.pth" if best else "last.pth"
    save_path = os.path.join(checkpoint_dir, filename)

    torch.save(model.state_dict(), save_path)
    print(f"[Checkpoint] Saved model to {save_path}")


def load_model_from_checkpoint(args, best):
    """
    Load model from checkpoint.
    If checkpoint does not exist → return None.
    """
    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)

    filename = "best.pth" if best else "last.pth"
    ckpt_path = os.path.join(checkpoint_dir, filename)

    if not os.path.exists(ckpt_path):
        print(f"[Checkpoint] No checkpoint found at {ckpt_path}, returning None")
        return None

    # Reinitialize model structure
    model = initialize_model(args)

    # Load saved weights
    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state_dict)

    print(f"[Checkpoint] Loaded model from {ckpt_path}")
    return model


def initialize_optimizer_and_scheduler(args, model, epoch_length):
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler

def initialize_optimizer(args, model):
    decay_parameters = get_parameter_names(model, transformers.pytorch_utils.ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8, betas=(0.9, 0.999)
        )
    else:
        pass

    return optimizer
        
def initialize_scheduler(args, optimizer, epoch_length):
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "none":
        return None
    elif args.scheduler_type == "cosine":
        return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif args.scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    else:
        raise NotImplementedError

def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

