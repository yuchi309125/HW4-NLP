import os
import argparse
from tqdm import tqdm

import torch 
import torch.nn as nn
import numpy as np
import wandb

from t5_utils import initialize_model, initialize_optimizer_and_scheduler, save_model, load_model_from_checkpoint, setup_wandb
from transformers import GenerationConfig
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records
import pickle
from utils import compute_metrics, save_queries_and_records

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0

def get_args():
    '''
    Arguments for training. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(description='T5 training loop')

    # Model hyperparameters
    parser.add_argument('--finetune', action='store_true', help="Whether to finetune T5 or not")
    
    # Training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"],
                        help="What optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--scheduler_type', type=str, default="cosine", choices=["none", "cosine", "linear"],
                        help="Whether to use a LR scheduler and what type to use if so")
    parser.add_argument('--num_warmup_epochs', type=int, default=0,
                        help="How many epochs to warm up the learning rate for if using a scheduler")
    parser.add_argument('--max_n_epochs', type=int, default=0,
                        help="How many epochs to train the model for")
    parser.add_argument('--patience_epochs', type=int, default=0,
                        help="If validation performance stops improving, how many epochs should we wait before stopping?")

    parser.add_argument('--use_wandb', action='store_true',
                        help="If set, we will use wandb to keep track of experiments")
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help="How should we name this experiment?")

    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--debug', action='store_true')


    args = parser.parse_args()
    return args

def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_f1 = -1
    epochs_since_improvement = 0

    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir
    experiment_name = 'ft_experiment'
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')
    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss}")

        (eval_loss, 
        record_f1, 
        record_em, 
        sql_em, 
        error_msgs) = eval_epoch(args, model, dev_loader,
                                gt_sql_path, model_sql_path,
                                gt_record_path, model_record_path)

        # error_msgs 是一個 list，每個元素是錯誤訊息（或 ""）
        num_errors = sum(1 for e in error_msgs if e != "")
        error_rate = num_errors / len(error_msgs)

        print(f"Epoch {epoch}: Dev loss: {eval_loss}, Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
        print(f"Epoch {epoch}: {error_rate*100:.2f}% of outputs led to SQL errors")


        if args.use_wandb:
            result_dict = {
                'train/loss' : tr_loss,
                'dev/loss' : eval_loss,
                'dev/record_f1' : record_f1,
                'dev/record_em' : record_em,
                'dev/sql_em' : sql_em,
                'dev/error_rate' : error_rate,
            }
            wandb.log(result_dict, step=epoch)

        if record_f1 > best_f1:
            best_f1 = record_f1
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        save_model(checkpoint_dir, model, best=False)
        if epochs_since_improvement == 0:
            save_model(checkpoint_dir, model, best=True)

        if epochs_since_improvement >= args.patience_epochs:
            break

def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    PAD_IDX = 0  # T5 pad token

    for encoder_input, encoder_mask, decoder_inputs, decoder_targets, _ in tqdm(train_loader):
        optimizer.zero_grad()
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        # 使用內建 loss（自動 ignore pad）
        outputs = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            labels=decoder_targets,
        )
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            # 這裡不需要 non_pad，因為 loss 已經是 normalized 過的
            total_loss += loss.item()
            total_tokens += 1

    return total_loss / total_tokens

        
def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    generated_sql = []

    with torch.no_grad():
        for encoder_input, encoder_mask, decoder_inputs, decoder_targets, init_dec in dev_loader:

            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            decoder_targets = decoder_targets.to(DEVICE)

            # 1. loss
            outputs = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                labels=decoder_targets,
            )
            loss = outputs.loss

            total_loss += loss.item()
            total_tokens += 1

            # 2. generate
            gen_ids = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_length=400,
                num_beams=4,
                early_stopping=True,
            )

            sql_texts = model.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            generated_sql.extend([s.strip() for s in sql_texts])

    # 3. save
    save_queries_and_records(generated_sql, model_sql_path, model_record_path)

    # 4. metrics
    sql_em, record_em, record_f1, sql_errors = compute_metrics(
        gt_sql_pth,
        model_sql_path,
        gt_record_path,
        model_record_path,
    )

    avg_loss = total_loss / total_tokens

    return avg_loss, record_f1, record_em, sql_em, sql_errors

        
def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    """
    Run inference on the test set.
    Generate SQL predictions and convert them into database records.
    Save both .sql and .pkl files.
    """
    model.eval()
    generated_sql = []

    with torch.no_grad():
        for encoder_input, encoder_mask, init_decoder_input in test_loader:

            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)

            # Generate SQL query (beam search)
            outputs = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_length=400,
                num_beams=4,
                early_stopping=True
            )

            # Convert token IDs → SQL strings
            sql_texts = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            generated_sql.extend([s.strip() for s in sql_texts])

    # ---- Save SQL predictions ----
    with open(model_sql_path, "w") as f:
        for s in generated_sql:
            f.write(s + "\n")

    # ---- Convert SQL → records (official utils) ----
    save_queries_and_records(
    generated_sql,
    model_sql_path,
    model_record_path
    )


    print(f"[Test] Saved SQL predictions to: {model_sql_path}")
    print(f"[Test] Saved predicted records to: {model_record_path}")


def main():

    # Get key arguments
    args = get_args()
    if args.use_wandb:
        # Recommended: Using wandb (or tensorboard) for result logging can make experimentation easier
        setup_wandb(args)
    

    # Load the data and the model
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    if args.debug:
            print(">>> DEBUG MODE: reducing dataset size to small batches")
            train_loader = list(train_loader)[:2]
            dev_loader = list(dev_loader)[:1]
            test_loader = list(test_loader)[:1]

    model = initialize_model(args)
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    # Train 
    train(args, model, train_loader, dev_loader, optimizer, scheduler)

    # Evaluate
    model = load_model_from_checkpoint(args, best=True)
    model.eval()
    
    # Dev set
    experiment_name = 'experiment'
    model_type = 'ft' if args.finetune else 'scr'
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')

    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, error_msgs = eval_epoch(
    args, model, dev_loader,
    gt_sql_path, model_sql_path,
    gt_record_path, model_record_path
    )

    # compute SQL error rate
    num_errors = sum(1 for e in error_msgs if e != "")
    error_rate = num_errors / len(error_msgs)

    print(f"Dev set results: Loss: {dev_loss}, Record F1: {dev_record_f1}, Record EM: {dev_record_em}, SQL EM: {dev_sql_em}")
    print(f"Dev set results: {error_rate*100:.2f}% of the generated outputs led to SQL errors")


    # Test set
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_test.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_test.pkl')
    test_inference(args, model, test_loader, model_sql_path, model_record_path)

if __name__ == "__main__":
    main()
