import os, sys, pdb
import random
import numpy as np
import torch
from tqdm import tqdm as progress_bar

from utils.arguments import solicit_params
from utils.help import set_seed, setup_gpus, check_directories, prepare_inputs, device
from utils.load import load_data, load_tokenizer, load_candidates, get_optimizer, get_scheduler
from utils.process import process_data, setup_dataloader
from utils.evaluate import ast_t5_report, quantify, qualify

from components.datasets import ActionDataset, CascadeDataset
from components.tools import ExperienceLogger
from components.models import ActionStateTracking, CascadeDialogSuccess
from components.generative_models import ActionStateTracking, CascadeDialogSuccess


def run_main(args, datasets, model, exp_logger):
  if args.task == 'cds':
    utt_data = load_candidates(args)
    model.add_candidate_data(*utt_data)
  kb_labels = {}
  if args.use_kb:
    kb_labels['intent'] = list(model.mappings['intent'].keys())
    kb_labels['action'] = list(model.mappings['action'].keys())

  exp_logger.init_tb_writers()
  if args.do_eval:
    result = run_eval(args, datasets, model, exp_logger, kb_labels, split='test')
    results = dict((k + f'_{args.filename}', v) for k, v in result.items())
    print('Test Results -', results)
  else:
    run_train(args, datasets, model, exp_logger, kb_labels)


def run_train(args, datasets, model, exp_logger, kb_labels):
  dataloader, num_examples = setup_dataloader(datasets, args.batch_size, split='train')
  t_total = len(dataloader) // args.grad_accum_steps * args.epochs    
  exp_logger.start_train(num_examples, total_step=t_total)
  optimizer = get_optimizer(args, model)
  scheduler = get_scheduler(args, optimizer, t_total)
  loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1)
  model.zero_grad()

  for epoch in range(args.epochs):
    print('========== Epoch {} =========='.format(epoch))
    model.train()
    total_loss = 0
    for step, batch in progress_bar(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {exp_logger.epoch}"):
      batch = tuple([t.to(device) for t in batch[:-1]]+[batch[-1]])

      if args.task == 'ast':
        full_history, targets, context_tokens, _ = prepare_inputs(args, batch)
        # scores = model(full_history, context_tokens)
        # loss = model.ast_loss(scores, targets, loss_func)
        loss = model(full_history, targets, context_tokens)
      elif args.task == 'cds':
        full_history, targets, context_tokens, tools = prepare_inputs(args, batch)
        scores = model(full_history, context_tokens, tools)
        loss = model.cds_loss(scores, targets, loss_func)

      if args.grad_accum_steps > 1:
        loss = loss / args.grad_accum_steps
      total_loss += loss
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

      if (step+1) % args.grad_accum_steps == 0:
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        # result, metric = quantify(args, scores, targets, "train")
        if exp_logger.log_interval > 0:
          # exp_logger.log_train(step, loss.item(), model)
          exp_logger.global_step += 1
          if exp_logger.global_step % exp_logger.log_interval == 0:
            log_str = 'Step {:>6d} | Loss {:5.4f} | Step: {}'.format(step, loss.item(), exp_logger.global_step)
            exp_logger.logger.info(log_str)
      
      if args.debug and step > 3*args.log_interval:
        break

    exp_logger.logger.info('Epoch {} Train Loss: {:.4f}'.format(epoch, total_loss / len(dataloader)))
    result, res_name = run_eval(args, datasets, model, exp_logger, kb_labels, split='dev')
    dev_score = result[res_name]
    exp_logger.logger.info('Step {} Eval Loss: {:.4f}'.format(exp_logger.global_step, exp_logger.eval_loss))

    if dev_score > exp_logger.best_score:
      exp_logger.logger.info('Save with the current joint accuracy {:.4f} over the previous accuracy {:.4f}...'.format(dev_score, exp_logger.best_score))
      model.save_pretrained(exp_logger.filepath) 
      exp_logger.best_score = dev_score
    exp_logger.log_dev(step+1, res_name, dev_score)

def run_eval(args, datasets, model, exp_logger, kb_labels, split='dev'):
  dataloader, num_examples = setup_dataloader(datasets, args.batch_size, split)  
  exp_logger.start_eval(num_examples, kind=args.filename)
  loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1)
  num_outputs = len(model.outputs)
  model.eval()

  preds, labels, convo_ids, turn_counts = [], [], [], []
  for batch in progress_bar(dataloader, total=len(dataloader), desc=f"Epoch {exp_logger.epoch}"):
    batch = tuple([t.to(device) for t in batch[:-1]]+[batch[-1]])
    full_history, batch_targets, context_tokens, tools = prepare_inputs(args, batch)

    with torch.no_grad():
      if args.task == 'ast':
        # batch_scores = model(full_history, context_tokens)
        # batch_loss = model.ast_loss(batch_scores, batch_targets, loss_func)
        batch_loss = model(full_history, batch_targets, context_tokens)
        batch_value = model.generate(full_history)
      elif args.task == 'cds':
        batch_scores = model(full_history, context_tokens, tools)
        batch_loss = model.cds_loss(batch_scores, batch_targets, loss_func)

    if args.cascade:
      batch_turn_count = batch_targets.pop()
      batch_convo_id = batch_targets.pop()

    if args.quantify or split=='dev':
      exp_logger.eval_loss += batch_loss.mean().item()
      exp_logger.batch_steps += 1
    
    preds.append(batch_value)
    labels.append(batch_targets)
    convo_ids.append(batch_convo_id if args.cascade else 0)
    turn_counts.append(batch_turn_count if args.cascade else 0)

    if args.debug:
      if len(turn_counts) > 10:
        break
    
  # grouped_preds = [torch.cat([pred[i] for pred in preds], dim=0) for i in range(num_outputs)]
  # grouped_labels = [torch.cat([label[i] for label in labels], dim=0) for i in range(num_outputs)]
  # ci_and_tc = (torch.cat(convo_ids, dim=0), torch.cat(turn_counts, dim=0)) if args.cascade else (0, 0)

  # utils = { 'kb_labels': kb_labels, 'ci_and_tc': ci_and_tc }
  # metrics, res_name = quantify(args, grouped_preds, grouped_labels, utils)
  metrics, res_name = ast_t5_report(preds, labels)
  exp_logger.end_eval(metrics, kind=args.filename)
  return (metrics, res_name) if split == 'dev' else metrics

if __name__ == "__main__":
  args = solicit_params()
  args = setup_gpus(args)
  set_seed(args)

  ckpt_dir, cache_results = check_directories(args)
  raw_data = load_data(args, cache_results[1])
  tokenizer, ontology, guidelines = load_tokenizer(args)
  features, mappings = process_data(args, tokenizer, ontology, guidelines, raw_data, *cache_results)
  exp_logger = ExperienceLogger(args, ckpt_dir)

  if args.task == 'ast':
    datasets = {split: ActionDataset(args, feats) for split, feats in features.items() if split not in ['action_descriptions', 'value_descriptions']}
    model = ActionStateTracking(args, mappings, ckpt_dir, tokenizer)
    if args.load_pretrain:
      filepath = os.path.join(ckpt_dir, 'pytorch_model.pt')
      model.load_state_dict(torch.load(filepath))
      print(f"Model loaded from {filepath}")
  elif args.task == 'cds':
    datasets = {split: CascadeDataset(args, feats) for split, feats in features.items()}
    model = CascadeDialogSuccess(args, mappings, ckpt_dir)

  model = model.to(device)
  model.encoder.resize_token_embeddings(len(tokenizer))
  run_main(args, datasets, model, exp_logger)
