import logging
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import csv
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report
from tqdm import tqdm
from args import args
from preprocess import read_dataWithIndices, tokenize_dataWithIndices, read_ground_truth_files
from torch.utils.data import SequentialSampler, DataLoader
from PIL import ImageFile, Image
from transformers import get_linear_schedule_with_warmup, BertForSequenceClassification, BertTokenizer

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

stats_columns = '{0:>5}|{1:>5}|{2:>5}|{3:>5}|{4:>5}|{5:>5}|{6:>5}|{7:>5}|{8:>5}|{9:>5}|{10:>5}'
stats_template = 'Epoch {epoch_idx}\n' \
                 '{mode} Accuracy: {acc}\n' \
                 '{mode} F1: {f1}\n' \
                 '{mode} ave. F1: {f1_ave}\n' \
                 '{mode} Recall: {recall}\n' \
                 '{mode} ave. Recall: {recall_ave}\n' \
                 '{mode} Precision: {prec}\n' \
                 '{mode} ave. Precision: {prec_ave}\n' \
                 '{mode} Loss: {loss}\n'
logger = logging.getLogger('bertresnet.py')


def train_avgbertresnet(num_epochs, model_resnet, model_bert, train_iter, dev_iter, train_iter_bert, dev_iter_bert,
                        optimizer_resnet, optimizer_bert, loss_fn, scheduler, device, checkpoint_dir_bert,
                        checkpoint_dir_resnet, results_file):
    device = torch.device(device)
    best_dev_acc = 0
    best_eval_loss = np.inf

    n_total_steps = len(train_iter)
    total_iter = len(train_iter) * num_epochs

    for epoch in range(num_epochs):
        model_resnet.to(device)
        model_resnet.train()
        model_bert.to(device)
        model_bert.train()
        truths = []
        predictions = []
        trues = []
        preds = []
        # data_path = os.path.join(args.data_dir, "absconc_data_raw.csv")
        # ground_truth = read_ground_truth_files(args.data_dir)

        logger.info('Training epoch: {}'.format(epoch))
        train_loss = 0
        train_loss_bert = 0

        for (inputs, labels, paths), batch_ids in tqdm(zip(train_iter, train_iter_bert)):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # prepare bert
            # pathlist = []
            # for path in paths:
            #	pathlist.append(path.split('/')[-1].split('.')[0])
            # data, labels_bert = read_dataWithIndices(data_path, ground_truth, args.feature, pathlist)
            # train_data = tokenize_dataWithIndices(data, labels_bert)
            # train_iter = DataLoader(train_data, sampler=SequentialSampler(train_data), batch_size=args.batch_size)

            # run bert
            scheduler_bert = get_linear_schedule_with_warmup(optimizer_bert, num_warmup_steps=0,
                                                             num_training_steps=total_iter)
            input_ids = batch_ids[0].to(device)
            att_masks = batch_ids[1].to(device)
            labels_bert = batch_ids[2].to(device)
            model_bert.zero_grad()
            _, logits_bert = model_bert(input_ids, token_type_ids=None, attention_mask=att_masks, labels=labels_bert)
            # record preds, trues
            _pred = logits_bert.cpu().data.numpy()
            _label = labels.cpu().data.numpy()
            preds.extend(_pred)
            trues.extend(_label)

            # run resnet
            outputs = model_resnet(inputs)

            # compute average
            ave = (outputs + logits_bert) / 2
            _, pred = torch.max(ave, axis=1)
            predictions.extend(pred)
            truths.extend(labels)
            loss = loss_fn(ave, labels)
            train_loss += loss.item()

            # backpropagate and update optimizer learning rate
            loss.backward()
            nn.utils.clip_grad_norm_(model_bert.parameters(), 1.0)
            optimizer_bert.step()
            scheduler_bert.step()

            # backpropagate and update optimizer learning rate
            optimizer_resnet.step()
            optimizer_resnet.zero_grad()
        scheduler.step()

        predictions = torch.as_tensor(predictions).cpu()
        truths = torch.as_tensor(truths).cpu()
        train_loss = train_loss / n_total_steps

        train_acc, train_f1, train_recall, train_prec, train_f1_ave, train_recall_ave, train_prec_ave = calculate_metrics(
            truths, predictions, average=None)
        print(stats_template.format(mode='train', epoch_idx=epoch, acc=train_acc, f1=train_f1, f1_ave=train_f1_ave,
                                    recall=train_recall,
                                    recall_ave=train_recall_ave, prec=train_prec, prec_ave=train_prec_ave,
                                    loss=train_loss))

        # validation
        acc, f1, recall, prec, f1_ave, recall_ave, prec_ave, valid_loss = eval_avgbertresnet(dev_iter, dev_iter_bert,
                                                                                             model_resnet, model_bert,
                                                                                             loss_fn, device)

        # write results to csv
        with open(results_file, 'a') as output_file:
            cw = csv.writer(output_file, delimiter='\t')
            cw.writerow(["train-" + str(epoch),
                         '%0.4f' % train_acc,
                         '%0.4f' % train_prec_ave,
                         '%0.4f' % train_recall_ave,
                         '%0.4f' % train_f1_ave,
                         '%0.4f' % train_f1[0],
                         '%0.4f' % train_f1[1],
                         '%0.4f' % train_prec[0],
                         '%0.4f' % train_prec[1],
                         '%0.4f' % train_recall[0],
                         '%0.4f' % train_recall[1]])
            cw.writerow(["valid-" + str(epoch),
                         '%0.4f' % acc,
                         '%0.4f' % prec_ave,
                         '%0.4f' % recall_ave,
                         '%0.4f' % f1_ave,
                         '%0.4f' % f1[0],
                         '%0.4f' % f1[1],
                         '%0.4f' % prec[0],
                         '%0.4f' % prec[1],
                         '%0.4f' % recall[0],
                         '%0.4f' % recall[1]])

        # save_model(model, checkpoint_dir)
        if best_dev_acc < f1_ave:
            logging.debug('New dev f1 {dev_acc} is larger than best dev f1 {best_dev_acc}'.format(dev_acc=f1,
                                                                                                  best_dev_acc=best_dev_acc))
            best_dev_acc = f1_ave
            best_eval_loss = valid_loss
            save_model(model_resnet, model_bert, checkpoint_dir_resnet, checkpoint_dir_bert)


def eval_avgbertresnet(dev_iter, dev_iter_bert, model_resnet, model_bert, loss_fn, device):
    device = torch.device(device)
    n_total_steps = len(dev_iter)
    model_bert.to(device)
    model_resnet.to(device)
    model_bert.eval()
    model_resnet.eval()
    dev_loss = 0
    predictions = []
    truths = []
    trues = []
    preds = []

    # forward pass
    with torch.no_grad():
        for (inputs, labels, paths), batch_ids in tqdm(zip(dev_iter, dev_iter_bert)):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # run bert
            input_ids = batch_ids[0].to(device)
            att_masks = batch_ids[1].to(device)
            labels_bert = batch_ids[2].to(device)
            _, logits_bert = model_bert(input_ids, token_type_ids=None, attention_mask=att_masks, labels=labels_bert)
            # run resnet
            outputs = model_resnet(inputs)

            # compute average
            ave = (outputs + logits_bert) / 2
            _, pred = torch.max(ave, axis=1)
            predictions.extend(pred)
            truths.extend(labels)
            loss = loss_fn(ave, labels)
            dev_loss += loss.item()

    predictions = torch.as_tensor(predictions).cpu()
    truths = torch.as_tensor(truths).cpu()

    dev_loss = dev_loss / n_total_steps
    acc, f1, recall, prec, f1_ave, recall_ave, prec_ave = calculate_metrics(truths, predictions, average=None)
    print(stats_template
          .format(mode='valid', epoch_idx='__', acc=acc, f1=f1, f1_ave=f1_ave, recall=recall,
                  recall_ave=recall_ave, prec=prec, prec_ave=prec_ave, loss=dev_loss))
    return acc, f1, recall, prec, f1_ave, recall_ave, prec_ave, dev_loss


def test_avgbertresnet(test_iter, test_iter_bert, model_resnet, model_bert, loss_fn, device):
    device = torch.device(device)
    n_total_steps = len(test_iter)
    model_bert.to(device)
    model_resnet.to(device)
    model_bert.eval()
    model_resnet.eval()
    test_loss = 0
    predictions = []
    truths = []
    trues = []
    preds = []

    # forward pass
    with torch.no_grad():
        for (inputs, labels, paths), batch_ids in tqdm(zip(test_iter, test_iter_bert)):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # run bert
            input_ids = batch_ids[0].to(device)
            att_masks = batch_ids[1].to(device)
            labels_bert = batch_ids[2].to(device)
            _, logits_bert = model_bert(input_ids, token_type_ids=None, attention_mask=att_masks, labels=labels_bert)
            # run resnet
            outputs = model_resnet(inputs)

            # compute average
            ave = (outputs + logits_bert) / 2
            _, pred = torch.max(ave, axis=1)
            predictions.extend(pred)
            truths.extend(labels)
            loss = loss_fn(ave, labels)
            test_loss += loss.item()

    predictions = torch.as_tensor(predictions).cpu()
    truths = torch.as_tensor(truths).cpu()

    test_loss = test_loss / n_total_steps
    acc, f1, recall, prec, f1_ave, recall_ave, prec_ave = calculate_metrics(truths, predictions, average=None)
    print(stats_template
          .format(mode='valid', epoch_idx='__', acc=acc, f1=f1, f1_ave=f1_ave, recall=recall,
                  recall_ave=recall_ave, prec=prec, prec_ave=prec_ave, loss=test_loss))
    return acc, f1, recall, prec, f1_ave, recall_ave, prec_ave


def calculate_metrics(label, pred, average='binary'):
    logging.debug('Expected: \n{}'.format(label[:20]))
    logging.debug('Predicted: \n{}'.format(pred[:20]))

    acc = round(accuracy_score(label, pred), 4)
    f1 = [round(score, 4) for score in f1_score(label, pred, average=average)]
    recall = [round(score, 4) for score in recall_score(label, pred, average=average)]
    prec = [round(score, 4) for score in precision_score(label, pred, average=average)]

    f1_ave = f1_score(label, pred, average='weighted')
    recall_ave = recall_score(label, pred, average='weighted')
    prec_ave = precision_score(label, pred, average='weighted')

    return acc, f1, recall, prec, f1_ave, recall_ave, prec_ave


def save_model(model_resnet, model_bert, checkpoint_dir_resnet, checkpoint_dir_bert):
    checkpoint_dir_resnet = os.path.join(checkpoint_dir_resnet,
                                         '{model_name}_epochs_{epoch}_lr_{lr}.bin'.format(model_name='resnet',
                                                                                          epoch=args.epochs,
                                                                                          lr=args.lr))
    torch.save(model_resnet.state_dict(), checkpoint_dir_resnet)
    model_bert.save_pretrained(checkpoint_dir_bert)
    return


def load_model(model_resnet, model_bert, checkpoint_dir_resnet, checkpoint_dir_bert):
    checkpoint_dir_resnet = os.path.join(checkpoint_dir_resnet,
                                         '{model_name}_epochs_{epoch}_lr_{lr}.bin'.format(model_name='resnet',
                                                                                          epoch=args.epochs,
                                                                                          lr=args.lr))
    model_resnet.load_state_dict(torch.load(checkpoint_dir_resnet))
    model_bert = BertForSequenceClassification.from_pretrained(checkpoint_dir_bert, num_labels=args.num_label,
                                                               output_attentions=False, output_hidden_states=False)
    return model_resnet, model_bert
