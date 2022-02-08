# -*- coding: utf-8 -*-
"""
basic generation utils funcition
"""
import collections
import torch
import torch.nn as nn
import time
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def aeq(*args):
    """Assert all arguments have the same value."""
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def sequence_mask(lengths, max_len=None):
    """Create a boolean mask from sequence lengths."""
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len).type_as(lengths).repeat(batch_size, 1).lt(
        lengths.unsqueeze(1)))


def rnn_factory(rnn_type, **kwargs):
    """Rnn factory, Use pytorch version when available."""
    rnn = getattr(nn, rnn_type)(**kwargs)
    return rnn


def gelu(x):
    """Gelu Function.

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    return 0.5 * x * (1 + torch.tanh(
        math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def gumbel_softmax(logits, tau=1.0, hard=False, log_mode=True, dim=-1):
    """Gumbel softmax.

    Args:
        logits (_type_): _description_
        tau (float, optional): _description_. Defaults to 1.0.
        hard (bool, optional): _description_. Defaults to False.
        log_mode (bool, optional): _description_. Defaults to True.
        dim (int, optional): _description_. Defaults to -1.

    Returns:
        _type_: _description_
    """
    while (True):
        gumbels = -torch.empty_like(
            logits).exponential_().log()  # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
        if log_mode:
            y_soft = gumbels.log_softmax(dim)
        else:
            y_soft = gumbels.softmax(dim)
        if torch.sum(torch.isnan(y_soft)).item() < 0.01:
            break

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


def generate_sent_masks(enc_hiddens, source_lengths):
    """ Generate sentence masks for encoder hidden states.
    @param enc_hiddens (Tensor): encodings of shape (b, src_len, h), where b = batch size,
                                 src_len = max source length, h = hidden size. 
    @param source_lengths (List[int]): List of actual lengths for each of the sentences in the batch.len = batch size
    @returns enc_masks (Tensor): Tensor of sentence masks of shape (b, src_len),
                                where src_len = max source length, b = batch size.
    """
    enc_masks = torch.zeros(enc_hiddens.size(0),
                            enc_hiddens.size(1),
                            dtype=torch.float)
    for e_id, src_len in enumerate(source_lengths):
        enc_masks[e_id, :src_len] = 1
    return enc_masks


def masked_softmax(tensor, mask):
    """
    Apply a masked softmax on the last dimension of a tensor.
    The input tensor and mask should be of size (batch, *, sequence_length).
    Args:
        tensor: The tensor on which the softmax function must be applied along
            the last dimension.
        mask: A mask of the same size as the tensor with 0s in the positions of
            the values that must be masked and 1s everywhere else.
    Returns:
        A tensor of the same size as the inputs containing the result of the
        softmax.
    """
    tensor_shape = tensor.size()
    reshaped_tensor = tensor.view(-1, tensor_shape[-1])
    # Reshape the mask so it matches the size of the input tensor.
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(tensor).contiguous().float()
    reshaped_mask = mask.view(-1, mask.size()[-1])

    result = nn.functional.softmax(reshaped_tensor * reshaped_mask, dim=-1)
    result = result * reshaped_mask
    # 1e-13 is added to avoid divisions by zero.
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)
    return result.view(*tensor_shape)


def weighted_sum(tensor, weights, mask):
    """
    Apply a weighted sum on the vectors along the last dimension of 'tensor',
    and mask the vectors in the result with 'mask'.
    Args:
        tensor: A tensor of vectors on which a weighted sum must be applied.
        weights: The weights to use in the weighted sum.
        mask: A mask to apply on the result of the weighted sum.
    Returns:
        A new tensor containing the result of the weighted sum after the mask
        has been applied on it.
    """
    weighted_sum = weights.bmm(tensor)

    while mask.dim() < weighted_sum.dim():
        mask = mask.unsqueeze(1)
    mask = mask.transpose(-1, -2)
    mask = mask.expand_as(weighted_sum).contiguous().float()

    return weighted_sum * mask


def correct_predictions(output_probabilities, targets):
    """
    Compute the number of predictions that match some target classes in the
    output of a model.
    Args:
        output_probabilities: A tensor of probabilities for different output
            classes.
        targets: The indices of the actual target classes.
    Returns:
        The number of correct predictions in 'output_probabilities'.
    """
    _, out_classes = output_probabilities.max(dim=1)
    correct = (out_classes == targets).sum()
    return correct.item()


# # from datasets import load_metric
# # import pandas as pd
# # bleu_metric = load_metric("sacrebleu")
# # rouge_metric = load_metric("rouge") # rouge_score==0.0.4 work well

# 以下两个函数 不适合 做bleu 的验证


def chunks(list_of_elements, batch_size):
    """Yield successive batch-sized chunks from list_of_elements."""
    for i in range(0, len(list_of_elements), batch_size):
        yield list_of_elements[i:i + batch_size]


def evaluate_summaries_pegasus(dataset,
                               metric,
                               model,
                               tokenizer,
                               batch_size=16,
                               column_text="dialogue",
                               column_summary="summary"):
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    article_batches = list(chunks(dataset[column_text], batch_size))
    target_batches = list(chunks(dataset[column_summary], batch_size))
    #for article_batch, target_batch in tqdm( zip(article_batches, target_batches), total=len(article_batches)):
    for article_batch, target_batch in zip(article_batches, target_batches):
        inputs = tokenizer(article_batch,
                           max_length=1024,
                           truncation=True,
                           padding="max_length",
                           return_tensors="pt")
        summaries = model.generate(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            length_penalty=0.8,
            num_beams=8,
            max_length=128)
        decoded_summaries = [
            tokenizer.decode(s,
                             skip_special_tokens=True,
                             clean_up_tokenization_spaces=True)
            for s in summaries
        ]
        decoded_summaries = [d.replace("<n>", " ") for d in decoded_summaries]
        metric.add_batch(predictions=decoded_summaries,
                         references=target_batch)
    score = metric.compute()
    return score


def validate(model, dataloader):
    """
    Compute the loss and accuracy of a model on some validation dataset.
    Args:
        model: A torch module for which the loss and accuracy must be
            computed.
        dataloader: A DataLoader object to iterate over the validation data.
        criterion: A loss criterion to use for computing the loss.
        epoch: The number of the epoch for which validation is performed.
        device: The device on which the model is located.
    Returns:
        epoch_time: The total time to compute the loss and accuracy on the
            entire validation set.
        epoch_loss: The loss computed on the entire validation set.
        epoch_bleu: The bleu4 computed on the entire validation set.
    """
    # Switch to evaluate mode.
    model.eval()
    device = model.device
    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0
    all_prob = []
    all_labels = []
    # Deactivate autograd for evaluation.
    # input_ids, token_type_ids, position_ids, attention_mask, masked_positions, labels
    with torch.no_grad():
        # for (batch_seqs, batch_seq_masks, position_ids, batch_seq_segments,
        #      batch_masked_positions, batch_labels) in dataloader:
            
        for batch in dataloader:

            # print(batch)
            
            # print(batch_seqs[0])
            # print(batch_seq_masks[0])
            # print(batch_seq_segments[0])
            # print(position_ids[0])
            # print(batch_masked_positions) # 不理解
            # print(batch_labels)
            # Move input and output data to the GPU if one is used.
            seqs = batch[0].numpy()
            masks = batch[1].numpy()
            # segments = batch[3].to(device)
            labels = batch[5].numpy()

            # seqs = batch_seqs
            # masks = batch_seq_masks
            # segments = batch_seq_segments
            # labels = batch_labels

            # loss, logits, probabilities = model(seqs, masks, segments, labels)

            batch_data = {'src': torch.from_numpy(seqs),
                          'tgt': torch.from_numpy(labels),
                          'mask_src': torch.from_numpy(masks)}
            decode_output, summary, attn = model(batch_data)

            # print(decode_output)
            # print(summary)
            # print(attn)

    #         running_loss += loss.item()
    #         running_accuracy += correct_predictions(probabilities, labels)
    #         all_prob.extend(probabilities[:, 1].cpu().numpy())
    #         all_labels.extend(batch_labels)
    # epoch_time = time.time() - epoch_start
    # epoch_loss = running_loss / len(dataloader)
    # epoch_accuracy = running_accuracy / (len(dataloader.dataset))
    # # return epoch_time, epoch_loss, epoch_accuracy, roc_auc_score(all_labels, all_prob)
    # return epoch_time, epoch_loss, epoch_accuracy, 1.0


def test(model, dataloader):
    """
    Test the accuracy of a model on some labelled test dataset.
    Args:
        model: The torch module on which testing must be performed.
        dataloader: A DataLoader object to iterate over some dataset.
    Returns:
        batch_time: The average time to predict the classes of a batch.
        total_time: The total time to process the whole dataset.
        accuracy: The accuracy of the model on the input data.
    """
    # Switch the model to eval mode.
    model.eval()
    device = model.device
    time_start = time.time()
    batch_time = 0.0
    accuracy = 0.0
    all_prob = []
    all_labels = []
    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for (batch_seqs, batch_seq_masks, batch_seq_segments,
             batch_labels) in dataloader:
            batch_start = time.time()
            # Move input and output data to the GPU if one is used.
            seqs, masks, segments, labels = batch_seqs.to(
                device), batch_seq_masks.to(device), batch_seq_segments.to(
                    device), batch_labels.to(device)
            _, _, probabilities = model(seqs, masks, segments, labels)
            accuracy += correct_predictions(probabilities, labels)
            batch_time += time.time() - batch_start
            all_prob.extend(probabilities[:, 1].cpu().numpy())
            all_labels.extend(batch_labels)
    batch_time /= len(dataloader)
    total_time = time.time() - time_start
    accuracy /= (len(dataloader.dataset))
    return batch_time, total_time, accuracy, roc_auc_score(
        all_labels, all_prob)


def train(model, dataloader, optimizer, epoch_number, max_gradient_norm):
    """
    Train a model for one epoch on some input data with a given optimizer and
    criterion.
    Args:
        model: A torch module that must be trained on some input data.
        dataloader: A DataLoader object to iterate over the training data.
        optimizer: A torch optimizer to use for training on the input model.
        criterion: A loss criterion to use for training.
        epoch_number: The number of the epoch for which training is performed.
        max_gradient_norm: Max. norm for gradient norm clipping.
    Returns:
        epoch_time: The total time necessary to train the epoch.
        epoch_loss: The training loss computed for the epoch.
        epoch_accuracy: The accuracy computed for the epoch.
    """
    # Switch the model to train mode.
    model.train()
    device = model.device
    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
    correct_preds = 0
    tqdm_batch_iterator = tqdm(dataloader)
    for batch_index, (batch_seqs, batch_seq_masks, batch_seq_segments,
                      batch_labels) in enumerate(tqdm_batch_iterator):
        batch_start = time.time()
        # Move input and output data to the GPU if it is used.
        seqs, masks, segments, labels = batch_seqs.to(device), batch_seq_masks.to(device), \
            batch_seq_segments.to(device), batch_labels.to(device)
        optimizer.zero_grad()
        loss, logits, probabilities = model(seqs, masks, segments, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()
        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()
        correct_preds += correct_predictions(probabilities, labels)
        description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}"\
                      .format(batch_time_avg / (batch_index+1 ), running_loss / ( batch_index+1 ))
        tqdm_batch_iterator.set_description(description)
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_preds / len(dataloader.dataset)
    return epoch_time, epoch_loss, epoch_accuracy
