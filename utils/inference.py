import torch, numpy as np

def predict_single_label(tokenizer, model, text, max_len=256):
    enc = tokenizer(text, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
    if torch.cuda.is_available():
        enc = {k:v.to("cuda") for k,v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
    pred = int(torch.argmax(logits, dim=1).cpu().numpy()[0])
    return pred, logits.softmax(dim=1).cpu().numpy()[0]

def predict_multilabel_topk(tokenizer, model, text, label_space, label_map, top_k=3, max_len=256):
    enc = tokenizer(text, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
    if torch.cuda.is_available():
        enc = {k:v.to("cuda") for k,v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
    scores = logits.sigmoid().cpu().numpy()[0]
    top_idx = np.argsort(-scores)[:top_k]
    codes = [label_space[i] for i in top_idx]
    names = [label_map[c] for c in codes]
    return list(zip(codes, names)), scores
