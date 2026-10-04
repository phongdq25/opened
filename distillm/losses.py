import math
import torch
import torch.nn.functional as F

def forward_kl(logits, teacher_logits, no_model_batch):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(logits)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    prod_probs = torch.masked_fill(teacher_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    mask = (no_model_batch["label"] != -100).int()
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def reverse_kl(logits, teacher_logits, no_model_batch):
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(teacher_logits) | torch.isinf(logits)
    prod_probs = torch.masked_fill(student_probs * teacher_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    mask = (no_model_batch["label"] != -100).int()
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def symmetric_kl(logits, teacher_logits, no_model_batch, lam=0.9):
    for_kl = forward_kl(logits, teacher_logits, no_model_batch)
    rev_kl = reverse_kl(logits, teacher_logits, no_model_batch)
    distil_loss = (1-lam) * for_kl + lam * rev_kl
    return distil_loss
    
def js_distance(logits, teacher_logits, no_model_batch, lam=0.9):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = (1-lam) * teacher_probs + lam * student_probs

    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    mixed_logprobs = torch.log(mixed_probs)

    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(student_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = lam * -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

    prod_probs = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(teacher_probs * teacher_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss += (1-lam) * -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss
    
def tv_distance(logits, teacher_logits, no_model_batch):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    
    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)
    prod_probs = 0.5 * torch.masked_fill(torch.abs(teacher_probs - student_probs), inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def skewed_forward_kl(logits, teacher_logits, no_model_batch, lam=0.1):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = lam * teacher_probs + (1-lam) * student_probs
    mixed_logprobs = torch.log(mixed_probs)
    
    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def skewed_reverse_kl(logits, teacher_logits, no_model_batch, lam=0.1):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = (1-lam) * teacher_probs + lam * student_probs
    
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    mixed_logprobs = torch.log(mixed_probs)

    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(student_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def csd(logits, teacher_logits, no_model_batch, mode="SS"):
    student_probs = F.softmax(logits, dim=-1)
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    if mode == "SS":
        loss = (logits - teacher_logits - torch.sum(student_probs * (logits - teacher_logits), \
            dim=-1,keepdim=True)).detach() * student_probs.detach() * logits
    elif mode == "TS":
        loss1 = (logits - teacher_logits - torch.sum(teacher_probs * (logits - teacher_logits), \
            dim=-1,keepdim=True)).detach() * student_probs.detach() * logits
        loss2 = (logits - teacher_logits - torch.sum(student_probs * (logits - teacher_logits), \
            dim=-1,keepdim=True)).detach() * teacher_probs * logits
        loss = (loss1 + loss2) / 2
        
    x = torch.sum(loss, dim=-1).view(-1) ## summation over vocab
    mask = (no_model_batch["label"] != -100).int()
    distil_loss = torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss


def amid(logits, teacher_logits, no_model_batch, args, **kwargs):
    # AMiD: alpha-mixture assistant distribution KD (aailab-kaist/AMiD).
    # softmaxes in fp32 for bf16-training stability (as forward_kl does).
    p = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    q = F.softmax(logits, dim=-1, dtype=torch.float32)
    logp = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    logq = F.log_softmax(logits, dim=-1, dtype=torch.float32)

    alpha = args.amid_alpha
    lam = args.amid_lam
    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(teacher_logits) | torch.isinf(logits)

    # Assistant distribution r (alpha-mixture of teacher p and student q)
    if lam <= 0.0:
        r, logr = q, logq
    elif lam >= 1.0:
        r, logr = p, logp
    else:
        if alpha >= 1.0:
            logr_unnorm = lam * logp + (1.0 - lam) * logq
        else:
            t1 = math.log(lam) + 0.5 * (1.0 - alpha) * logp
            t2 = math.log(1.0 - lam) + 0.5 * (1.0 - alpha) * logq
            logr_unnorm = 2.0 / (1.0 - alpha) * torch.logaddexp(t1, t2)
        r = F.softmax(logr_unnorm, dim=-1)
        logr = F.log_softmax(logr_unnorm, dim=-1)

    div_name = args.amid_div_name
    div_order = args.amid_div_order
    if div_name == "fkl":
        if div_order == "pr":
            prod_probs = torch.masked_fill(p * (logp - logr), inf_mask, 0)
        elif div_order == "qr":
            prod_probs = torch.masked_fill(q * (logq - logr), inf_mask, 0)
        elif div_order == "rp":
            prod_probs = torch.masked_fill(r * (logr - logp), inf_mask, 0)
        elif div_order == "rq":
            prod_probs = torch.masked_fill(r * (logr - logq), inf_mask, 0)
        else:
            raise ValueError(f"amid: bad div_order {div_order}")
        x = torch.sum(prod_probs, dim=-1).view(-1)
        return torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    elif div_name == "ab":
        ab_alpha, ab_beta = 0.2, 0.7
        apb = ab_alpha + ab_beta
        if div_order == "pr":
            a1, a2 = logp, logr
        elif div_order == "qr":
            a1, a2 = logq, logr
        elif div_order == "rp":
            a1, a2 = logr, logp
        elif div_order == "rq":
            a1, a2 = logr, logq
        else:
            raise ValueError(f"amid: bad div_order {div_order}")
        term1 = torch.exp(torch.logsumexp(ab_alpha * a1 + ab_beta * a2, dim=-1))
        term2 = (ab_alpha / apb) * torch.exp(torch.logsumexp(apb * a1, dim=-1))
        term3 = (ab_beta / apb) * torch.exp(torch.logsumexp(apb * a2, dim=-1))
        divergence = -(term1 - term2 - term3) / (ab_alpha * ab_beta)
        safe = torch.where(torch.isfinite(divergence), divergence, 0.0)
        masked_sum = (safe * mask).sum()
        mask_sum = mask.sum()
        return masked_sum / mask_sum if mask_sum > 0 else masked_sum
    else:
        raise ValueError(f"amid: bad div_name {div_name}")
