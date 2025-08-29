"""
Tiny distillation demo: Transformer-like teacher → xLSTM student.
Uses synthetic data and a toy teacher so it runs fast on CPU/MPS.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ukm.hrm.xlstm import StabilizedXLSTM
from ukm.training.distill_hrm import DistillTrainer, KDSchedule


class ToyTeacher(nn.Module):
    def __init__(self, vocab_size=256, d_model=64, nhead=4, nlayers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, nlayers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        x = self.emb(input_ids)
        h = self.enc(x)
        logits = self.lm_head(h)
        return {'logits': logits, 'hidden': h}


class XLSTMStudent(nn.Module):
    def __init__(self, vocab_size=256, d_model=64, nlayers=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, d_model)
        self.core = StabilizedXLSTM(d_model, num_layers=nlayers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        x = self.emb(input_ids)
        h = self.core(x)
        logits = self.lm_head(h)
        return {'logits': logits, 'hidden': h}


def make_batch(vocab_size=256, B=4, L=32, device='cpu'):
    input_ids = torch.randint(0, vocab_size, (B, L), device=device)
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    return input_ids, labels


def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    vocab_size = 256
    d_model = 64
    teacher = ToyTeacher(vocab_size, d_model).to(device).eval()
    student = XLSTMStudent(vocab_size, d_model).to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=2e-4)

    kd = DistillTrainer(
        student=student,
        vocab_size=vocab_size,
        teacher_hidden_dim=d_model,
        student_hidden_dim=d_model,
        schedule=KDSchedule(),
        beta=0.1,
    ).to(device)

    for step in range(50):
        student.train(); opt.zero_grad()
        input_ids, labels = make_batch(vocab_size, device=device)
        with torch.no_grad():
            t_out = teacher(input_ids)
        metrics = kd.train_step(
            input_ids,
            labels,
            t_out['logits'].detach(),
            teacher_hidden=t_out['hidden'].detach(),
        )
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        if step % 10 == 0:
            print(f"step {step:03d} loss={metrics['loss']:.4f} CE={metrics['ce']:.4f} KL={metrics['kl']:.4f} F={metrics['frob']:.6f} α={metrics['alpha']:.2f} T={metrics['T']:.2f}")


if __name__ == '__main__':
    main()

