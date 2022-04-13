import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.AverageMeter import AverageMeter
from src.PGD import PGD
import tqdm

class Trainer:
    def __init__(self, model, optimizer, criterion, pgd, K):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,milestones=[15,25],gamma = 0.4)
        self.pgd = pgd
        self.K = K

    def train_step(self, loader, epoch, grad_clip):
        loss_tracker, acc_tracker = AverageMeter(), AverageMeter()
        self.model.train()
        progress_bar = tqdm.tqdm(enumerate(loader), total=len(loader))
        for i, batch in progress_bar:
            src, trg = batch.src, batch.trg
            self.optimizer.zero_grad()
            logits, _ = self.model(src, trg[:, :-1])  # [batch_size, dest_len, vocab_size]
            loss = self.criterion(logits.contiguous().view(-1, self.model.decoder.vocab_size),
                                  trg[:, 1:].contiguous().view(-1))
            loss.backward()
            self.pgd.backup_grad()
            # 对抗训练
            for t in range(self.K):
                self.pgd.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data
                if t != self.K - 1:
                    self.model.zero_grad()
                else:
                    self.pgd.restore_grad()
                loss_adv,_ = self.model(src, trg[:, :-1])
                loss_adv = self.criterion(loss_adv.contiguous().view(-1, self.model.decoder.vocab_size),
                                      trg[:, 1:].contiguous().view(-1))
                loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            self.pgd.restore()  # 恢复embedding参数
            nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            loss_tracker.update(loss.item())
            acc_tracker.update(accuracy(logits, trg[:, 1:]))
            loss_, ppl_, acc_ = loss_tracker.average, np.exp(loss_tracker.average), acc_tracker.average
            progress_bar.set_description(
                f'Epoch: {epoch + 1:02d} -     loss: {loss_:.3f} -     ppl: {ppl_:.3f} -     acc: {acc_:.3f}%')
        return loss_tracker.average, np.exp(loss_tracker.average), acc_tracker.average

    def validate(self, loader, epoch):
        loss_tracker, acc_tracker = AverageMeter(), AverageMeter()
        self.model.eval()
        with torch.no_grad():
            progress_bar = tqdm.tqdm(enumerate(loader), total=len(loader))
            for i, batch in progress_bar:
                src, trg = batch.src, batch.trg
                logits, _ = self.model(src, trg[:, :-1])  # [batch_size, dest_len, vocab_size]
                loss = self.criterion(logits.contiguous().view(-1, self.model.decoder.vocab_size),
                                      trg[:, 1:].contiguous().view(-1))
                loss_tracker.update(loss.item())
                acc_tracker.update(accuracy(logits, trg[:, 1:]))
                loss_, ppl_, acc_ = loss_tracker.average, np.exp(loss_tracker.average), acc_tracker.average
                progress_bar.set_description(
                    f'Epoch: {epoch + 1:02d} - val_loss: {loss_:.3f} - val_ppl: {ppl_:.3f} - val_acc: {acc_:.3f}%')
        return loss_tracker.average, np.exp(loss_tracker.average), acc_tracker.average

    def train(self, train_loader, valid_loader, n_epochs, grad_clip):
        history, best_loss = {'acc': [], 'loss': [], 'ppl': [], 'val_ppl': [], 'val_acc': [], 'val_loss': []}, np.inf
        for epoch in range(n_epochs):
            loss, ppl, acc = self.train_step(train_loader, epoch, grad_clip)
            val_loss, val_ppl, val_acc = self.validate(valid_loader, epoch)
            if best_loss > val_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
            history['acc'].append(acc)
            history['val_acc'].append(val_acc)
            history['ppl'].append(ppl)
            history['val_ppl'].append(val_ppl)
            history['loss'].append(loss)
            history['val_loss'].append(val_loss)
        return history