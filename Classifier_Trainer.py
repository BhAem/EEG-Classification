# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2022/7/4 20:40
import torch
import time


def train_on_batch(num_epochs, train_iter, valid_iter, lr, criterion, net, mlp, device, wd=0, lr_jitter=False):
    trainer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer, T_max=num_epochs * len(train_iter), eta_min=5e-6)
    for epoch in range(num_epochs):
        # training
        net.train()
        mlp.train()
        sum_loss = 0.0
        sum_acc = 0.0
        for (X, X_aug, y) in train_iter:
        # for (X, y) in train_iter:
            X = X.type(torch.FloatTensor)
            X_aug = X_aug.type(torch.FloatTensor)
            y = torch.as_tensor(y.reshape(y.shape[0]), dtype=torch.int64)
            X = X.to(device)
            # X_aug = X_aug.to(device)
            y = y.to(device)
            y_hat, _ = net(X)

            # y_hat_aug, feature_aug = net(X_aug)
            # temperature = 0.5
            # out_1, out_2 = mlp(feature), mlp(feature_aug)
            # # [2*B, D]
            # out = torch.cat([out_1, out_2], dim=0)
            # # [2*B, 2*B]
            # sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
            # mask = (torch.ones_like(sim_matrix) - torch.eye(2 * y.shape[0], device=device)).bool()
            # # [2*B, 2*B-1]
            # sim_matrix = sim_matrix.masked_select(mask).view(2 * y.shape[0], -1)
            # # 分子： *为对应位置相乘，也是点积
            # # compute loss
            # pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
            # # [2*B]
            # pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
            # loss_contra = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

            loss = criterion(y_hat, y).sum()

            total_loss = loss
            # total_loss = loss + 0.01 * loss_contra
            trainer.zero_grad()
            total_loss.backward()

            trainer.step()
            if lr_jitter:
                scheduler.step()
            sum_loss += total_loss.item() / y.shape[0]
            sum_acc += (y == y_hat.argmax(dim=-1)).float().mean()
        train_loss = sum_loss / len(train_iter)
        train_acc = sum_acc / len(train_iter)

        # test
        if epoch == num_epochs - 1:
            net.eval()
            sum_acc = 0.0
            for (X, y) in valid_iter:
                X = X.type(torch.FloatTensor)
                y = torch.as_tensor(y.reshape(y.shape[0]), dtype=torch.int64)
                X = X.to(device)
                y = y.to(device)
                y_hat, _ = net(X)
                sum_acc += (y == y_hat.argmax(dim=-1)).float().mean()
            val_acc = sum_acc / len(valid_iter)
        # if epoch % 100 == 0:
        print(f"epoch{epoch + 1}, train_loss={train_loss:.3f}, train_acc={train_acc:.3f}")
    print(
        f'training finished at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} with final_valid_acc={val_acc:.3f}')
    torch.cuda.empty_cache()
    return val_acc.cpu().data