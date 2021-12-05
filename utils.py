import torch


def train_ae(epochs, model, train_loader, test_loader, loss_func, optimizer, noti_rate=80, transform=None):
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        model.train()

        total_loss = 0

        for batch_idx, (data, _) in enumerate(train_loader, start=1):

            data = data.cuda()
            data = data.to(torch.float32)
            data /= 255
            input_data = data

            if transform:
                input_data = transform(input_data)

            output = model(input_data)

            loss = loss_func(output, data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if not batch_idx % noti_rate:
                print(
                    f'Train batch: {batch_idx} train loss: {round(total_loss / (batch_idx + 1), 4)}'
                )

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        model.eval()
        test_loss = 0

        for data, _ in test_loader:
            with torch.no_grad():
                data = data.cuda()
                data = data.to(torch.float32)
                data /= 255
                output = model(data)

                test_loss += loss_func(output, data).item()

        test_loss /= len(test_loader)
        test_losses.append(test_loss)

        print(
            f'Train Epoch: {epoch} train loss: {round(avg_loss, 4)} test loss: {round(test_loss, 4)}'
        )

    return train_losses, test_losses
