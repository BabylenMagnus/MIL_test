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


def train_classification(
        epochs, model, classification_model, train_loader, test_loader, loss_func, optimizer, noti_rate=80,
        transform=None
):
    model.eval()
    train_losses = []
    test_accuracies = []
    train_accuracies = []

    for epoch in range(epochs):
        classification_model.train()

        total_loss = 0
        total_accuracy = 0

        for batch_idx, (data, classes) in enumerate(train_loader, start=1):

            data = data.cuda()
            data = data.to(torch.float32)
            data /= 255
            input_data = data

            classes = classes.cuda()

            if transform:
                input_data = transform(input_data)

            output = classification_model(model(input_data))
            loss = loss_func(output, classes)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_accuracy += sum(output.argmax(1) == classes).item()

            if not batch_idx % noti_rate:
                print(
                    f'Train batch: {batch_idx} train loss: {round(total_loss / (batch_idx + 1), 4)}, '
                    f'train accuracy: {round(total_accuracy / (batch_idx + 1), 4)}'
                )

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        avg_acc = total_accuracy / len(train_loader)
        train_accuracies.append(avg_acc)

        classification_model.eval()
        test_accuracy = 0

        for data, classes in test_loader:
            with torch.no_grad():
                data = data.cuda()
                data = data.to(torch.float32)
                data /= 255
                classes = classes.cuda()

                output = classification_model(model(data))

                test_accuracy += sum(output.argmax(1) == classes).item()

        test_accuracy /= len(test_loader)
        test_accuracies.append(test_accuracy)

        print(
            f'Train Epoch: {epoch} train loss: {round(avg_loss, 4)}, train accuracy: {round(avg_acc, 4)} '
            f' test accuracy: {round(test_accuracy, 4)}'
        )

    return train_losses, train_accuracies, test_accuracies


def train_mnist_ae(epochs, model, train_loader, test_loader, loss_func, optimizer, noti_rate=5):
    train_losses = []
    test_losses = []

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0

        for data, _ in train_loader:

            data = data.cuda()
            data = data.to(torch.float32)
            data /= 255
            input_data = data

            output = model(input_data)

            loss = loss_func(output, data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

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

        if not epoch % noti_rate:
            print(
                f'Train Epoch: {epoch} train loss: {round(avg_loss, 4)} test loss: {round(test_loss, 4)}'
            )

    return train_losses, test_losses


def train_mnist_classification(
        epochs, model, classification_model, train_loader, test_loader, loss_func, optimizer, noti_rate=5
):
    train_losses = []
    test_accuracies = []
    train_accuracies = []

    for epoch in range(1, epochs + 1):
        classification_model.train()

        total_loss = 0
        total_accuracy = 0

        for batch_idx, (data, classes) in enumerate(train_loader, start=1):

            data = data.cuda()
            data = data.to(torch.float32)
            data /= 255
            input_data = data

            classes = classes.cuda().long()

            output = classification_model(model(input_data))
            loss = loss_func(output, classes)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_accuracy += sum(output.argmax(1) == classes).item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        avg_acc = total_accuracy / len(train_loader)
        train_accuracies.append(avg_acc)

        classification_model.eval()
        test_accuracy = 0

        for data, classes in test_loader:
            with torch.no_grad():
                data = data.cuda()
                data = data.to(torch.float32)
                data /= 255
                classes = classes.cuda()

                output = classification_model(model(data))

                test_accuracy += sum(output.argmax(1) == classes).item()

        test_accuracy /= len(test_loader)
        test_accuracies.append(test_accuracy)

        if not epoch % noti_rate:
            print(
                f'Train Epoch: {epoch} train loss: {round(avg_loss, 4)}, train accuracy: {round(avg_acc, 4)} '
                f' test accuracy: {round(test_accuracy, 4)}'
            )

    return train_losses, train_accuracies, test_accuracies
