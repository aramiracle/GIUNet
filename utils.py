import torch

def test(model, loader, criterion):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for data in loader:
            output = model(data)
            loss = criterion(output, data.y)

            total_loss += loss.item()
            total_correct += (output.argmax(dim=1) == data.y).sum().item()
            total_samples += data.y.size(0)

    avg_loss = total_loss / len(loader)
    avg_acc = total_correct / total_samples

    return avg_loss, avg_acc