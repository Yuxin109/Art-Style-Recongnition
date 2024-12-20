import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score

class Trainer(object):
    def __init__(self,
                 model=None,
                 train_loader=None,
                 testloader=None,
                 optimizer=None,
                 loss_fn=None,
                 patience=None,
                 save_path=None):

        self.model = model
        self.train_loader = train_loader
        self.testloader = testloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.patience = patience
        self.save_path = save_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_acc = 0.0
        self.early_stop_count = 0

    def train_epoch(self, epoch):

        self.model.train()
        train_loss = 0
        train_preds = []
        train_targets = []

        self.model.to(self.device)


        for data, labels in tqdm(self.train_loader):
            # print(features)
            # print(features.shape)
            # print(data)
            # print(data.shape)

            datas , labels = data.to(self.device) ,labels.to(self.device)

            outputs = self.model(datas)
            loss = self.loss_fn(outputs, labels)
            train_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())

        train_acc = accuracy_score(train_targets, train_preds)

        self.model.eval()
        test_loss = 0
        test_preds = []
        test_targets = []

        with torch.no_grad():

            for data, labels in tqdm(self.testloader):
                datas, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(datas)
                loss = self.loss_fn(outputs, labels)
                test_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                test_preds.extend(preds.cpu().numpy())
                test_targets.extend(labels.cpu().numpy())

        test_acc = accuracy_score(test_targets, test_preds)
        if test_acc > self.best_acc:
            self.best_acc = test_acc
            self.early_stop_count = 0
        else:
            self.early_stop_count += 1

        model_save_path = f"{self.save_path}/epoch_{epoch}_acc_{test_acc}"
        torch.save(self.model.state_dict(), model_save_path)

        print(f"Epoch {epoch}, "
              f"Train Loss: {train_loss / len(self.train_loader):.4f}, Train Acc: {train_acc:.4f}, "
              f"Test Loss: {test_loss / len(self.testloader):.4f}, Test Acc: {test_acc:.4f}, "
              f"Best ACC: {self.best_acc:.4f}, Early Stop Count: {self.early_stop_count}")

        if self.early_stop_count >= self.patience:
            print(f"Early Stopping at {epoch} epoch")

            return True, self.early_stop_count, self.best_acc

        return False, self.early_stop_count, self.best_acc





