import torch
import wandb
import yaml
import argparse
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str)

parser_args = parser.parse_args()

with open(parser_args.config, 'r') as f:
    args = yaml.safe_load(f)
    args = argparse.Namespace(**args)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)

train_transforms = [transforms.Resize((300, 300))]

if args.use_augmentation:
    train_transforms.extend([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    ])

train_transforms.extend([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_train = transforms.Compose(train_transforms)

transform_test = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dtd_trn = datasets.DTD(root=args.dataroot, split='train', download=True, transform=transform_train)
dtd_tst = datasets.DTD(root=args.dataroot, split='val', download=True, transform=transform_test)

dtd_train = DataLoader(dtd_trn, batch_size=args.batch_size, num_workers=0, shuffle=True)
dtd_test = DataLoader(dtd_tst, batch_size=args.batch_size, num_workers=0, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Residual block
class ResBlk(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch_out)
        if ch_out == ch_in:
            self.downsample = nn.Sequential()
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if out.shape != identity.shape:
            identity = self.downsample(identity)

        out += identity
        out = F.relu(out)

        return out


# ResNet18
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = nn.Sequential(
            ResBlk(64, 64, stride=1),
            ResBlk(64, 64, stride=1)
        )
        self.layer2 = nn.Sequential(
            ResBlk(64, 128, stride=2),
            ResBlk(128, 128, stride=1)
        )
        self.layer3 = nn.Sequential(
            ResBlk(128, 256, stride=2),
            ResBlk(256, 256, stride=1)
        )
        self.layer4 = nn.Sequential(
            ResBlk(256, 512, stride=2),
            ResBlk(512, 512, stride=1)
        )
        self.fc = nn.Linear(512, 47)  # 47 classes in DTD dataset

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, [1, 1])
        x = x.view(x.size(0), -1)  # [b, 512, 1, 1] => [b, 512]
        x = self.fc(x)
        return x


def load_pretrained_model(model, pretrained_path):
    state_dict = torch.load(pretrained_path)
    state_dict.pop('fc.weight')  # Remove the last layer
    state_dict.pop('fc.bias')
    model.load_state_dict(state_dict, strict=False)  # Set strict=False to load only matching layers


def load_checkpoint(model, checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))


def freeze_backbone(model: ResNet18):
    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True


def unfreeze_backbone(model: ResNet18):
    for param in model.parameters():
        param.requires_grad = True


def save_checkpoint(model, epoch):
    torch.save(model.state_dict(), f'best_model_{epoch + 1}.pth')


def train(model, total_epoch):
    loss_fun = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    scheduler = None
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_scheduler_step_size,
                                              gamma=args.lr_scheduler_gamma)

    best_val_acc = 0.0
    for epoch in range(1, total_epoch + 1):

        # check if we need to unfreeze backbone
        if args.use_pretrained and args.freeze_backbone and epoch == args.freeze_until:
            unfreeze_backbone(model)

        model.train()
        loss_sum = 0.0
        total_correct = 0
        total_num = 0
        for inputs, labels in tqdm(dtd_train, desc=f"[Train Epoch {epoch}]"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fun(outputs, labels)
            loss.backward()

            if args.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)  # gradient clipping

            optimizer.step()

            pred = outputs.argmax(dim=1)
            total_correct += torch.eq(pred, labels).sum().item()
            total_num += inputs.size(0)

            loss_sum += loss.item()

        if args.use_scheduler:
            scheduler.step()

        acc = total_correct / total_num
        print(f"Epoch {epoch} Loss: {loss_sum:.5f}, train accuracy: {acc * 100:.4f}%")
        wandb.log({"Loss": loss_sum}, step=epoch)
        wandb.log({"Train Accuracy": 100 * acc}, step=epoch)

        val_acc = test(model, epoch)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_model_{epoch}.pth')


@torch.no_grad()
def test(model, epoch):
    model.eval()
    total_correct = 0
    total_num = 0

    for inputs, labels in tqdm(dtd_test, desc=f"[Eval Epoch {epoch}]"):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        pred = outputs.argmax(dim=1)
        total_correct += torch.eq(pred, labels).sum().item()
        total_num += inputs.size(0)

    acc = total_correct / total_num
    print(f"Accuracy of the network on the test dataset: {acc * 100:.4f}%")
    wandb.log({"Val Accuracy": 100 * acc}, step=epoch)
    return acc


if __name__ == '__main__':
    model = ResNet18()

    if args.use_pretrained:
        load_pretrained_model(model, args.pretrained_path)
        if args.freeze_backbone:
            freeze_backbone(model)

    model.to(device)

    wandb.init(project="ResNet18_DTD_full_experiment", config=args, name=args.name)
    wandb.watch(model, log_freq=1)
    train(model, args.epochs)
