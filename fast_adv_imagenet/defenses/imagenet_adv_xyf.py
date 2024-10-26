import argparse
import tqdm
from progressbar import *
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

BASE_DIR = os.path.dirname(os.path.abspath("../"))
sys.path.append(BASE_DIR)
from sklearn.model_selection import train_test_split
from fast_adv_imagenet.attacks import DDN

"""

"""

parser = argparse.ArgumentParser(description='ImageNet Training data augmentation')

parser.add_argument('--input_dir', default=r'D:\12045\adv-SR\PLP\fast_adv_imagenet\data\imagenet_train_10000\train', help='path to dataset')
parser.add_argument('--img_size', default=224, type=int, help='size of image')
parser.add_argument('--workers', default=2, type=int, help='number of data loading workers')
parser.add_argument('--cpu', action='store_true', help='force training on cpu')
parser.add_argument('--save-folder', '--sf', default='weights/imagenet_adv_train/',
                    help='folder to save state dicts')
parser.add_argument('--visdom_env', '--ve', type=str, default="imagenet100_wrn_baseline_at")
parser.add_argument('--report_msg', '--rm', type=str, default="imagenet100_wrn_baseline， 新的baseline at实验")
parser.add_argument('--save-freq', '--sfr', default=10, type=int, help='save frequency')
parser.add_argument('--save-name', '--sn', default='ImageNet', help='name for saving the final state dict')

parser.add_argument('--batch-size', '-b', default=32, type=int, help='mini-batch size')
parser.add_argument('--epochs', '-e', default=140, type=int, help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float, help='initial learning rate')
parser.add_argument('--lr_decay', '--lrd', default=0.9, type=float, help='decay for learning rate')
parser.add_argument('--lr_step', '--lrs', default=2, type=int, help='step size for learning rate decay')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float, help='weight decay')
parser.add_argument('--drop', default=0.3, type=float, help='dropout rate of the classifier')

parser.add_argument('--adv', type=int, default=0, help='epoch to start training with adversarial images')
parser.add_argument('--max-norm', type=float, default=1, help='max norm for the adversarial perturbations')
parser.add_argument('--steps', default=10, type=int, help='number of steps for the attack')

parser.add_argument('--visdom_available', '--va', type=bool, default=True)
parser.add_argument('--visdom-port', '--vp', type=int, default=8097,
                    help='For visualization, which port visdom is running.')
parser.add_argument('--print-freq', '--pf', default=10, type=int, help='print frequency')

parser.add_argument('--num-attentions', '--na', default=32, type=int, help='number of attention maps')
parser.add_argument('--backbone-net', '--bn', default='wide_resnet', help='feature extractor')
parser.add_argument('--beta', '--b', default=5e-2, help='param for update feature centers')
parser.add_argument('--amp', '--amp', default=True)

args = parser.parse_args()
# 参数设置
input_dir = r'D:\12045\adv-SR\PLP\fast_adv_imagenet\data\imagenet_train_10000\train'
batch_size = 32
epochs = 140
learning_rate = 0.001
save_folder = 'weights/'
# csv_file = 'dev.csv'

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据集定义
class ImageSet(Dataset):
    def __init__(self, df, transformer):
        self.df = df
        self.transformer = transformer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        image_path = self.df.iloc[item]['image_path']
        image = Image.open(image_path).convert('RGB')
        image = self.transformer(image)
        label_idx = self.df.iloc[item]['label_idx']
        return image, label_idx

# 数据加载
def load_data(csv, input_dir, img_size=args.img_size, batch_size=args.batch_size):
    jir = pd.read_csv(csv)
    all_imgs = [os.path.join(input_dir, str(i)) for i in jir['ImageId'].tolist()]
    all_labels = jir['TrueLabel'].tolist()

    train_df = pd.DataFrame({'image_path': all_imgs, 'label_idx': all_labels})
    train_data, val_data = train_test_split(train_df, stratify=train_df['label_idx'].values, train_size=0.8)

    transformer_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transformer_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageSet(train_data, transformer_train)
    val_dataset = ImageSet(val_data, transformer_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

train_loader, val_loader = load_data(os.path.join(args.input_dir, '../dev.csv'), input_dir)

# train_loader, val_loader = load_data(os.path.join(args.input_dir, 'dev.csv'), input_dir, transformer_train, transformer_val)

# 模型定义
model = models.resnet50(pretrained=True).to(device)
model.train()

# 优化器和学习率调度器
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# 对抗攻击实例
attacker = DDN(steps=10, device=device)
best_val_acc = 0.0
# 训练循环
for epoch in range(epochs):
    model.train()
    for images, labels in tqdm.tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        # 原始图像的损失
        logits = model(images)
        loss = nn.CrossEntropyLoss()(logits, labels)

        # 对抗训练
        if epoch >= 5:  # 假设从第 6 个 epoch 开始使用对抗训练
            # print(f"Image min before attack: {images.min()}, Image max before attack: {images.max()}")
            # adv_images = attacker.attack(model, images, labels)
            # logits_adv = model(adv_images)
            # loss_adv = nn.CrossEntropyLoss()(logits_adv, labels)
            # loss = loss + loss_adv

            # Denormalize images to bring values back to the [0, 1] range
            mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(1, 3, 1, 1)
            images_denormalized = images * std + mean  # Denormalize

            # Ensure the denormalized images are clamped to [0, 1] range
            images_denormalized = images_denormalized.clamp(0, 1)

            # Perform the attack on denormalized images
            adv_images = attacker.attack(model, images_denormalized, labels)

            # Re-apply normalization to adversarial images for model input
            adv_images = (adv_images - mean) / std  # Renormalize back

            logits_adv = model(adv_images)
            loss_adv = nn.CrossEntropyLoss()(logits_adv, labels)
            loss = loss + loss_adv

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()

    # 验证
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        val_acc = correct / total
        print(f'Epoch {epoch}, Val Accuracy: {val_acc}')
        # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_save_path = os.path.join(save_folder, 'resnet50_best.pth')
        torch.save(model.state_dict(), best_save_path)
        print(f'Best model saved with accuracy: {best_val_acc:.4f}')

        # 保存每5轮的模型
    if epoch % 5 == 0 or epoch == epochs - 1:  # Save on every 5th epoch and the last epoch
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            save_path = os.path.join(save_folder, f'resnet50_epoch_{epoch}.pth')
            torch.save(model.state_dict(), save_path)
    # # 保存模型
    # if epoch == epochs - 1:  # 只在最后一个 epoch 保存模型
    #     if not os.path.exists(save_folder):
    #         os.makedirs(save_folder)
    #     save_path = os.path.join(save_folder, 'resnet50_final.pth')
    #     torch.save(model.state_dict(), save_path)
    #