import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torchmetrics.classification import Dice, JaccardIndex
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


# 设置本地模型路径
model_path = "F:/models/segformer"  # 本地保存的SegFormer模型路径

# # 从本地加载模型
# model = SegformerForSemanticSegmentation.from_pretrained(model_path)



# 2. 数据加载与预处理
class BrainTumorDataset(Dataset):
    def __init__(self, data_dir, modality_list, transform=None, num_slices=20, patient_ids=None):       #减少到20份切片
        """
        :param data_dir: 数据文件夹路径
        :param modality_list: 使用的MRI模态列表
        :param transform: 数据增强与预处理
        :param num_slices: 每个患者切的2D切片数
        :param patient_ids: 用于数据集分割的患者ID
        """
        self.data_dir = data_dir
        self.modality_list = modality_list[:3]  # 只选择前3个模态，没办法啊通道改成4个输入改不来
        self.transform = transform
        self.num_slices = num_slices
        self.patient_ids = patient_ids if patient_ids is not None else os.listdir(data_dir)
        self.patient_ids = [pid for pid in self.patient_ids if os.path.isdir(os.path.join(data_dir, pid))]  # 确保是文件夹

        self.resize_transform = transforms.Resize((512, 512))  # 调整输入图像为512*512大小
        print(f"Dataset for {data_dir} initialized with {len(self.patient_ids)} patients.")
        
    def __len__(self):
        return len(self.patient_ids) * self.num_slices  # 每个患者切成多个2D切片
    
    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx // self.num_slices]  # 获取患者ID
        slice_idx = idx % self.num_slices  # 
        print(f"Loading data for patient {patient_id} slice {slice_idx}...")

        images = []
        for modality in self.modality_list:
            modality_path = os.path.join(self.data_dir, patient_id, f"{modality}.nii.gz")
            if not os.path.exists(modality_path):
                raise FileNotFoundError(f"File {modality_path} not found.")
                
            img = nib.load(modality_path).get_fdata()
            img = preprocess_data(img)  # 对每个模态进行归一化预处理

            # 从3D图像中获取指定索引的切片
            slice_image = img[slice_idx, :, :]  # 根据slice_idx选择切片
            slice_image = Image.fromarray(slice_image)  # 将NumPy数组转换为PIL图像
            slice_image = self.resize_transform(slice_image)  # 调整图像大小
            slice_image = np.array(slice_image)  # 转换回NumPy数组
            images.append(slice_image)

        # 将所有模态图像堆叠起来
        images = np.stack(images, axis=0)

        # 标签文件是经人工纠正后的分割图
        label_path = os.path.join(self.data_dir, patient_id, "Label.nii.gz")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found for patient {patient_id}: {label_path}")
        label = nib.load(label_path).get_fdata()  # 标签：0表示背景，1表示肿瘤区域

        # 将标签中所有非背景区域（即非0的区域）设置为1，完成二分类
        label[label != 0] = 1

        # 获取当前切片的标签
        label = label[slice_idx, :, :]  # 获取标签的对应切片

        # 转换为PIL图像进行resize，再转换回tensor
        label = Image.fromarray(label)  # 将NumPy数组转换为PIL图像
        label = self.resize_transform(label)  # 调整标签大小
        label = torch.tensor(np.array(label), dtype=torch.long)  # 将PIL图像转换为NumPy数组后再转为Tensor

        # **此处添加标签维度扩展**
        label = label.unsqueeze(0)  # 将标签维度扩展为 (1, height, width)

        images = torch.tensor(images, dtype=torch.float32)

        sample = {'image': images, 'label': label}
        
        if self.transform:
            sample = self.transform(sample)
        return sample

# 3. 数据预处理函数
def preprocess_data(image_data):
    """
    数据预处理函数，将数据归一化到[0, 1]范围。
    :param image_data: 输入的MRI图像数据
    :return: 归一化后的图像数据
    """
    return (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))


# 4. 模型定义
# 加载预训练模型和特征提取器
model_name = "nvidia/segformer-b0-finetuned-ade-512-512"  # 选择适合的预训练模型
feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
model = SegformerForSemanticSegmentation.from_pretrained(
    model_name, 
    num_labels=2,  # 设置为2类（肿瘤 vs 背景）
    ignore_mismatched_sizes=True  # 忽略形状不匹配的权重
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# 5. 训练循环
def train_model(train_loader, val_loader, test_loader, model, optimizer, epochs=10):

    dice_metric = Dice()  # 使用新的 Dice 类
    
    for epoch in range(epochs):
        print(f"Starting training for epoch {epoch+1}/{epochs}")
        model.train()
        train_loss = 0
        for batch in train_loader:
            print("Processing batch...")
            inputs = batch['image'].to(device)
            labels = batch['label'].to(device)

            # 将标签维度调整为 (batch_size, height, width)，去除多余的通道维度
            labels = labels.squeeze(1)  # 去掉 (batch_size, 1, height, width) -> (batch_size, height, width)

            # 将数据输入SegFormer模型进行前向传播
            outputs = model(inputs)
            outputs_resized = F.interpolate(outputs.logits, size=(512, 512), mode='bilinear', align_corners=False)
            loss = torch.nn.CrossEntropyLoss()(outputs_resized, labels.long())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            print("Batch processed.")

        # 在验证集上进行评估
        model.eval()
        val_dice = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['image'].to(device)
                labels = batch['label'].to(device)

                # 将标签维度调整为 (batch_size, height, width)，去除多余的通道维度
                labels = labels.squeeze(1)  # 去掉 (batch_size, 1, height, width) -> (batch_size, height, width)


                outputs = model(inputs)
                preds = torch.argmax(outputs_resized, dim=1)

                dice_metric.update(preds.int(), labels.int())  # 更新metric
        val_dice = dice_metric.compute()  # 计算最终的Dice值
        dice_metric.reset()  # 重置metric对象以便下次使用
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Dice: {val_dice:.4f}")

    # 保存训练好的模型
    save_path = "C:/Users/TaLycent/Desktop/Tumor_map"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, "segformer_model.pth"))
    print("Model has learned the above configuration.:)")

    # 在测试集上进行评估
    print("Evaluating model on test set...")
    evaluate_model(test_loader, model)


# 6. 评估函数
def evaluate_model(test_loader, model):
    print("Evaluating model on validation set...")
    model.eval()
    dice_metric = Dice()  # 使用新的 Dice 类
    iou_metric = JaccardIndex(task="binary")  # 使用新的 Jaccard 类 如果用多分类：task="multiclass", num_classes=2

    total_dice = 0
    total_iou = 0
    num_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['image'].to(device)
            labels = batch['label'].to(device)

            # 将标签维度调整为 (batch_size, height, width)，去除多余的通道维度
            labels = labels.squeeze(1)  # 去掉 (batch_size, 1, height, width) -> (batch_size, height, width)


            # 将数据输入模型
            outputs = model(inputs)
            # 调整输出大小到标签的尺寸 (512x512)
            outputs_resized = F.interpolate(outputs.logits, size=(512, 512), mode='bilinear', align_corners=False)
            preds = torch.argmax(outputs_resized, dim=1)

            # 计算Dice和IoU
            # total_dice += dice_metric(preds, labels).item()
            total_dice += dice_metric(preds.int(), labels.int()).item()
            total_iou += iou_metric(preds, labels).item()
            num_samples += 1

    avg_dice = total_dice / num_samples
    avg_iou = total_iou / num_samples

    print(f"Final Model Evaluation - Average Dice: {avg_dice:.4f}, Average IoU: {avg_iou:.4f}")
    print("Validation evaluation complete.")
    return avg_dice, avg_iou


# 7. 数据加载与训练
# 数据路径和文件夹结构
data_dir = "C:/Users/TaLycent/Desktop/Tumor_map/data"
modalities = ['FLAIR', 'T1', 'T1Gd', 'T2']  # MRI模态列表

# 数据集分割
patient_ids = os.listdir(data_dir)
patient_ids = [pid for pid in patient_ids if os.path.isdir(os.path.join(data_dir, pid))]  # 获取患者ID

train_patient_ids, test_patient_ids = train_test_split(patient_ids, test_size=0.2, random_state=42)
val_patient_ids, test_patient_ids = train_test_split(test_patient_ids, test_size=0.5, random_state=42)  # 从测试集中进一步划分验证集

train_dataset = BrainTumorDataset(data_dir, modalities, patient_ids=train_patient_ids)
val_dataset = BrainTumorDataset(data_dir, modalities, patient_ids=val_patient_ids)
test_dataset = BrainTumorDataset(data_dir, modalities, patient_ids=test_patient_ids)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

# 优化器
optimizer = Adam(model.parameters(), lr=1e-4)

# 训练和评估模型
train_model(train_loader, val_loader, test_loader, model, optimizer, epochs=10)


# 8. 结果可视化
def visualize_results(model, sample):
    print("Visualizing results for a sample...")
    image = sample['image'].unsqueeze(0).to(device)  # 扩展维度以符合batch size要求
    label = sample['label'].squeeze(0).to(device)  # **移除多余的维度**
    
    # 模型预测
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        pred = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        
    # 移除多余的维度
    label = label.squeeze().cpu().numpy()  # 保证label是二维
    pred = pred[0, :, :]  # Extract 2D prediction

    # 可视化原图、标签和预测结果
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image[0, 0, :, :].cpu().numpy(), cmap='gray')
    axs[0].set_title("Original Image")
    axs[1].imshow(label, cmap='gray')  # 修正后
    axs[1].set_title("Ground Truth")
    axs[2].imshow(pred, cmap='gray')
    axs[2].set_title("Predicted")
    plt.show()
    print("Results visualization complete.")






# 测试可视化
sample = test_dataset[0]  # 获取一个样本
visualize_results(model, sample)