# import Dataset
import baselineUtils_single as baselineUtils
import torch
import individual_TF_single as individual_TF
import torch.nn.functional as F
import os
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

device=torch.device("cuda")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_dataset,me,st=baselineUtils.create_pie_dataset('train',device)
test_dataset,_,_=baselineUtils.create_pie_dataset('test',device,mean=me,std=st)

test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)



# def set_lr(lr,optim):
#     print("setting learning rate to: {}".format(lr))
#     for param_group in self.optimizer.param_groups:
#         param_group["lr"] =
# print(len(tr_dl))
def draw_image(data1,data2,data3,w1,w2,w3,label,idx,batch_idx,sub_area=None):

    # x = np.arange(701)
    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # axs[0].plot(x, data1)
    # axs[1].plot(x, data2)
    # axs[2].plot(x, data3)
    # axs[0].set_title('figure1_'+str(w1))
    # axs[1].set_title('figure2_'+str(w2))
    # axs[2].set_title('figure3_'+str(w3))
    # plt.tight_layout()
    # if not os.path.exists(os.path.join(root_dir_show,str(label),str(sub_area))):
    #     os.makedirs(os.path.join(root_dir_show,str(label),str(sub_area)))
    #
    # plt.savefig(os.path.join(root_dir_show,str(label),str(sub_area),str(batch_idx)+str('_')+str(idx)+'.png'))
    if not os.path.exists(os.path.join(root_dir_npy,str(label),str(sub_area))):
        os.makedirs(os.path.join(root_dir_npy,str(label),str(sub_area)))
    np.save(os.path.join(root_dir_npy,str(label),str(sub_area),str(batch_idx)+str('_')+str(idx)+'.npy'), np.array([[data1,data2,data3],[w1,w2,w3]]))
    return None
def draw_input_map(weight_att,ori_data,label,batch_idx):
    for idx,(weight,ori_input,label) in enumerate(zip(weight_att,ori_data,label)):
        data1=ori_input[0]
        data2=ori_input[1]
        data3=ori_input[2]
        weight1=weight[0]
        weight2=weight[1]
        weight3 = weight[2]
        # print(weight1,weight2,weight3)
        # exit(0)
        if label==1:
            print(weight1,weight2,weight3)
            if weight1>5 or weight2>5 or weight3>5:
                # print('show_att_0627')
                if weight1>5:
                    draw_image(data1, data2, data3, weight1,weight2,weight3,label,idx,batch_idx,sub_area=1)
                if weight2>5:
                    draw_image(data1, data2, data3, weight1,weight2,weight3,label,idx,batch_idx,sub_area=2)
                if weight3>5:
                    draw_image(data1, data2, data3, weight1,weight2,weight3,label,idx,batch_idx,sub_area=3)
        # else:
        #     if weight1>0.9 or weight2>0.9 or weight3>0.9:
        #         # print('show_att_0627')
        #         if weight1>0.9:
        #             draw_image(data1, data2, data3, weight1,weight2,weight3,label,idx,batch_idx,sub_area=1)
        #         if weight2>0.9:
        #             draw_image(data1, data2, data3, weight1,weight2,weight3,label,idx,batch_idx,sub_area=2)
        #         if weight3>0.9:
        #             draw_image(data1, data2, data3, weight1,weight2,weight3,label,idx,batch_idx,sub_area=3)

        # if label==0:
        #
        # else:
        #
        # print(label)
        # exit()

    return None
def draw_att_map(plot_list,epoch,idx):
    data1 = np.array(np.concatenate((plot_list[0],plot_list[1]),axis=1))
    data2 = np.array(np.concatenate((plot_list[2],plot_list[3]),axis=1))
    data3 = np.array(np.concatenate((plot_list[4],plot_list[5]),axis=1))

    # 创建一个包含3个子图的图形
    fig, axs = plt.subplots(3, 1, figsize=(10, 6))

    # 隐藏坐标轴
    for ax in axs:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(True)

    # 在每个子图上绘制热力图并设置纵坐标的刻度标签
    sns.heatmap(data=data1, cmap='Reds', ax=axs[0], cbar=False, yticklabels=['cortex', 'tha', 'hippo'])
    sns.heatmap(data=data2, cmap='Reds', ax=axs[1], cbar=False, yticklabels=['cortex', 'tha', 'hippo'])
    sns.heatmap(data=data3, cmap='Reds', ax=axs[2], cbar=False, yticklabels=['cortex', 'tha', 'hippo'])
    # 保存图像
    plt.savefig(os.path.join(root_dir,str(epoch),str(idx)+'.png'))
    return 0

    # 显示图像
    # plt.savefig('heatmap1.png')
def cal_acc(result,out):
    result=list(result.numpy())
    label=list(out.numpy())
    correct=0
    bs=0
    pos_correct=0
    pos_all_correct=0
    for a,b in zip(result,label):
        if b==1:
            if a==1:
                pos_correct+=1
                pos_all_correct += 1
            else:
                pos_all_correct+=1

        if a==b:
            correct+=1
            bs+=1
        else:
            bs+=1
    return correct,bs,pos_correct,pos_all_correct
# correct,bs=cal_acc(result,out)
input_list=[]
label_list=[]
#0.0005 mose2
#0.0001 mos
print('start')
for recur in range(1):
    xx = True
    train_dataset,me,st=baselineUtils.create_pie_dataset('train',device)
    tr_dl = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    learning_rate=0.0005
    decay_rate=20
    epoch=0
    max_epoch=15
    stepsize=10
    loss_fn = torch.nn.MSELoss()
    model = individual_TF.IndividualTF().to(device)
    optim = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=0.0001,
                    weight_decay=1e-4
                )
    weight_list=[]

    while epoch<max_epoch:
            model.train()
            train_loss=0
            epoch += 1
            # print('start training')
            # l=[]
            for batch in tqdm(tr_dl):
                optim.zero_grad()
                # print(id_b, batch['input'].shape, batch['label'].shape)
                input=batch['input'].to(device)
                input = input.squeeze(2).squeeze(1)
                label = batch['label'][:, 0]
                if int(input.shape[0])==1:
                    continue
                out=model(input,pred=False)
                loss = F.cross_entropy(out, label)
                train_loss+=loss
                loss.backward()
                optim.step()
                if epoch % stepsize == 0:
                    learning_rate /= decay_rate
                    for param_group in optim.param_groups:
                        param_group["lr"] = learning_rate
            plot_list=[]
            # weight_list=[]
            if epoch%3==0:
                total_num=0
                correct_count=0
                model.eval()
                val_loss=0
                pos_correct_correct=0
                pos_all_correct=0
                idx=0
                for id_b, batch in enumerate(test_dl):
                    input = batch['input'].to(device)
                    input = input.squeeze(2).squeeze(1)
                    out = model(input)
                    # out=F.softmax(out, dim=1)

                    label = batch['label'][:, 0]
                    # print(label.shape,out.detach().cpu().shape)
                    result = torch.argmax(out.detach().cpu(), dim=1)
                    # print(result)
                    # print(label)
                    # exit()
                    correct,bs,pos_correct,pos_all=cal_acc(result,label.detach().cpu())
                    pos_all_correct+=pos_all
                    pos_correct_correct+=pos_correct
                    correct_count+=correct
                    total_num+=bs
                    # loss = F.cross_entropy(out, label)
                    # val_loss+=loss
                print('acc',correct_count/total_num,pos_correct_correct/pos_all_correct)
                # val_loss/=len(test_dl)
                model.train()

            # ======= 第一步：计算背景数据（训练集均值） ========
model.eval()
all_train_inputs = []
allputs=[]
all_labels = []
for batch in tqdm(tr_dl, desc="Collecting training data"):
    inputs = batch['input'].to(device)  # shape: (batch_size, 1, 1, 701)
    # print(inputs.shape)
    inputs = inputs.view(inputs.size(0), -1)  # (batch_size, 701)
    # print(inputs.shape)
    # exit()
    all_train_inputs.append(inputs.detach().cpu().numpy())
    allputs.append(inputs)
    label = batch['label'][:, 0].cpu().numpy()
    all_labels.append(label)
all_train_inputs = np.concatenate(all_train_inputs, axis=0)  # shape: (total_train_samples, 701)


background = torch.tensor(all_train_inputs.mean(axis=0), dtype=torch.float32).to(device)  # shape: (701,)
background = background.unsqueeze(0)#.numpy()
print(background.shape)
import torch
import matplotlib.pyplot as plt

# 生成一个示例 1×701 的 tensor
# vector = torch.rand(1, 701)  # 请替换为你的实际 tensor

# 转换为 NumPy 数组
vector_np = background.squeeze().cpu().numpy()

# 画图
import pandas as pd
acs_ori = pd.read_csv("../AD_2023_processed.csv", encoding='latin-1')
rows = [row for row in acs_ori]
x_label = rows[9:710]  # 701个波长标签 (string)
xticks_interval = 50
xticks_positions = np.arange(0, 701, xticks_interval)
xticks_labels = [x_label[i] for i in xticks_positions]

plt.figure(figsize=(12, 4))
plt.plot(vector_np, color='purple', linewidth=2,label="Vector Values")
plt.xticks(xticks_positions, xticks_labels, rotation=45, ha='right')
plt.tight_layout()
# plt.xlabel("Index")
# plt.ylabel("Value")
# plt.title("1×701 Tensor Visualization")
plt.savefig(f"wave.png", dpi=300)
plt.close()
# exit()

all_test_inputs = []

for batch in tqdm(test_dl, desc="Collecting test data"):
    inputs = batch['input'].to(device)  # shape: (batch_size, 1, 1, 701)
    inputs = inputs.view(inputs.size(0), -1)  # (batch_size, 701)
    all_test_inputs.append(inputs)
    allputs.append(inputs)
    label = batch['label'][:, 0].cpu().numpy()
    all_labels.append(label)
all_test_inputs = torch.cat(allputs, dim=0)  # shape: (num_test_samples, 701)
# all_test_inputs=
print(all_test_inputs.shape)
all_labels = np.concatenate(all_labels).tolist()
print(len(all_labels))
all_labels = np.array(all_labels)
# print(all_labels)
# exit()
indices_class0 = np.where(all_labels == 0)[0]
indices_class1 = np.where(all_labels == 1)[0]

##############################################
# test_samples = all_test_inputs.numpy()  # Convert to numpy array (N,701)

# Partition features into groups (every 16 features)
# segment_size = 16  # You mentioned wanting 16 points per group
# num_features = 701
# partitions = np.zeros(num_features, dtype=int)
# for idx in range(num_features):
#     partitions[idx] = idx // segment_size
# print(partitions)
# print(partitions.shape)
# exit()

# # Create a masked prediction function
# def model_mask(input_masked):
#     print(input_masked.shape)
#     input_tensor = torch.tensor(input_masked, dtype=torch.float32).to(device)
#
#     input_tensor = input_tensor#.unsqueeze(1).unsqueeze(2)  # (batch_size,1,1,701)
#     with torch.no_grad():
#         logits = model(input_tensor)
#         print(logits.shape)
#         probs = F.softmax(logits, dim=1).cpu().numpy()
#         print(probs.shape)
#     return probs
#
# # Create the masker
# masker = shap.maskers.Partition(background, clustering=partitions)
#
# # Create the explainer with background data and use the masker
# explainer = shap.KernelExplainer(model_mask, background, masker=masker)
#
# # 计算SHAP值，推荐设置较小的nsamples (50~100)
# all_test_inputs_numpy = all_test_inputs.cpu().numpy()
# shap_values = explainer.shap_values(all_test_inputs_numpy)
# print(shap_values[0].shape,shap_values[1].shape)
# exit()
# # 分析正类(label=1) SHAP值
# shap_segment_values_class1 = np.abs(shap_values[1]).mean(axis=0)  # shape=(35,)


##############################################

# shap_values[0]: 对类别0的贡献
# shap_values[1]: 对类别1的贡献

# 仅对标签为0的样本，提取SHAP值 (类别0的贡献)


# exit()
# exit()
# def baseline_als(y, lam=10000, p=0.0001, niter=10):
#     L = len(y)
#     D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
#     D = lam * D.dot(D.transpose()) # Precompute this term since it does not depend on `w`
#     w = np.ones(L)
#     W = sparse.spdiags(w, 0, L, L)
#     for i in range(niter):
#         W.setdiag(w) # Do not create a new matrix, just update diagonal values
#         Z = W + D
#         z = spsolve(Z, w*y)
#         w = p * (y > z) + (1-p) * (y < z)
#     return z
# x=y-z

# ======= 第二步：定义一个适配SHAP的模型函数 =======
def model_forward(x):
    x = x  # 还原为 (batch_size, 1, 1, 701)
    with torch.no_grad():
            logits = model(x)
            # print(logits.shape)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            # print(probs.shape)
    return probs


# 创建 DeepExplainer
explainer = shap.DeepExplainer(model, background)

# 计算所有测试样本的SHAP值
shap_values = explainer.shap_values(all_test_inputs)
# 兼容新/旧版 SHAP：旧版返回 list[ndarray]，新版返回 ndarray(shape=(N, F, C))
if isinstance(shap_values, list):
    shap_class0 = shap_values[0][indices_class0]
    shap_class1 = shap_values[1][indices_class1]
else:
    shap_class0 = shap_values[indices_class0, :, 0]
    shap_class1 = shap_values[indices_class1, :, 1]
print(shap_class0.shape,type(shap_class0),shap_class1.shape,type(shap_class1))

# exit()
sub_dir='mose2_cortex'
os.makedirs(sub_dir,exist_ok=True)
os.makedirs(os.path.join(sub_dir,'test'),exist_ok=True)

# shap_values是一个列表（分类数目个元素），二分类任务则 shap_values[0].shape=(num_test_samples, 701)

# ======= 第二步：对SHAP值进行分析（波谱区间的重要性） ========
# 按每段10个点合并，形成70段波谱

# 分别计算每个特征点的平均绝对SHAP值
import pandas as pd
acs_ori = pd.read_csv("../AD_2023_processed.csv", encoding='latin-1')
rows = [row for row in acs_ori]
x_label = rows[9:710]  # 701个波长标签 (string)

# SHAP值 (已计算好)
mean_abs_shap_class0 = np.abs(shap_class0).mean(axis=0)
mean_abs_shap_class1 = np.abs(shap_class1).mean(axis=0)

# 选择每隔50个标签显示一次，避免拥挤
xticks_interval = 50
xticks_positions = np.arange(0, 701, xticks_interval)
xticks_labels = [x_label[i] for i in xticks_positions]

# 类别0 SHAP 值绘图
plt.figure(figsize=(16, 5))
plt.plot(range(701), mean_abs_shap_class0, color='skyblue', label='Class 0 SHAP')
plt.xticks(xticks_positions, xticks_labels, rotation=45, ha='right')
plt.xlabel("Wavelength")
plt.ylabel("Average SHAP Value (Class=0)")
plt.title("Spectral Importance for Class 0")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f"{sub_dir}/test/shap_class0_701points.png", dpi=300)
plt.close()

# 类别1 SHAP 值绘图
plt.figure(figsize=(16, 5))
plt.plot(mean_abs_shap_class1, color='salmon')
plt.xticks(xticks_positions, xticks_labels, rotation=45)
plt.xlabel("Wavelength")
plt.ylabel("Average SHAP Value (Class=1)")
plt.title("Spectral Importance for Class 1")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f"{sub_dir}/test/shap_class1_701points.png", dpi=300)
plt.close()

# 类别0最重要的20个波长点（SHAP值最大）
top20_idx_class0 = np.argsort(mean_abs_shap_class0)[-20:][::-1]
top20_labels_class0 = [x_label[i] for i in top20_idx_class0]
top20_values_class0 = mean_abs_shap_class0[top20_idx_class0]

print("\n类别0中SHAP值最大的20个波长点:")
for idx, (label, value) in enumerate(zip(top20_labels_class0, top20_values_class0), 1):
    print(f"{idx+1}. 波长点: {label}, SHAP值: {value:.6f}")

# 类别1
top20_idx_class1 = np.argsort(mean_abs_shap_class1)[-20:][::-1]
top20_labels_class1 = [x_label[i] for i in top20_idx_class1]
top20_values_class1 = mean_abs_shap_class1[top20_idx_class1]

print("\n类别1 最重要的20个波谱点:")
for idx, (label, value) in enumerate(zip(top20_labels_class1, top20_values_class1), 1):
    print(f"{idx+1:2d}. 波长: {label}, SHAP值: {value:.6f}")


with torch.no_grad():
    orig_logits = model(all_test_inputs.to(device))  # (N,2)
    # orig_probs = F.softmax(logits, dim=1).cpu().numpy()  # (N,2)
prob_diff = np.zeros((701, all_test_inputs.shape[0], 2))
print(orig_logits.shape)
for feature_idx in range(701):
    # 克隆原始输入
    masked_input = all_test_inputs.clone()
    # masked_input_numpy = masked_input#.numpy()

    # 用背景数据替换当前维度
    masked_value = background[0,feature_idx]
    # print(masked_input_numpy[:, feature_idx].shape,background[0,feature_idx])
    # exit()
    masked_input[:, feature_idx] = masked_value
    # print(masked_input[:, feature_idx])


    # 转回tensor
    # masked_input_tensor = torch.tensor(masked_input, dtype=torch.float32).to(device)
    # masked_input_tensor = masked_input_tensor

    # 计算mask后的概率
    with torch.no_grad():
        logits_masked = model(masked_input)
        # probs_masked = F.softmax(logits, dim=1).cpu().numpy()


    # 记录概率差异（原始概率- mask后的概率）
    prob_diff[feature_idx] = orig_logits.cpu().numpy() - logits_masked.cpu().numpy()
    # print(prob_diff[feature_idx])
    # exit()
prob_diff0 = prob_diff[:,indices_class0,0]

# 仅对标签为1的样本，提取SHAP值 (类别1的贡献)
prob_diff1 = prob_diff[:,indices_class1,1]
print(prob_diff0,prob_diff1)
print(prob_diff0.shape,prob_diff1.shape)
mean_abs_prob_diff0 = np.abs(prob_diff0).mean(axis=1)  # (701,)
mean_abs_prob_diff1 = np.abs(prob_diff1).mean(axis=1)  # (701,)

# X轴标签间隔，避免密集
xticks_interval = 50
xticks_positions = np.arange(0, 701, xticks_interval)
xticks_labels = [rows[9:710][i] for i in xticks_positions]

# 类别0绘图
plt.figure(figsize=(16, 5))
plt.plot(mean_abs_prob_diff0, color='skyblue')
plt.xticks(xticks_positions, xticks_labels, rotation=45)
plt.xlabel("Wavelength")
plt.ylabel("Average Probability Change (Class=0)")
plt.title("Spectral Importance (probability change) for Class 0")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f"{sub_dir}/test/prob_class0_701points.png", dpi=300)
plt.close()

# 类别1
plt.figure(figsize=(16, 5))
plt.plot(mean_abs_prob_diff1, color='salmon')
plt.xticks(xticks_positions, xticks_labels, rotation=45)
plt.xlabel("Wavelength")
plt.ylabel("Average Probability Change (Class=1)")
plt.title("Spectral Importance (probability change) for Class 1")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f"{sub_dir}/test/prob_class1_701points.png", dpi=300)
plt.close()

print(mean_abs_prob_diff1.shape,mean_abs_shap_class1.shape,type(mean_abs_shap_class1),type(mean_abs_prob_diff1))
mean2=mean_abs_prob_diff1.tolist()
mean1=mean_abs_shap_class1.tolist()
import json
with open(f"{sub_dir}/test/shap.json",'w') as file:
    json.dump(mean1,file,indent=4)
with open(f"{sub_dir}/test/prob.json",'w') as file:
    json.dump(mean2,file,indent=4 )


# 打印影响力最大的20个波长点对应的标签

# 类别0
top20_indices_class0 = np.argsort(mean_abs_prob_diff0)[-20:][::-1]
top20_labels_class0 = [x_label[i] for i in top20_indices_class0]
top20_values_class0 = mean_abs_prob_diff0[top20_indices_class0]

print("Top 20 important wavelengths for Class 0:")
for idx, val in zip(top20_labels_class0, top20_values_class0):
    print(f"Label: {idx}, Avg Prob Change: {val:.6f}")

# 类别1
top20_indices_class1 = np.argsort(mean_abs_prob_diff1)[-20:][::-1]
top20_labels_class1 = [x_label[i] for i in top20_indices_class1]
top20_values_class1 = mean_abs_prob_diff1[top20_indices_class1]

print("\nClass 1 Most Important 20 wavelengths:")
for label, val in zip(top20_labels_class1, top20_values_class1):
    print(f"Wavelength: {label}, Probability Change: {val:.6f}")
