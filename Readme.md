# ATTA paper verify
**该仓库是对ATTA论文的代码实现([论文相关链接](https://openaccess.thecvf.com/content/CVPR2021/html/Wu_Improving_the_Transferability_of_Adversarial_Samples_With_Adversarial_Transformations_CVPR_2021_paper.html))**
## Directory Structure
* `defense/` 论文中SOTA defense方法的实现代码
* `dev_data/` 该文件目录下的`val_rs`包含本次实验的攻击测试样本，样本来自于[here](https://drive.google.com/drive/folders/1CfobY6i8BfqfWPHL31FKFDipNjqWwAhS),`val_rs.csv`包含攻击测试样本所对应的标签
* `instructions/`包含验证本次实验的全部指令，该文件下的文件名字和所执行的攻击相一致，如`atta_ni_fgsm.sh`就执行的是atta_ni_fgsm攻击，执行该文件就可以生成相应的对抗样本并得到相应的实验结果，以`atta_ni_fgsm.sh`文件为例，**--output_dir**参数就是执行程序后所生成对抗样本存储的位置，你可以在对应路径下找到生成的对抗样本，当然也可以直接在`outputs`文件夹下按名字直接找到相应的对抗样本，不过最终对抗样本的存储路径最好还是以.sh文件下的 **--output_dir**路径为准,而`simple_eval.py`程序则是对生成的对抗样本进行评估，评估结果最终存储于`logs/`文件夹中，该文件夹下对应的文件名称就是相应的攻击结果，以`atta_ni_fgsm.csv`为例，该文件就记录了利用atta_ni_fgsm生成的对抗样本去迁移攻击其他模型的最终效果，文件中的数字是对抗样本使其他模型误分类的成功率，值越大说明效果越好.想要复刻本次实验的实验结果可以直接运行相应的.sh文件。
* `models/`记录了本次实验所用到模型的预训练权重，而`nets/`这包含着模型的architecture
* `outputs/`记录了实验生成的对抗样本，`logs/`记录了本次实验的最终结果.
* 最后介绍一下实验程序，`nipsr3.py,adv_FD.py,rp_defense.py`是nipsr3,FD,rp这三种sota defense方法的代码实现，`train_nni_2.py`是atta超参数搜索的相关代码，`search_space.json,config.yml`分别记录了nni超参数搜索的范围和相关的实验配置，`atta_hyperparameter.json`记录了最终实验结果最好的参数的值.`duplicate-model.py`用于修改所加载模型的variable_scope,剩下的其他代码和其名字相一致，就是相应的攻击方式代码实现，如`atta_ni_fgsm.py`就是atta_ni_fgsm攻击方法的代码实现
