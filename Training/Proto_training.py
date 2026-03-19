"""
For the revised DASMN.
yancy F. 2020/11/1
"""

import sys
import os
# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import numpy as np
import visdom
import time
import torch.nn.modules as nn
import matplotlib.pyplot as plt

from Model.encoder_model import MetricNet
from my_utils.training_utils import sample_task_tr, sample_task_te
from my_utils.init_utils import weights_init0, weights_init1, weights_init2, set_seed
from my_utils.visualize_utils import Visualize_v2
from my_utils.plot_utils import tSNE_fun
from my_utils.confusion_matrix_utils import plot_confusion_matrix, print_classification_metrics, save_confusion_matrix_data
from Data_generate.DataLoadFn import DataGenFn

device = torch.device('cuda:0')
vis = visdom.Visdom(env='yancy_env')
generator = DataGenFn()

# ====== hyper params =======
CHN = 1
DIM = 1024  # 2048
CHECK_EPOCH = 10
WEIGHT_DECAY = 0  # 1e-5

# ========== t-SNE 可视化配置 ==========
ENABLE_TSNE = True  # 主开关：是否启用t-SNE可视化
TSNE_START_EPOCH = 1  # 开始绘制t-SNE的epoch
TSNE_END_EPOCH = 60   # 结束绘制t-SNE的epoch
TSNE_SAVE_DIR = './tsne_results_proto'  # t-SNE图保存目录
# =====================================

CHECK_D = False  # check domain adaptation by t-SNE
Load = [3, 0]
CB_NUM = 8
WEIGHTS_INIT = weights_init2  # init2 ans 1 are better.

running_params = dict(train_epochs=150, test_epochs=3,
                      train_episodes=30, test_episodes=100,
                      train_split=100, test_split=0,  # generally same with train_split.
                      )
# ==========================


class ProtoLearner:
    def __init__(self, n_way, n_support, n_query):
        self.way = n_way
        self.ns = n_support
        self.nq = n_query
        self.visualization = Visualize_v2(vis)
        self.domain_criterion = nn.NLLLoss()

        self.model = MetricNet(self.way, self.ns, self.nq, vis=vis, cb_num=CB_NUM).to(device)


    def plot_adaptation(self, x_s, x_t, save_path=None, epoch=None, src_way=None, tar_way=None):
        """
        绘制源域和目标域的t-SNE可视化图（支持不同类别数）

        Args:
            x_s: 源域特征 [src_way * ns, dim]
            x_t: 目标域特征 [tar_way * nq, dim]
            save_path: 保存路径（如果为None则不保存）
            epoch: 当前epoch数（用于文件命名）
            src_way: 源域类别数（如果为None则使用self.way）
            tar_way: 目标域类别数（如果为None则使用self.way）
        """
        # 如果没有指定类别数，使用默认值
        if src_way is None:
            src_way = self.way
        if tar_way is None:
            tar_way = self.way

        x = torch.cat((x_s, x_t), dim=0)  # [src_way*ns + tar_way*nq, dim]

        # 根据源域类别数生成源域标签
        src_labels = []
        if src_way == 3:
            src_labels = ['NC-s', 'IF-s', 'OF-s']
        elif src_way == 4:
            src_labels = ['NC-s', 'IF-s', 'OF-s', 'ReF-s']
        elif src_way == 10:
            src_labels = ['NC-s', 'IF007-s', 'IF014-s', 'IF021-s', 'OF007-s',
                         'OF014-s', 'OF021-s', 'ReF007-s', 'ReF014-s', 'ReF021-s']
        else:
            src_labels = [f'Class{i}-s' for i in range(src_way)]

        # 根据目标域类别数生成目标域标签
        tar_labels = []
        if tar_way == 3:
            tar_labels = ['NC-t', 'IF-t', 'OF-t']
        elif tar_way == 4:
            tar_labels = ['NC-t', 'IF-t', 'OF-t', 'ReF-t']
        elif tar_way == 10:
            tar_labels = ['NC-t', 'IF007-t', 'IF014-t', 'IF021-t', 'OF007-t',
                         'OF014-t', 'OF021-t', 'ReF007-t', 'ReF014-t', 'ReF021-t']
        else:
            tar_labels = [f'Class{i}-t' for i in range(tar_way)]

        # 合并标签
        labels = src_labels + tar_labels

        print(f't-SNE visualization: {src_way} source classes + {tar_way} target classes')

        # 绘制t-SNE图
        fig = tSNE_fun(x.cpu().detach().numpy(), shot=self.ns,
                       name=None, labels=labels, n_dim=2)

        # 保存图片
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if epoch is not None:
                save_file = save_path.replace('.png', f'_epoch{epoch}.png')
            else:
                save_file = save_path

            plt.savefig(save_file, dpi=300, bbox_inches='tight', pad_inches=0.05)
            print(f'✓ t-SNE图已保存到: {save_file}')
            plt.close(fig)
        else:
            plt.show()

    def model_training(self, src_tasks, tgt_tasks, model_path):
        self.model.train()
        self.model.apply(WEIGHTS_INIT)  # not recommended weights initialization

        optimizer1 = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)  # lr初始值设为0.1-0.2
        c_optimizer = optimizer1  # for encoder
        c_scheduler = torch.optim.lr_scheduler.ExponentialLR(c_optimizer, gamma=0.95)  # lr=lr∗gamma^epoch

        print('source set for training:', src_tasks.shape)
        print('target set for validation', tgt_tasks.shape)
        print('(n_s, n_q)==> ', (self.ns, self.nq))

        epochs = running_params['train_epochs']
        episodes = running_params['train_episodes']
        counter = 0
        draw = False
        avg_ls = torch.zeros([episodes])
        times = np.zeros([epochs])

        # ========== 训练前的 t-SNE 可视化（初始 CNN 特征）==========
        if ENABLE_TSNE:
            print(f'\n{"="*60}')
            print('生成训练前的 t-SNE 图（初始 CNN 特征，未经对比学习优化）...')
            print(f'{"="*60}')

            # 获取源域和目标域的实际类别数
            src_way_actual = src_tasks.shape[0]
            tar_way_actual = tgt_tasks.shape[0]

            # 采样一批数据用于t-SNE可视化
            src_sample, _ = sample_task_tr(src_tasks, src_way_actual, self.ns, length=DIM)
            tgt_sample, _ = sample_task_tr(tgt_tasks, tar_way_actual, self.ns, length=DIM)

            # 提取特征（使用初始化的模型）
            self.model.eval()
            with torch.no_grad():
                src_features = self.model.get_features(src_sample.reshape([-1, CHN, DIM]), domain_label=0)
                tgt_features = self.model.get_features(tgt_sample.reshape([-1, CHN, DIM]), domain_label=1)
            self.model.train()

            # 设置保存路径
            tsne_save_path = os.path.join(TSNE_SAVE_DIR, 'tsne_training.png')

            # 绘制并保存t-SNE图（epoch 0）
            self.plot_adaptation(src_features, tgt_features,
                               save_path=tsne_save_path, epoch=0,
                               src_way=src_way_actual, tar_way=tar_way_actual)
            print(f'{"="*60}\n')
        # ==================================

        print(f'Start to train! {epochs} epochs, {episodes} episodes, {episodes * epochs} steps.\n')
        for ep in range(epochs):
            # if (ep + 1) <= 3 and CHECK_D:
            #     draw = True
            # elif 25 <= (ep + 1) <= 40 and CHECK_D:
            #     draw = True

            delta = 10 if (ep + 1) <= 30 else 5
            t0 = time.time()
            for epi in range(episodes):
                support, query = sample_task_tr(src_tasks, self.way, self.ns, length=DIM)
                tgt_v_s, tgt_v_q = sample_task_tr(tgt_tasks, self.way, self.ns, length=DIM)

                src_loss, src_acc, _, _, _ = self.model.forward(xs=support, xq=query, sne_state=False)
                draw = False
                loss = src_loss

                c_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=0.5)
                # nn.utils.clip_grad_norm_(parameters=self.d_classifier.parameters(), max_norm=0.5)
                # To clip the grads of d_classifier is Not recommended.
                c_optimizer.step()

                self.model.eval()
                with torch.no_grad():
                    tgt_loss, tgt_acc, _, _, _ = self.model.forward(xs=tgt_v_s, xq=tgt_v_q, sne_state=False)
                self.model.train()

                src_ls, src_ac = src_loss.cpu().item(), src_acc.cpu().item()
                tgt_ls, tgt_ac = tgt_loss.cpu().item(), tgt_acc.cpu().item()
                avg_ls[epi] = src_ls

                if (epi + 1) % 5 == 0:
                    self.visualization.plot([src_ls, tgt_ls], ['Source_cls', 'Target_cls'],
                                            counter=counter, scenario="proto_Cls Loss")
                    self.visualization.plot([src_ac, tgt_ac], ['C_src', 'C_tgt'],
                                            counter=counter, scenario="proto_Cls_Acc")
                    counter += 1

                # if (epi + 1) % 10 == 0:
                #     print('[epoch {}/{}, episode {}/{}] => loss: {:.6f}, acc: {:.6f}'.format(
                #         ep + 1, epochs, epi + 1, episodes, src_ls, src_ac))
            # epoch
            t1 = time.time()
            times[ep] = t1 - t0
            print('[epoch {}/{}] time: {:.6f} Total: {:.6f}'.format(ep + 1, epochs, times[ep], np.sum(times)))
            ls_ = torch.mean(avg_ls).cpu()  # .item()
            print('[epoch {}/{}] avg_loss: {:.6f}\n'.format(ep + 1, epochs, ls_))

            # ========== t-SNE 可视化 ==========
            if ENABLE_TSNE and TSNE_START_EPOCH <= (ep + 1) <= TSNE_END_EPOCH:
                print(f'\n{"="*60}')
                print(f'生成第 {ep + 1} 轮的 t-SNE 图...')
                print(f'{"="*60}')

                # 获取源域和目标域的实际类别数
                src_way_actual = src_tasks.shape[0]
                tar_way_actual = tgt_tasks.shape[0]

                # 采样一批数据用于t-SNE可视化
                src_sample, _ = sample_task_tr(src_tasks, src_way_actual, self.ns, length=DIM)
                tgt_sample, _ = sample_task_tr(tgt_tasks, tar_way_actual, self.ns, length=DIM)

                # 提取特征
                self.model.eval()
                with torch.no_grad():
                    src_features = self.model.get_features(src_sample.reshape([-1, CHN, DIM]), domain_label=0)
                    tgt_features = self.model.get_features(tgt_sample.reshape([-1, CHN, DIM]), domain_label=1)
                self.model.train()

                # 设置保存路径
                tsne_save_path = os.path.join(TSNE_SAVE_DIR, 'tsne_training.png')

                # 绘制并保存t-SNE图
                self.plot_adaptation(src_features, tgt_features,
                                   save_path=tsne_save_path, epoch=ep+1,
                                   src_way=src_way_actual, tar_way=tar_way_actual)
                print(f'{"="*60}\n')
            # ==================================

            # if isinstance(c_optimizer, torch.optim.SGD):
            c_scheduler.step()

            if ep + 1 >= CHECK_EPOCH and (ep + 1) % delta == 0:
                flag = input("Shall we stop the training? Y/N\n")
                if flag == 'y' or flag == 'Y':
                    print('Training stops!(manually)')
                    new_path = os.path.join(model_path, f"final_epoch{ep+1}")
                    self.save(new_path, running_params['train_epochs'])
                    break
                else:
                    flag = input(f"Save model at epoch {ep+1}? Y/N\n")
                    if flag == 'y' or flag == 'Y':
                        child_path = os.path.join(model_path, f"epoch{ep+1}")
                        self.save(child_path, ep+1)

        print("The total time: {:.5f} s\n".format(np.sum(times)))

    def test(self, tar_tasks, src_tasks=None, mask=False, model_eval=True):
        """
        :param mask:
        :param src_tasks: for t-sne
        :param tar_tasks: target tasks [way, n, dim]
        :return:
        """
        if model_eval:
            self.model.eval()
        else:
            self.model.train()
        print('target set', tar_tasks.shape)
        print('(n_s, n_q)==> ', (self.ns, self.nq))

        epochs = running_params['test_epochs']
        episodes = running_params['test_episodes']
        # episodes = tar_tasks.shape[1] // self.ns
        print('Start to train! {} epochs, {} episodes, {} steps.\n'.format(epochs, episodes,
                                                                           episodes * epochs))
        counter = 0
        avg_acc_all = 0.
        avg_loss_all = 0.

        print('Model.eval() is:', not self.model.training)
        for ep in range(epochs):
            avg_acc_ep = 0.
            avg_loss_ep = 0.
            sne_state = False
            for epi in range(episodes):
                tar_s, tar_q = sample_task_te(tar_tasks, self.way, self.ns, length=DIM)
                if src_tasks is not None and self.ns > 1:
                    src_s, src_q = sample_task_te(src_tasks, self.way, self.ns, length=DIM)

                # sne_state = True if epi + 1 == episodes else False
                with torch.no_grad():
                    tar_loss, tar_acc, zq_t, _, _ = self.model.forward(xs=tar_s, xq=tar_q, sne_state=sne_state)
                    if src_tasks is not None and self.ns > 1:
                        _, _, zq_s, _, _ = self.model.forward(xs=src_s, xq=src_s, sne_state=False)
                        if mask:
                            _, _, zq_t, _, _ = self.model.forward(xs=src_q, xq=src_q, sne_state=False)

                tar_ls, tar_ac = tar_loss.cpu().item(), tar_acc.cpu().item()
                avg_acc_ep += tar_ac
                avg_loss_ep += tar_ls

                self.visualization.plot([tar_ac, tar_ls], ['Acc', 'Loss'],
                                        counter=counter, scenario="proto-Test")
                counter += 1

                # if (epi + 1) == episodes:
                #     print('[{}/{}, {}/{}] loss: {:.8f}\t acc: {:.8f}'.format(
                #         # ep + 1, epochs, epi + 1, episodes, tar_ls, tar_ac))
                #     if src_tasks is not None and self.ns > 1:
                #         self.plot_adaptation(zq_s, zq_t)

            # epoch
            avg_acc_ep /= episodes
            avg_loss_ep /= episodes
            avg_acc_all += avg_acc_ep
            avg_loss_all += avg_loss_ep
            print(f'[epoch {ep + 1}/{epochs}] avg_loss: {avg_loss_ep:.8f}\tavg_acc: {avg_acc_ep:.8f}')
        avg_acc_all /= epochs
        avg_loss_all /= epochs
        print('\n------------------------Average Result----------------------------')
        print('Average Test Loss: {:.6f}'.format(avg_loss_all))
        print('Average Test Accuracy: {:.6f}\n'.format(avg_acc_all))
        vis.text(text='Eval:{} Average Accuracy: {:.6f}'.format(not self.model.training, avg_acc_all),
                 win='Eval:{} Test result'.format(not self.model.training))

    def save(self, filename, epoch):
        if os.path.exists(filename):
            filename += '(1)'
        state = {'epoch': epoch,
                 'model_state': self.model.state_dict(),
                 }
        torch.save(state, filename)
        print('This model is saved at [%s]' % filename)

    def load(self, filename):
        state = torch.load(filename)
        model_state = state['model_state']

        # Check if saved model has DPA parameters
        has_dpa_in_checkpoint = any('proto_decoder' in key for key in model_state.keys())

        # Check if current model uses DPA
        from Model.encoder_model import USE_DPA
        current_uses_dpa = USE_DPA

        # Initialize proto_decoder if needed
        if current_uses_dpa and self.model.proto_decoder is None and has_dpa_in_checkpoint:
            # Need to initialize proto_decoder before loading
            # Create a dummy forward pass to trigger initialization
            dummy_xs = torch.randn(self.way * self.ns, 1, 1024).to(device)
            dummy_xq = torch.randn(self.way * self.nq, 1, 1024).to(device)
            with torch.no_grad():
                zs = self.model.get_features(dummy_xs)
                zq = self.model.get_features(dummy_xq)
                # Initialize decoder
                if self.model.proto_decoder is None:
                    from Model.encoder_model import PrototypeDecoder, DPA_NUM_LAYERS
                    self.model.z_dim = zs.shape[-1]
                    self.model.proto_decoder = PrototypeDecoder(self.model.z_dim, num_layers=DPA_NUM_LAYERS).to(device)
            print('[Load] Initialized proto_decoder for DPA model')

        # Load with flexible matching
        if has_dpa_in_checkpoint != current_uses_dpa:
            print(f'[Load] Warning: Model architecture mismatch!')
            print(f'       Checkpoint has DPA: {has_dpa_in_checkpoint}')
            print(f'       Current model uses DPA: {current_uses_dpa}')
            print(f'       Loading with strict=False (ignoring mismatched parameters)')
            self.model.load_state_dict(model_state, strict=False)
        else:
            self.model.load_state_dict(model_state, strict=True)

        print('Load Encoder successfully from [%s]' % filename)


def train_operate(way, ns, nq, source_domain, target_domain, model_root, final_test=True, load_path=None):
    """
    训练函数，根据源域和目标域自动生成保存路径
    :param source_domain: 源域名称，如 'CQU', 'CW', 'IMS', 'OTW', 'HIT'
    :param target_domain: 目标域名称，如 'IMS', 'CW', 'CQU', 'OTW', 'HIT'
    :param model_root: 模型保存根目录
    """
    set_seed(0)

    # ============ 根据源域和目标域加载数据 ============
    print(f"\n{'='*60}")
    print(f"Source Domain: {source_domain} -> Target Domain: {target_domain}")
    print(f"{'='*60}\n")

    # 加载源域数据
    if source_domain == 'CQU':
        src, _ = generator.CQU_4way(examples=100, split=running_params['train_split'], way=4,
                                    normalize=True, label=False, data_len=DIM)
    elif source_domain == 'IMS':
        src, _ = generator.IMS_3way(examples=100, split=running_params['train_split'], way=4,
                                    normalize=True, label=False, data_len=DIM)
    elif source_domain == 'CW':
        src, _ = generator.CW_10way(way=10, order=Load[0], examples=100, split=running_params['train_split'],
                                    normalize=True, data_len=DIM, label=False)
    elif source_domain == 'OTW':
        src, _ = generator.OTW_3way(examples=100, split=running_params['train_split'], way=3,
                                    normalize=True, label=False, data_len=DIM)
    elif source_domain == 'HIT':
        src, _ = generator.HIT_3way(examples=100, split=running_params['train_split'], way=3,
                                    normalize=True, label=False, data_len=DIM, fault_type='outer')
    else:
        raise ValueError(f"Unknown source domain: {source_domain}")

    # 加载目标域数据
    if target_domain == 'IMS':
        _, tar = generator.IMS_3way(examples=100, split=0, way=4, normalize=True,
                                    label=False, data_len=DIM)
    elif target_domain == 'CQU':
        _, tar = generator.CQU_4way(examples=100, split=0, way=4, normalize=True,
                                    label=False, data_len=DIM)
    elif target_domain == 'CW':
        _, tar = generator.CW_10way(way=10, order=Load[1], examples=100, split=0,
                                    normalize=True, data_len=DIM, label=False)
    elif target_domain == 'OTW':
        _, tar = generator.OTW_3way(examples=100, split=0, way=3, normalize=True,
                                    label=False, data_len=DIM)
    elif target_domain == 'HIT':
        _, tar = generator.HIT_3way(examples=100, split=0, way=3, normalize=True,
                                    label=False, data_len=DIM, fault_type='outer')
    else:
        raise ValueError(f"Unknown target domain: {target_domain}")

    # ============ 自动对齐类别数 ============
    src_way = src.shape[0]
    tar_way = tar.shape[0]

    print(f"Source domain loaded: {src.shape} ({src_way} classes)")
    print(f"Target domain loaded: {tar.shape} ({tar_way} classes)")

    # 使用源域的类别数初始化模型
    if src_way != tar_way:
        print(f"\n[Info] Class number mismatch: {src_way} vs {tar_way}")
        print(f"       Using source domain's {src_way} classes for model initialization")
        print(f"       Target domain will use its {tar_way} classes (no forced alignment)\n")
    else:
        print(f"Class number matched: {src_way} classes\n")

    # 使用源域的类别数初始化模型
    way = src_way

    # 初始化模型
    nets = ProtoLearner(n_way=way, n_support=ns, n_query=nq)
    if load_path is not None:  # 若加载路径不为空，则默认：模型微调
        nets.load(load_path)

    # ============ 自动生成保存路径 ============
    transfer_name = f"{source_domain}2{target_domain}"
    save_path = os.path.join(model_root, transfer_name,
                             f"proto_{transfer_name.lower()}_split{running_params['train_split']}")

    # 如果路径已存在，添加时间戳
    if os.path.exists(save_path):
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_path = f"{save_path}_{timestamp}"
        print(f"⚠️  Path exists, adding timestamp: {save_path}")

    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    print(f"✓ Model will be saved to: {save_path}\n")

    # ============ 开始训练 ============
    print('Train ProtoNet!')
    nets.model_training(src, tar, save_path)

    # ============ 训练后测试 ============
    if final_test:
        print('\n' + '='*60)
        print('Final Test on Target Domain')
        print('='*60)
        nets.test(tar_tasks=tar, src_tasks=None, model_eval=True)
        nets.test(tar_tasks=tar, src_tasks=None, model_eval=False)


def test_operate(way, ns, nq, source_domain, target_domain, model_root,
                 model_name=None, eval_stats='both', ob_domain=False, num_domain=30):
    """
    测试函数，根据源域和目标域自动加载数据和模型
    :param source_domain: 源域名称，如 'CQU', 'IMS', 'CW', 'OTW', 'HIT'
    :param target_domain: 目标域名称，如 'IMS', 'CQU', 'CW', 'OTW', 'HIT'
    :param model_root: 模型保存根目录
    :param model_name: 模型文件名（如 'epoch50', 'final_epoch150'），如果为None则列出所有可用模型
    """
    set_seed(0)

    # ============ 根据源域和目标域加载测试数据 ============
    print(f"\n{'='*60}")
    print(f"Testing: {source_domain} -> {target_domain}")
    print(f"{'='*60}\n")

    # 加载源域数据（用于对比）
    if source_domain == 'CQU':
        src_tasks, _ = generator.CQU_4way(examples=100, split=running_params['train_split'], way=4,
                                          normalize=True, label=False, data_len=DIM)
    elif source_domain == 'IMS':
        src_tasks, _ = generator.IMS_3way(examples=100, split=running_params['train_split'], way=4,
                                          normalize=True, label=False, data_len=DIM)
    elif source_domain == 'CW':
        src_tasks, _ = generator.CW_10way(way=10, order=Load[0], examples=100,
                                          split=running_params['train_split'],
                                          normalize=True, data_len=DIM, label=False)
    elif source_domain == 'OTW':
        src_tasks, _ = generator.OTW_3way(examples=100, split=running_params['train_split'], way=3,
                                          normalize=True, label=False, data_len=DIM)
    elif source_domain == 'HIT':
        src_tasks, _ = generator.HIT_3way(examples=100, split=running_params['train_split'], way=3,
                                          normalize=True, label=False, data_len=DIM, fault_type='outer')
    else:
        raise ValueError(f"Unknown source domain: {source_domain}")

    # 加载目标域数据（用于测试）
    if target_domain == 'IMS':
        _, test_tasks = generator.IMS_3way(examples=100, split=0, way=4, normalize=True,
                                           label=False, data_len=DIM)
    elif target_domain == 'CQU':
        _, test_tasks = generator.CQU_4way(examples=100, split=0, way=4, normalize=True,
                                           label=False, data_len=DIM)
    elif target_domain == 'CW':
        _, test_tasks = generator.CW_10way(way=10, order=Load[1], examples=100, split=0,
                                           normalize=True, data_len=DIM, label=False)
    elif target_domain == 'OTW':
        _, test_tasks = generator.OTW_3way(examples=100, split=0, way=3, normalize=True,
                                           label=False, data_len=DIM)
    elif target_domain == 'HIT':
        _, test_tasks = generator.HIT_3way(examples=100, split=0, way=3, normalize=True,
                                           label=False, data_len=DIM, fault_type='outer')
    else:
        raise ValueError(f"Unknown target domain: {target_domain}")

    # ============ 自动对齐类别数 ============
    src_way = src_tasks.shape[0]
    tar_way = test_tasks.shape[0]

    print(f"Source domain loaded: {src_tasks.shape} ({src_way} classes)")
    print(f"Target domain loaded: {test_tasks.shape} ({tar_way} classes)")

    # 使用源域的类别数初始化模型
    if src_way != tar_way:
        print(f"\n[Info] Class number mismatch: {src_way} vs {tar_way}")
        print(f"       Using source domain's {src_way} classes for model initialization")
        print(f"       Target domain will use its {tar_way} classes (no forced alignment)\n")
    else:
        print(f"Class number matched: {src_way} classes\n")

    # 使用源域的类别数初始化模型
    way = src_way

    # 初始化模型
    model = ProtoLearner(n_way=way, n_support=ns, n_query=nq)

    # ============ 自动查找模型路径 ============
    transfer_name = f"{source_domain}2{target_domain}"
    transfer_dir = os.path.join(model_root, transfer_name)

    if not os.path.exists(transfer_dir):
        print(f"❌ Error: Transfer directory not found: {transfer_dir}")
        print(f"Please train the model first using source={source_domain}, target={target_domain}")
        return

    # 如果没有指定模型名，列出所有可用模型
    if model_name is None:
        print(f"Available models in {transfer_dir}:\n")
        all_models = []
        for item in os.listdir(transfer_dir):
            item_path = os.path.join(transfer_dir, item)
            if os.path.isdir(item_path):
                # 列出这个训练目录下的所有模型文件
                for model_file in os.listdir(item_path):
                    if 'epoch' in model_file:
                        all_models.append(os.path.join(item, model_file))

        if not all_models:
            print("❌ No trained models found!")
            return

        for i, model_path in enumerate(all_models, 1):
            print(f"  {i}. {model_path}")

        choice = input("\nEnter model number to test (or full path): ").strip()
        try:
            model_name = all_models[int(choice) - 1]
        except (ValueError, IndexError):
            model_name = choice

    # 构建完整的模型路径
    if not os.path.isabs(model_name):  # 如果不是绝对路径
        load_path = os.path.join(transfer_dir, model_name)
    else:
        load_path = model_name

    if not os.path.exists(load_path):
        print(f"❌ Error: Model file not found: {load_path}")
        return

    print(f"✓ Loading model from: {load_path}\n")

    # ============ 执行测试 ============
    src_tasks = src_tasks if ob_domain else None
    running_params['test_episodes'] = 10 if ob_domain else 100

    print('test_task shape:', test_tasks.shape)
    if ob_domain and src_tasks is not None:
        print('src_task shape:', src_tasks.shape)

    # 根据eval_stats参数决定测试模式
    if eval_stats == 'yes':
        model.load(load_path)
        model.test(test_tasks, src_tasks, model_eval=True)

    elif eval_stats == 'both':
        model.load(load_path)
        print(f"\n{'='*60}")
        print("Testing with Model.eval() = True")
        print('='*60)
        model.test(test_tasks, src_tasks, model_eval=True)

        print('\n' + '='*60)
        print("Reloading model for Model.eval() = False")
        print('='*60)
        model.load(load_path)
        model.test(test_tasks, src_tasks, model_eval=False)

    elif eval_stats == 'no':
        model.load(load_path)
        model.test(test_tasks, src_tasks, model_eval=False)
    else:
        print(f"❌ Unknown eval_stats: {eval_stats}. Use 'yes', 'no', or 'both'.")


if __name__ == "__main__":
    import os

    # ============ 配置参数 ============
    n_cls = 4  # 类别数会自动根据源域调整
    ns = nq = 5

    # ============ 设置源域和目标域 ============
    SOURCE_DOMAIN = 'HIT'   # 'CQU', 'IMS', 'CW', 'OTW', 'HIT'
    TARGET_DOMAIN = 'IMS'  # 'IMS', 'CQU', 'CW', 'OTW', 'HIT'

    # ============ 设置模型保存根目录 ============
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_ROOT = os.path.join(project_root, "Modelsave", "ProtoNet")

    # 创建根目录
    if not os.path.exists(MODEL_ROOT):
        os.makedirs(MODEL_ROOT, exist_ok=True)
        print(f"✓ Created model root directory: {MODEL_ROOT}")

    print("\n" + "="*60)
    print("ProtoNet Training and Testing")
    print("="*60)
    print(f"Source Domain: {SOURCE_DOMAIN}")
    print(f"Target Domain: {TARGET_DOMAIN}")
    print(f"Model Root: {MODEL_ROOT}")
    print(f"Number of Classes: {n_cls}")
    print(f"Support/Query Samples: {ns}/{nq}")
    print("="*60 + "\n")

    # ============ 训练模式 ============
    flag = input('Train? (y/n): ').strip().lower()
    if flag in ['y', 'yes']:
        train_operate(
            way=n_cls,
            ns=ns,
            nq=nq,
            source_domain=SOURCE_DOMAIN,
            target_domain=TARGET_DOMAIN,
            model_root=MODEL_ROOT,
            final_test=True
        )

    # ============ 测试模式 ============
    flag = input('\nTest? (y/n): ').strip().lower()
    if flag in ['y', 'yes']:
        model_name = input('Model name (leave empty to list all): ').strip()
        model_name = model_name if model_name else None

        test_operate(
            way=n_cls,
            ns=ns,
            nq=nq,
            source_domain=SOURCE_DOMAIN,
            target_domain=TARGET_DOMAIN,
            model_root=MODEL_ROOT,
            model_name=model_name,
            eval_stats='both',
            ob_domain=False
        )
