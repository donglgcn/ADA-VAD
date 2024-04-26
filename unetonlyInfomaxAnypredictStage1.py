import gc
import os
import torch
import torch.nn as nn
import numpy as np
import yaml
import shutil
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from tqdm import tqdm
from argparse import ArgumentParser

from losses.loss import Gradient_Loss, Intensity_Loss, aggregate_kl_loss
from datasets.dataset import Chunked_sample_dataset_new, img_batch_tensor2numpy

from models.mem_cvae import HFVAD
from models.unet import MLP_Projection, GlobalDiscriminator, SelfCompleteNet1raw1ofAnyPredictStage1

from utils.initialization_utils import weights_init_kaiming
from utils.vis_utils import visualize_sequences
from utils.model_utils import loader, saver, only_model_saver
from evaluation_Anypredictstage1 import evaluate


def train(config, training_chunked_samples_dir, testing_chunked_samples_files):
    paths = dict(log_dir="%s/%s" % (config["log_root"], config["exp_name"]),
                 ckpt_dir="%s/%s" % (config["ckpt_root"], config["exp_name"]))
    os.makedirs(paths["ckpt_dir"], exist_ok=True)

    batch_size = config["batchsize"]
    epochs = config["num_epochs"]
    num_workers = config["num_workers"]
    device = config["device"]
    lr = config["lr"]
    training_chunk_samples_files = sorted(os.listdir(training_chunked_samples_dir))

    # grad_loss = Gradient_Loss(config["alpha"],
    #                           config["model_paras"]["img_channels"] * config["model_paras"]["clip_pred"],
    #                           device).to(device)
    # intensity_loss = Intensity_Loss(l_num=config["intensity_loss_norm"]).to(device)
    mse_loss = nn.MSELoss()

    # model = HFVAD(num_hist=config["model_paras"]["clip_hist"],
    #               num_pred=config["model_paras"]["clip_pred"],
    #               config=config,
    #               features_root=config["model_paras"]["feature_root"],
    #               num_slots=config["model_paras"]["num_slots"],
    #               shrink_thres=config["model_paras"]["shrink_thres"],
    #               mem_usage=config["model_paras"]["mem_usage"],
    #               skip_ops=config["model_paras"]["skip_ops"],
    #               finetune=config["model_paras"]["finetune"]).to(device)

    model = SelfCompleteNet1raw1ofAnyPredictStage1(**config["model_paras"]).to(device)
    mi_model = GlobalDiscriminator().to(device)
    of_mi_model = GlobalDiscriminator().to(device)
    paramlist = list(model.parameters())+list(mi_model.parameters()) + list(of_mi_model.parameters())

    optimizer = optim.Adam(paramlist, lr=lr, eps=1e-7, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)

    step = 0
    epoch_last = 0
    if not config["pretrained"]:
        model.apply(weights_init_kaiming)
    else:
        assert (config["pretrained"] is not None)
        model_state_dict = torch.load(config["pretrained"])["model_state_dict"]
        model.load_state_dict(model_state_dict)

    writer = SummaryWriter(paths["log_dir"])
    # copy hyper-params settings
    shutil.copyfile(config["config_file"],
                    os.path.join(config["log_root"], config["exp_name"], "cfg.yaml"))

    source_dataset_name = config["dataset_name"]
    # target_dataset_name = config["crossdomain_dataset_name"]
    target_dataset_names = config["crossdomain_dataset_name"] if type(config["crossdomain_dataset_name"]) is list else [
        config["crossdomain_dataset_name"]]
    best_auc = {"auc": -1, "w_r": -1, "w_p": -1}
    best_auc_source = {"auc": -1, "w_r": -1, "w_p": -1}
    best_auc_target = {target_dataset_name: {"auc": -1, "w_r": -1, "w_p": -1} for target_dataset_name in
                target_dataset_names}
    w_rs = config.get("w_rs", [config["w_r"]])
    w_ps = config.get("w_ps", [config["w_p"]])
    for epoch in range(epoch_last, epochs + epoch_last):
        for chunk_file_idx, chunk_file in enumerate(training_chunk_samples_files):
            dataset = Chunked_sample_dataset_new(os.path.join(training_chunked_samples_dir, chunk_file))
            dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
            for idx, train_data in tqdm(enumerate(dataloader),
                                        desc="Training Epoch %d, Chunked File %d" % (epoch + 1, chunk_file_idx),
                                        total=len(dataloader)):
                model.train()

                sample_frames, sample_ofs, _, _, _ = train_data
                sample_ofs = sample_ofs.to(device)
                sample_frames = sample_frames.to(device)

                of_output, raw_output, of_target, raw_target, raw_emb, of_emb, negative_raw_emb, negative_of_emb\
                = model(sample_frames, sample_ofs)

                # modal invariance loss SIMCLR
                raw_emb = torch.flatten(raw_emb, start_dim=1)
                of_emb = torch.flatten(of_emb, start_dim=1)
                negative_raw_emb = torch.flatten(negative_raw_emb, start_dim=1)
                negative_of_emb = torch.flatten(negative_of_emb, start_dim=1)

                # shuffle of_emb to make positive samples todo can use moco version comparison
                raw_emb_prime = torch.cat((raw_emb[1:], raw_emb[0].unsqueeze(0)), dim=0).detach()
                of_emb_prime = torch.cat((of_emb[1:], of_emb[0].unsqueeze(0)), dim=0).detach()
                Ej = -F.softplus(-mi_model(raw_emb, raw_emb_prime)).mean()
                Em = F.softplus(mi_model(raw_emb, negative_raw_emb)).mean()

                of_Ej = -F.softplus(-of_mi_model(of_emb, of_emb_prime)).mean()
                of_Em = F.softplus(of_mi_model(of_emb, negative_of_emb)).mean()
                loss_invariance = Em - Ej + of_Em - of_Ej  # mi lowerbound. deep infomax global, https://github.com/DuaneNielsen/DeepInfomaxPytorch/blob/master/train.py

                # loss recon = loss_of + loss_frame
                loss_recon_of = mse_loss(of_output, of_target)
                loss_recon_frame = mse_loss(raw_output, raw_target)
                loss_recon = loss_recon_of + loss_recon_frame

                # total loss
                loss_total = loss_recon + config["loss_invariance_weight"] * loss_invariance


                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

                if step % config["logevery"] == config["logevery"] - 1:
                    print("[Step: {}/ Epoch: {}]: Loss: {:.4f} ".format(step + 1, epoch + 1, loss_recon))

                    writer.add_scalar('loss_total/train', loss_total, global_step=step + 1)
                    writer.add_scalar('loss_recon/train', loss_recon, global_step=step + 1)
                    writer.add_scalar('loss_frame/train', loss_recon_frame, global_step=step + 1)
                    writer.add_scalar('loss_flow_recon/train', loss_recon_of, global_step=step + 1)
                    writer.add_scalar('loss_invariance/train', loss_invariance, global_step=step + 1)

                    num_vis = 6
                    writer.add_figure("img/train_sample_frames",
                                      visualize_sequences(img_batch_tensor2numpy(
                                          sample_frames.cpu()[:num_vis, :, :, :]),
                                          seq_len=sample_frames.size(1) // 3,
                                          return_fig=True),
                                      global_step=step + 1)
                    writer.add_figure("img/train_frame_recon",
                                      visualize_sequences(img_batch_tensor2numpy(
                                          raw_output.detach().cpu()[:num_vis, :, :, :]),
                                          seq_len=config["clip_pred"],
                                          return_fig=True),
                                      global_step=step + 1)
                    # memAE输入的光流和重建的光流
                    writer.add_figure("img/train_of_target",
                                      visualize_sequences(img_batch_tensor2numpy(
                                          sample_ofs.cpu()[:num_vis, :, :, :]),
                                          seq_len=sample_ofs.size(1) // 2,
                                          return_fig=True),
                                      global_step=step + 1)
                    writer.add_figure("img/train_of_recon",
                                      visualize_sequences(img_batch_tensor2numpy(
                                          of_output.detach().cpu()[:num_vis, :, :, :]),
                                          seq_len=config["clip_pred"],  
                                          return_fig=True),
                                      global_step=step + 1)

                    writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], global_step=step + 1)

                step += 1
            del dataset

        scheduler.step()

        if epoch % config["saveevery"] == config["saveevery"] - 1:
            model_save_path = os.path.join(paths["ckpt_dir"], config["model_savename"])
            saver(model.state_dict(), optimizer.state_dict(), model_save_path, epoch + 1, step, max_to_save=5)

            # training stats
            stats_save_path = os.path.join(paths["ckpt_dir"], "training_stats.npy-%d" % (epoch + 1))
            cal_training_stats(config, model_save_path + "-%d" % (epoch + 1), training_chunked_samples_dir,
                               stats_save_path)

            with torch.no_grad():
                for r,p in zip(w_rs, w_ps):
                    config["w_r"] = r
                    config["w_p"] = p
                    auc = 0.0
                    auc_source = evaluate(config, model_save_path + "-%d" % (epoch + 1),
                                        testing_chunked_samples_files[0],
                                        stats_save_path,
                                        suffix=str(epoch + 1), dataset_name=source_dataset_name)
                    auc += auc_source
                    writer.add_scalar("auc_source_" + source_dataset_name+"/wr"+str(r)+"wp"+str(p), auc_source, global_step=epoch + 1)

                    for i, target_dataset_name in enumerate(target_dataset_names):
                        auc_target = evaluate(config, model_save_path + "-%d" % (epoch + 1),
                                              testing_chunked_samples_files[1+i],
                                              stats_save_path,
                                              suffix=str(epoch + 1), dataset_name=target_dataset_name)
                        auc += auc_target
                        writer.add_scalar("auc_target_" + target_dataset_name+"/wr"+str(r)+"wp"+str(p), auc_target, global_step=epoch + 1)
                        if auc_target > best_auc_target[target_dataset_name]["auc"]:
                            best_auc_target[target_dataset_name]["auc"] = auc_target
                            best_auc_target[target_dataset_name]["w_r"] = r
                            best_auc_target[target_dataset_name]["w_p"] = p
                            only_model_saver(model.state_dict(), os.path.join(paths["ckpt_dir"], "best_target_"+target_dataset_name+".pth"))


                    auc /= len(testing_chunked_samples_files)
                    writer.add_scalar("auc"+"/wr"+str(r)+"wp"+str(p), auc, global_step=epoch + 1)

                    if auc > best_auc["auc"]:
                        best_auc["auc"] = auc
                        best_auc["w_r"] = r
                        best_auc["w_p"] = p
                        only_model_saver(model.state_dict(), os.path.join(paths["ckpt_dir"], "best.pth"))
                    if auc_source > best_auc_source["auc"]:
                        best_auc_source["auc"] = auc_source
                        best_auc_source["w_r"] = r
                        best_auc_source["w_p"] = p
                        only_model_saver(model.state_dict(), os.path.join(paths["ckpt_dir"], "best_source.pth"))
    with open(os.path.join(paths["ckpt_dir"], "best_model.txt"), "w") as f:
        f.write(str({"best_auc":best_auc, "best_auc_source":best_auc_source, "best_auc_target":best_auc_target}))
    print({"best_auc":best_auc, "best_auc_source":best_auc_source, "best_auc_target":best_auc_target})





def cal_training_stats(config, ckpt_path, training_chunked_samples_dir, stats_save_path):
    device = config["device"]
    # model = HFVAD(num_hist=config["model_paras"]["clip_hist"],
    #               num_pred=config["model_paras"]["clip_pred"],
    #               config=config,
    #               features_root=config["model_paras"]["feature_root"],
    #               num_slots=config["model_paras"]["num_slots"],
    #               shrink_thres=config["model_paras"]["shrink_thres"],
    #               skip_ops=config["model_paras"]["skip_ops"],
    #               mem_usage=config["model_paras"]["mem_usage"],
    #               ).to(device).eval()

    model = SelfCompleteNet1raw1ofAnyPredictStage1(**config["model_paras"]).to(device).eval()

    model_weights = torch.load(ckpt_path)["model_state_dict"]
    model.load_state_dict(model_weights)
    print("load pre-trained success!")

    score_func = nn.MSELoss(reduction="none")
    training_chunk_samples_files = sorted(os.listdir(training_chunked_samples_dir))

    of_training_stats = []
    frame_training_stats = []

    print("=========Forward pass for training stats ==========")
    with torch.no_grad():

        for chunk_file_idx, chunk_file in enumerate(training_chunk_samples_files):
            dataset = Chunked_sample_dataset_new(os.path.join(training_chunked_samples_dir, chunk_file))
            dataloader = DataLoader(dataset=dataset, batch_size=config["batchsize"], num_workers=0, shuffle=False)

            for idx, data in tqdm(enumerate(dataloader),
                                  desc="Training stats calculating, Chunked File %02d" % chunk_file_idx,
                                  total=len(dataloader)):
                sample_frames, sample_ofs, _, _, _ = data
                sample_frames = sample_frames.to(device)
                sample_ofs = sample_ofs.to(device)

                of_output, raw_output, of_target, raw_target,_,_,_,_ = model(sample_frames, sample_ofs)

                loss_frame = score_func(raw_output, raw_target).cpu().data.numpy()
                loss_of = score_func(of_output, of_target).cpu().data.numpy()

                of_scores = np.sum(np.sum(np.sum(loss_of, axis=3), axis=2), axis=1)
                frame_scores = np.sum(np.sum(np.sum(loss_frame, axis=3), axis=2), axis=1)

                of_training_stats.append(of_scores)
                frame_training_stats.append(frame_scores)
            del dataset
            gc.collect()

    print("=========Forward pass for training stats done!==========")
    of_training_stats = np.concatenate(of_training_stats, axis=0)
    frame_training_stats = np.concatenate(frame_training_stats, axis=0)

    training_stats = dict(of_training_stats=of_training_stats,
                          frame_training_stats=frame_training_stats)
    # save to file
    torch.save(training_stats, stats_save_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", dest="config_file", help="config yaml file address",
                        default="./cfgs/unetOnly_AnypredictStage1_ave.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config_file))
    config["config_file"] = args.config_file
    dataset_name = config["dataset_name"]
    dataset_base_dir = config["dataset_base_dir"]
    training_chunked_samples_dir = os.path.join(dataset_base_dir, dataset_name, "training/chunked_samples")
    testing_chunked_samples_files = []
    testing_chunked_samples_files.append(os.path.join(dataset_base_dir, dataset_name,
                                                "testing/chunked_samples/chunked_samples_00.pkl"))
    cross_domain_dataset_dirs = config["crossdomain_dataset_base_dir"] if type(config["crossdomain_dataset_base_dir"]) is list else [config["crossdomain_dataset_base_dir"]]
    cross_domaindataset_names = config["crossdomain_dataset_name"] if type(config["crossdomain_dataset_name"]) is list else [config["crossdomain_dataset_name"]]
    for cross_domaindataset_name, cross_domain_dataset_dir in zip(cross_domaindataset_names, cross_domain_dataset_dirs):
        testing_chunked_samples_files.append(os.path.join(cross_domain_dataset_dir, cross_domaindataset_name,
                                                "testing/chunked_samples/chunked_samples_00.pkl"))
    # testing_chunked_samples_files are [source test, target tests]
    train(config, training_chunked_samples_dir, testing_chunked_samples_files)
