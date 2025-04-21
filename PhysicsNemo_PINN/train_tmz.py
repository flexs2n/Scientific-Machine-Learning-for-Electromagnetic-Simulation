import hydra
from omegaconf import DictConfig
import torch
import plotly
import os

from torch.optim import lr_scheduler
from torch.nn.parallel import DistributedDataParallel
from omegaconf import OmegaConf

from physicsnemo.models.fno import FNO
from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint
from physicsnemo.launch.logging import PythonLogger, LaunchLogger
from physicsnemo.launch.logging.wandb import initialize_wandb
# from modulus.sym.hydra import to_absolute_path

from loss_fn import LossTMz

from torch.optim import AdamW
from dataloader import FDTD2DDataset, EMTMzDataloader
from plot_utils import plot_predictions_tmz, plot_predictions_tmz_plotly
import wandb

dtype = torch.float
torch.set_default_dtype(dtype)


@hydra.main(version_base="1.3", config_path="config", config_name="tmz.yaml")
def main(cfg: DictConfig) -> None:
    """Training for the 2D TMz FDTD electromagnetics problem.

    This script trains a Fourier Neural Operator (FNO) on 2D TMz FDTD simulation data,
    predicting Ez, Hx, Hy fields. It uses a dynamic physics-informed loss function and
    supports distributed training, checkpointing, and visualization via wandb.
    """

    DistributedManager.initialize()
    dist = DistributedManager()

    # Initialize monitoring
    log = PythonLogger(name="tmz_pino")
    log.file_logging()

    wandb_dir = cfg.wandb_params.wandb_dir
    wandb_project = cfg.wandb_params.wandb_project
    wandb_group = cfg.wandb_params.wandb_group

    initialize_wandb(
        project=wandb_project,
        entity="fresleven",
        mode="offline",
        group=wandb_group,
        config=dict(cfg),
        results_dir=wandb_dir,
    )

    LaunchLogger.initialize(use_wandb=cfg.use_wandb)

    # Load config parameters
    model_params = cfg.model_params
    dataset_params = cfg.dataset_params
    train_loader_params = cfg.train_loader_params
    val_loader_params = cfg.val_loader_params
    test_loader_params = cfg.test_loader_params
    loss_params = cfg.loss_params
    optimizer_params = cfg.optimizer_params
    train_params = cfg.train_params
    wandb_params = cfg.wandb_params

    load_ckpt = cfg.load_ckpt
    output_dir = cfg.output_dir
    use_wandb = cfg.use_wandb

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    data_dir = dataset_params.data_dir
    ckpt_path = train_params.ckpt_path

    # Construct dataloaders
    dataset_train = FDTD2DDataset(
        dataset_params.data_dir,
        output_names=dataset_params.output_names,
        num_train=dataset_params.num_train,
        num_test=dataset_params.num_test,
        use_train=True,
    )
    dataset_val = FDTD2DDataset(
        data_dir,
        output_names=dataset_params.output_names,
        num_train=dataset_params.num_train,
        num_test=dataset_params.num_test,
        use_train=False,
    )

    tmz_dataloader_train = EMTMzDataloader(
        dataset_train,
        sub_x=dataset_params.sub_x,
        sub_y=dataset_params.sub_y,
        sub_t=dataset_params.sub_t,
        ind_x=dataset_params.ind_x,
        ind_y=dataset_params.ind_y,
        ind_t=dataset_params.ind_t,
    )
    tmz_dataloader_val = EMTMzDataloader(
        dataset_val,
        sub_x=dataset_params.sub_x,
        sub_y=dataset_params.sub_y,
        sub_t=dataset_params.sub_t,
        ind_x=dataset_params.ind_x,
        ind_y=dataset_params.ind_y,
        ind_t=dataset_params.ind_t,
    )

    dataloader_train, sampler_train = tmz_dataloader_train.create_dataloader(
        batch_size=train_loader_params.batch_size,
        shuffle=train_loader_params.shuffle,
        num_workers=train_loader_params.num_workers,
        pin_memory=train_loader_params.pin_memory,
        distributed=dist.distributed,
    )
    dataloader_val, sampler_val = tmz_dataloader_val.create_dataloader(
        batch_size=val_loader_params.batch_size,
        shuffle=val_loader_params.shuffle,
        num_workers=val_loader_params.num_workers,
        pin_memory=val_loader_params.pin_memory,
        distributed=dist.distributed,
    )

    # Define FNO model
    model = FNO(
        in_channels=model_params.in_dim,
        out_channels=model_params.out_dim,
        decoder_layers=model_params.decoder_layers,
        decoder_layer_size=model_params.fc_dim,
        dimension=model_params.dimension,
        latent_channels=model_params.layers,
        num_fno_layers=model_params.num_fno_layers,
        num_fno_modes=model_params.modes,
        padding=[model_params.pad_z, model_params.pad_y, model_params.pad_x],
    ).to(dist.device)

    # Set up DistributedDataParallel if needed
    if dist.distributed:
        ddps = torch.cuda.Stream()
        with torch.cuda.stream(ddps):
            model = DistributedDataParallel(
                model,
                device_ids=[dist.local_rank],
                output_device=dist.device,
                broadcast_buffers=dist.broadcast_buffers,
                find_unused_parameters=dist.find_unused_parameters,
            )
        torch.cuda.current_stream().wait_stream(ddps)

    # Construct optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        betas=optimizer_params.betas,
        lr=optimizer_params.lr,
        weight_decay=optimizer_params.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=optimizer_params.milestones, gamma=optimizer_params.gamma
    )

    # Construct loss class
    tmz_loss = LossTMz(**loss_params)

    # Load model from checkpoint (if exists)
    loaded_epoch = 0
    if load_ckpt:
        loaded_epoch = load_checkpoint(
            ckpt_path, model, optimizer, scheduler, device=dist.device
        )

    # Training loop
    epochs = train_params.epochs
    ckpt_freq = train_params.ckpt_freq
    names = dataset_params.fields
    input_norm = torch.tensor(model_params.input_norm).to(dist.device)
    output_norm = torch.tensor(model_params.output_norm).to(dist.device)

    for epoch in range(max(1, loaded_epoch + 1), epochs + 1):
        with LaunchLogger(
            "train",
            epoch=epoch,
            num_mini_batch=len(dataloader_train),
            epoch_alert_freq=1,
        ) as log:

            if dist.distributed:
                sampler_train.set_epoch(epoch)

            # Train loop
            model.train()

            for i, (inputs, outputs) in enumerate(dataloader_train):
                inputs = inputs.type(dtype).to(dist.device)
                outputs = outputs.type(dtype).to(dist.device)
                # Zero gradients
                optimizer.zero_grad()
                # Compute predictions
                pred = (
                    model((inputs / input_norm).permute(0, 4, 1, 2, 3)).permute(
                        0, 2, 3, 4, 1
                    )
                    * output_norm
                )
                # Compute loss
                loss, loss_dict = tmz_loss(pred, outputs, inputs, return_loss_dict=True)
                # Compute gradients
                loss.backward()
                # Update weights
                optimizer.step()

                log.log_minibatch(loss_dict)

            # Update dynamic loss weights
            tmz_loss.step()

            log.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})
            scheduler.step()

        with LaunchLogger("valid", epoch=epoch) as log:
            # Validation loop
            model.eval()
            val_loss_dict = {}
            plot_count = 0
            plot_dict = {name: {} for name in names}
            with torch.no_grad():
                for i, (inputs, outputs) in enumerate(dataloader_val):
                    inputs = inputs.type(dtype).to(dist.device)
                    outputs = outputs.type(dtype).to(dist.device)

                    # Compute predictions
                    pred = (
                        model((inputs / input_norm).permute(0, 4, 1, 2, 3)).permute(
                            0, 2, 3, 4, 1
                        )
                        * output_norm
                    )
                    # Compute loss
                    loss, loss_dict = tmz_loss(
                        pred, outputs, inputs, return_loss_dict=True
                    )

                    log.log_minibatch(loss_dict)

                    # Plot predictions for wandb
                    if (i < wandb_params.wandb_num_plots) and (
                        epoch % wandb_params.wandb_plot_freq == 0
                    ):
                        for j, _ in enumerate(pred):
                            for index, name in enumerate(names):
                                if use_wandb:
                                    figs = plot_predictions_tmz_plotly(
                                        pred[j].cpu(),
                                        outputs[j].cpu(),
                                        inputs[j].cpu(),
                                        index=index,
                                        name=name,
                                    )
                                    plot_dict[name] = {
                                        f"{plot_type}-{plot_count}": wandb.Html(
                                            plotly.io.to_html(fig)
                                        )
                                        for plot_type, fig in zip(
                                            wandb_params.wandb_plot_types, figs
                                        )
                                    }

                            plot_count += 1

                    # Save local plots
                    if (i < 2) and (epoch % wandb_params.wandb_plot_freq == 0):
                        for j, _ in enumerate(pred):
                            plot_predictions_tmz(
                                pred[j].cpu(),
                                outputs[j].cpu(),
                                inputs[j].cpu(),
                                names=names,
                                save_path=os.path.join(
                                    output_dir,
                                    "TMz_" + str(dist.rank),
                                ),
                                save_suffix=i,
                            )

            if use_wandb and epoch % wandb_params.wandb_plot_freq == 0:
                wandb.log({"plots": plot_dict})

            if epoch % ckpt_freq == 0 and dist.rank == 0:
                save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch=epoch)


if __name__ == "__main__":
    main()
