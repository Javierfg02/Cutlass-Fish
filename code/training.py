import argparse
import time
import shutil
import os
import queue
import numpy as np

import tensorflow as tf
from tensorflow import keras
from model import Model
from model import build_model 
from batch import Batch 
from helpers import load_config, log_cfg, load_checkpoint, make_model_dir, \
    make_logger, set_seed, symlink_update, ConfigurationError, get_latest_checkpoint
from prediction import validate_on_data 
from loss import RegLoss
from preprocess import create_src, create_trg, create_examples, make_data_iter, map_src_sentences
from vocabulary import Vocabulary
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchtext.data import Dataset
from builders import build_optimizer, build_gradient_clipper
# from builders import build_optimizer, build_scheduler, build_gradient_clipper
from plot_videos import plot_video, alter_DTW_timing

TARGET_PAD = 0.0

class TrainManager:
    def __init__(self, model:Model, config, test=False):
        train_config = config["training"]
        model_dir = train_config["model_dir"]
        model_continue = train_config.get("continue", True)
        if not os.path.isdir(model_dir):
            model_continue = False
        if test:
            model_continue = True

        self.model_dir = make_model_dir(model_dir, overwrite=train_config.get("overwrite", False), model_continue=model_continue)
        self.logger = make_logger(self.model_dir) # TODO: put back in when done to see log
        self.logging_freq = train_config.get("logging_freq", 100)
        self.valid_report_file = "{}/validations.txt".format(self.model_dir)
        self.tb_writer = tf.summary.create_file_writer(self.model_dir + "/tensorboard/")

        self.model = model
        self.pad_index = self.model.pad_index
        self.bos_index = self.model.bos_index
        # self._log_parameters_list()
        self.target_pad = TARGET_PAD

        self.loss = RegLoss(config, target_pad=self.target_pad)

        self.learning_rate_min = train_config.get("learning_rate_min", 1.0e-8)
        self.clip_grad_fun = build_gradient_clipper(config=train_config)
        self.optimizer = build_optimizer(config=train_config)

        self.validation_freq = train_config.get("validation_freq", 1000)
        self.ckpt_best_queue = queue.Queue(maxsize=train_config.get("keep_last_ckpts", 1))
        self.ckpt_queue = queue.Queue(maxsize=1)

        self.val_on_train = config["data"].get("val_on_train", False)

        self.eval_metric = train_config.get("eval_metric", "dtw").lower()
        if self.eval_metric not in ['bleu', 'chrf', "dtw"]:
            raise ConfigurationError("Invalid setting for 'eval_metric', valid options: 'bleu', 'chrf', 'DTW'")

        self.early_stopping_metric = train_config.get("early_stopping_metric", "eval_metric")
        self.minimize_metric = True if self.early_stopping_metric in ["loss", "dtw"] else False

        # self.scheduler, self.scheduler_step_at = build_scheduler(config=train_config, scheduler_mode="min" if self.minimize_metric else "max", optimizer=self.optimizer, hidden_size=config["model"]["encoder"]["hidden_size"])

        self.level = "word"
        self.shuffle = train_config.get("shuffle", True)
        self.epochs = train_config["epochs"]
        self.batch_size = train_config["batch_size"]
        self.batch_type = "sentence"
        self.eval_batch_size = train_config.get("eval_batch_size", self.batch_size)
        self.eval_batch_type = train_config.get("eval_batch_type", self.batch_type)
        self.batch_multiplier = train_config.get("batch_multiplier", 1)

        self.max_output_length = train_config.get("max_output_length", None)

        self.use_cuda = train_config["use_cuda"]
        if self.use_cuda:
            self.model = self.model.cuda()  # For TensorFlow, adapt this for GPU settings

        self.steps = 0
        self.stop = False
        self.total_tokens = 0
        self.best_ckpt_score = np.inf if self.minimize_metric else -np.inf
        self.is_best = lambda score: score < self.best_ckpt_score if self.minimize_metric else score > self.best_ckpt_score

        if model_continue:
            ckpt = get_latest_checkpoint(model_dir)
            if ckpt is None:
                self.logger.info("Can't find checkpoint in directory %s", ckpt)
            else:
                self.logger.info("Continuing model from %s", ckpt)
                self.init_from_checkpoint(ckpt)

        self.skip_frames = config["data"].get("skip_frames", 1)
        self.just_count_in = config["model"].get("just_count_in", False)
        self.gaussian_noise = config["model"].get("gaussian_noise", False)
        if self.gaussian_noise:
            self.noise_rate = config["model"].get("noise_rate", 1.0)

        if self.just_count_in and self.gaussian_noise:
            raise ConfigurationError("Can't have both just_count_in and gaussian_noise as True")

        self.future_prediction = config["model"].get("future_prediction", 0)
        if self.future_prediction != 0:
            frames_predicted = [i for i in range(self.future_prediction)]
            self.logger.info("Future prediction. Frames predicted: %s", frames_predicted)

    def _save_checkpoint(self, type="every"):
        model_path = "{}/{}_{}.ckpt".format(self.model_dir, self.steps, type)
        state = {
            "steps": self.steps,
            "total_tokens": self.total_tokens,
            "best_ckpt_score": self.best_ckpt_score,
            "best_ckpt_iteration": self.best_ckpt_iteration,
            "model_state": self.model.get_weights(),
            "optimizer_state": self.optimizer.get_weights(),
            # "scheduler_state": self.scheduler.get_config() if self.scheduler is not None else None,
        }
        np.save(model_path, state)  # Use TensorFlow's save mechanism or np.save for simplicity
        if type == "best":
            if self.ckpt_best_queue.full():
                to_delete = self.ckpt_best_queue.get()
                try:
                    os.remove(to_delete)
                except FileNotFoundError:
                    self.logger.warning("Wanted to delete old checkpoint %s but file does not exist.", to_delete)
            self.ckpt_best_queue.put(model_path)
            best_path = "{}/best.ckpt".format(self.model_dir)
            try:
                symlink_update("{}_best.ckpt".format(self.steps), best_path)
            except OSError:
                np.save(best_path, state)  # Use TensorFlow's save mechanism or np.save for simplicity

        elif type == "every":
            if self.ckpt_queue.full():
                to_delete = self.ckpt_queue.get()
                try:
                    os.remove(to_delete)
                except FileNotFoundError:
                    self.logger.warning("Wanted to delete old checkpoint %s but file does not exist.", to_delete)
            self.ckpt_queue.put(model_path)
            every_path = "{}/every.ckpt".format(self.model_dir)
            try:
                symlink_update("{}_every.ckpt".format(self.steps), every_path)
            except OSError:
                np.save(every_path, state)  # Use TensorFlow's save mechanism or np.save for simplicity

    def init_from_checkpoint(self, path: str):
        model_checkpoint = np.load(path, allow_pickle=True).item()
        self.model.set_weights(model_checkpoint["model_state"])
        self.optimizer.set_weights(model_checkpoint["optimizer_state"])
        # if model_checkpoint["scheduler_state"] is not None and self.scheduler is not None:
        #     self.scheduler.from_config(model_checkpoint["scheduler_state"])
        self.steps = model_checkpoint["steps"]
        self.total_tokens = model_checkpoint["total_tokens"]
        self.best_ckpt_score = model_checkpoint["best_ckpt_score"]
        self.best_ckpt_iteration = model_checkpoint["best_ckpt_iteration"]
        if self.use_cuda:
            self.model = self.model.cuda()  # Adapt this for TensorFlow GPU settings

    def train_and_validate(self, train_data, valid_data):
        #! Figure out make_data_iter
        mapped_dataset = map_src_sentences(dataset=train_data)
        # for ex in mapped_dataset:
        #     print("OUTPUT MAPPED: ", ex.src)
        train_iter = make_data_iter(dataset=mapped_dataset, train=True, shuffle=self.shuffle)
        # for ex in train_iter:
        #     print("OUTPUT TRAIN ITER: ", ex.src)
        # train_iter = make_data_iter(dataset=train_data, train=True, shuffle=self.shuffle)
        # src_dataset, trg_dataset, filepath = make_data_iter(dataset=train_data, batch_size=self.batch_size, pad_token_id=PAD_TOKEN_ID, train=True, shuffle=self.shuffle)
        val_step = 0
        if self.gaussian_noise:
            all_epoch_noise = []

        for epoch_no in range(self.epochs):
            self.logger.info("EPOCH %d", epoch_no + 1)
            # if self.scheduler is not None and self.scheduler_step_at == "epoch":
            #     self.scheduler.step(epoch=epoch_no)

            # TODO: comment back in for tensorflow version
            # self.model.train()
            start = time.time()
            total_valid_duration = 0
            start_tokens = self.total_tokens
            count = self.batch_multiplier - 1
            epoch_loss = 0

            # TODO: make list ourself to train each batch
            # train_iter = zip(src_dataset, trg_dataset)
            for batch in iter(train_iter):
                # self.model.train() 
                # print("trg: ", batch.trg)
                # print("trg 0: ", batch.trg[0] )
                # print("trg 0 0: ", batch.trg[0][0])
                # print("pre: ", batch.src)
                batch = Batch(torch_batch=batch, pad_index=self.pad_index, model=self.model)
                update = count == 0

                # print("postbatch: ", batch.src)
                batch_loss, noise = self._train_batch(batch, update=update)
                if self.gaussian_noise:
                    if self.future_prediction != 0:
                        all_epoch_noise.append(noise.reshape(-1, self.model.out_trg_size // self.future_prediction))
                    else:
                        all_epoch_noise.append(noise.reshape(-1, self.model.out_trg_size))

                with self.tb_writer.as_default():
                    tf.summary.scalar("train/train_batch_loss", batch_loss, step=self.steps)

                count = self.batch_multiplier if update else count
                count -= 1
                epoch_loss += batch_loss.numpy()

                # if self.scheduler is not None and self.scheduler_step_at == "step" and update:
                #     self.scheduler.step()

                if self.steps % self.logging_freq == 0 and update:
                    elapsed = time.time() - start - total_valid_duration
                    elapsed_tokens = self.total_tokens - start_tokens
                    self.logger.info(
                        "Epoch %3d Step: %8d Batch Loss: %12.6f "
                        "Tokens per Sec: %8.0f, Lr: %.6f",
                        epoch_no + 1, self.steps, batch_loss,
                        elapsed_tokens / elapsed,
                        self.optimizer.learning_rate.numpy())
                    start = time.time()
                    total_valid_duration = 0
                    start_tokens = self.total_tokens

                if self.steps % self.validation_freq == 0 and update:
                    valid_start_time = time.time()
                    valid_score, valid_loss, valid_references, valid_hypotheses, valid_inputs, all_dtw_scores, valid_file_paths = \
                        validate_on_data(
                            model=self.model,
                            data=valid_data,
                            batch_size=self.eval_batch_size,
                            max_output_length=self.max_output_length,
                            loss_function=self.loss,
                            batch_type=self.eval_batch_type,
                            type="val",
                        )

                    val_step += 1

                    with self.tb_writer.as_default():
                        tf.summary.scalar("valid/valid_loss", valid_loss, step=self.steps)
                        tf.summary.scalar("valid/valid_score", valid_score, step=self.steps)

                    ckpt_score = valid_loss if self.early_stopping_metric == "loss" else valid_score

                    new_best = False
                    self.best = False
                    if self.is_best(ckpt_score):
                        self.best = True
                        self.best_ckpt_score = ckpt_score
                        self.best_ckpt_iteration = self.steps
                        self.logger.info(
                            'New best validation result [%s]!',
                            self.early_stopping_metric)
                        if self.ckpt_queue.maxsize > 0:
                            self.logger.info("Saving new checkpoint.")
                            new_best = True
                            self._save_checkpoint(type="best")

                        display = list(range(0, len(valid_hypotheses), int(np.ceil(len(valid_hypotheses) / 13.15))))
                        self.produce_validation_video(
                            output_joints=valid_hypotheses,
                            inputs=valid_inputs,
                            references=valid_references,
                            model_dir=self.model_dir,
                            steps=self.steps,
                            display=display,
                            type="val_inf",
                            file_paths=valid_file_paths,
                        )

                    self._save_checkpoint(type="every")

                    # if self.scheduler is not None and self.scheduler_step_at == "validation":
                    #     self.scheduler.step(ckpt_score)

                    self._add_report(
                        valid_score=valid_score, valid_loss=valid_loss,
                        eval_metric=self.eval_metric,
                        new_best=new_best, report_type="val",)

                    valid_duration = time.time() - valid_start_time
                    total_valid_duration += valid_duration
                    self.logger.info(
                        'Validation result at epoch %3d, step %8d: Val DTW Score: %6.2f, '
                        'loss: %8.4f,  duration: %.4fs',
                        epoch_no+1, self.steps, valid_score,
                        valid_loss, valid_duration)

                if self.stop:
                    break
            if self.stop:
                self.logger.info(
                    'Training ended since minimum lr %f was reached.',
                     self.learning_rate_min)
                break

            self.logger.info('Epoch %3d: total training loss %.5f', epoch_no+1,
                             epoch_loss)
        else:
            self.logger.info('Training ended after %3d epochs.', epoch_no+1)
        self.logger.info('Best validation result at step %8d: %6.2f %s.',
                         self.best_ckpt_iteration, self.best_ckpt_score,
                         self.early_stopping_metric)

        self.tb_writer.close()  # close Tensorboard writer

    def produce_validation_video(self, output_joints, inputs, references, display, model_dir, type, steps="", file_paths=None):
        if type != "test":
            dir_name = model_dir + "/videos/Step_{}/".format(steps)
            if not os.path.exists(model_dir + "/videos/"):
                os.mkdir(model_dir + "/videos/")

        elif type == "test":
            dir_name = model_dir + "/test_videos/"

        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        for i in display:
            seq = output_joints[i]
            ref_seq = references[i]
            input = inputs[i]
            gloss_label = input[0]
            if input[1] != "</s>":
                gloss_label += "_" + input[1]
            if input[2] != "</s>":
                gloss_label += "_" + input[2]

            timing_hyp_seq, ref_seq_count, dtw_score = alter_DTW_timing(seq, ref_seq)

            video_ext = "{}_{}.mp4".format(gloss_label, "{0:.2f}".format(float(dtw_score)).replace(".", "_"))

            if file_paths is not None:
                sequence_ID = file_paths[i]
            else:
                sequence_ID = None

            if "<" not in video_ext:
                plot_video(joints=timing_hyp_seq,
                           file_path=dir_name,
                           video_name=video_ext,
                           references=ref_seq_count,
                           skip_frames=self.skip_frames,
                           sequence_ID=sequence_ID)

    def _train_batch(self, batch, update=True):
        batch_loss, noise = self.model.get_loss_for_batch(batch, loss_function=self.loss)
        normalizer = batch.nseqs if self.normalization == "batch" else batch.ntokens
        norm_batch_loss = batch_loss / normalizer
        norm_batch_multiply = norm_batch_loss / self.batch_multiplier

        with tf.GradientTape() as tape:
            gradients = tape.gradient(norm_batch_multiply, self.model.trainable_variables)

        # if self.clip_grad_fun is not None:
        #     self.clip_grad_fun(gradients, self.model.trainable_variables)

        if update:
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            self.steps += 1

        self.total_tokens += batch.ntokens

        return norm_batch_loss, noise

    def _add_report(self, valid_score, valid_loss, eval_metric, new_best=False, report_type="val"):
        current_lr = -1
        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']

        if current_lr < self.learning_rate_min:
            self.stop = True

        if report_type == "val":
            with open(self.valid_report_file, 'a') as opened_file:
                opened_file.write(
                    "Steps: {} Loss: {:.5f}| DTW: {:.3f}|"
                    " LR: {:.6f} {}\n".format(
                        self.steps, valid_loss, valid_score,
                        current_lr, "*" if new_best else ""))

    def _log_parameters_list(self):
        # model_parameters = filter(lambda p: p.requires_grad, self.model.trainable_variables)
        model_parameters = self.model.trainable_variables
        n_params = sum(np.prod(v.shape) for v in model_parameters)
        # n_params = sum([np.prod(p.shape) for p in model_parameters])
        self.logger.info("Total params: %d", n_params)
        # TODO: CHANGE TO MODEL.LAYERS?
        trainable_params = [v.name for v in model_parameters]

        # trainable_params = [n for (n, p) in self.model.named_parameters() if p.requires_grad]
        self.logger.info("Trainable parameters: %s", sorted(trainable_params))
        assert trainable_params


def train(cfg_file, ckpt=None):
    cfg = load_config(cfg_file)
    vocab = Vocabulary()
    set_seed(seed=cfg["training"].get("random_seed", 42))
    # train_data, dev_data, test_data, src_vocab, trg_vocab = load_data(cfg=cfg) #! Important for the data processing
    trg = create_trg()
    src = create_src()
    train_data = create_examples(src, trg)
    # print("FIRST TRAIN DATA KEYS: ", train_data[0].__dict__.keys())
    print("FIRST TRAIN DATA SRC: ", train_data[0].src)
    print("TRAIN DATA TRG: ", train_data[0].trg)
    print("FIRST TRAIN DATA TRG LEN: ", len(train_data[0].trg))
    print("FIRST TRAIN DATA TRG 0 SHAPE: ", train_data[0].trg[0].shape)

    # print("TRAIN DATA TRG: ", train_data[1])
    dev_data = create_examples(src, trg) #! at the moment, validation and training datasets are the same - should not be
    vocab._from_file('../configs/src_vocab.txt')
    # print("vocab stoi: ", vocab.stoi)
    src_vocab = vocab
    trg_vocab = [None] * len(src_vocab)
    model = build_model(cfg, src_vocab=src_vocab, trg_vocab=trg_vocab)

    if ckpt is not None:
        use_cuda = cfg["training"].get("use_cuda", False)
        model_checkpoint = load_checkpoint(ckpt, use_cuda=use_cuda)
        model.load_state_dict(model_checkpoint["model_state"])

    trainer = TrainManager(model=model, config=cfg)
    shutil.copy2(cfg_file, trainer.model_dir + "/config.yaml")
    log_cfg(cfg, trainer.logger)
    # train_data = torch.Tensor(train_data)
    # train_data = Dataset(train_data)
    print("train data: ", train_data)
    trainer.train_and_validate(train_data, dev_data)
    test(cfg_file)

def test(cfg_file, ckpt=None):
    cfg = load_config(cfg_file)
    model_dir = cfg["training"]["model_dir"]
    if ckpt is None:
        ckpt = get_latest_checkpoint(model_dir, post_fix="_best")
        if ckpt is None:
            raise FileNotFoundError("No checkpoint found in directory {}.".format(model_dir))

    batch_size = cfg["training"].get("eval_batch_size", cfg["training"]["batch_size"])
    batch_type = cfg["training"].get("eval_batch_type", cfg["training"].get("batch_type", "sentence"))
    use_cuda = cfg["training"].get("use_cuda", False)
    eval_metric = cfg["training"]["eval_metric"]
    max_output_length = cfg["training"].get("max_output_length", None)
    # train_data, dev_data, test_data, src_vocab, trg_vocab = load_data(cfg=cfg) #! Again, adapt for our data

    data_to_predict = {"test": test_data}
    model_checkpoint = load_checkpoint(ckpt, use_cuda=use_cuda)
    model = build_model(cfg, src_vocab=src_vocab, trg_vocab=trg_vocab)
    model.load_state_dict(model_checkpoint["model_state"])
    if use_cuda:
        model.cuda()

    trainer = TrainManager(model=model, config=cfg, test=True)
    for data_set_name, data_set in data_to_predict.items():
        score, loss, references, hypotheses, inputs, all_dtw_scores, file_paths = validate_on_data(
            model=model, data=data_set, batch_size=batch_size, max_output_length=max_output_length, eval_metric=eval_metric, loss_function=None, batch_type=batch_type, type="val" if data_set_name != "train" else "train_inf"
        )
        display = list(range(len(hypotheses)))
        trainer.produce_validation_video(output_joints=hypotheses, inputs=inputs, references=references, model_dir=model_dir, display=display, type="test", file_paths=file_paths)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Progressive Transformers")
    parser.add_argument("config", default="configs/default.yaml", type=str, help="Training configuration file (yaml).")
    args = parser.parse_args()
    train(cfg_file=args.config)
