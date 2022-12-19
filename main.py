import sys
sys.path.append('..')
from datasets.mosi_dataset import MosiDataset
from models.encoder import DOMFN
from engines.mosi_trainer import *
from config.mosi_config import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import torch.backends.cudnn as cudnn
import pickle as pkl
import torch
import numpy as np

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True


def main(config):
    data_path = config.data_dir
    data = pkl.load(open(data_path, 'rb'))
    train_data = data['train']
    valid_data = data['valid']
    test_data = data['test']
    train_dataset = MosiDataset(**train_data)
    valid_dataset = MosiDataset(**valid_data)
    test_dataset = MosiDataset(**test_data)
    model = DOMFN(config.text_dim, config.vision_dim, config.audio_dim,
                 config.feature_dim, config.num_label, config.fusion)
    trainer = DomfnTrainer(config, model)
    model_path = config.model_path + 'checkpoint_' + config.version + '.pt'

    if config.is_pretrain:
        train_loader = DataLoader(train_dataset, batch_size=config.pre_batch, shuffle=True)
        for epoch in range(config.pre_epoch):
            text_loss, vision_loss, audio_loss = trainer.pre_train(train_loader)
            trainer.text_scheduler.step(text_loss)
            trainer.vision_scheduler.step(vision_loss)
            trainer.vision_scheduler.step(audio_loss)

    train_loader = DataLoader(train_dataset, batch_size=config.batch, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch, shuffle=False)
    best_mae = 10e5
    for epoch in range(config.epoch):
        train_loss, train_mae = trainer.train(train_loader)
        evaluate_loss, evaluate_mae, _, _, _ = trainer.evaluate(trainer.model, valid_loader)
        if evaluate_mae > best_mae:
            best_mae = evaluate_mae
            trainer.save(model_path)
        trainer.vision_scheduler.step(train_loss)
        trainer.audio_scheduler.step(train_loss)
        trainer.text_scheduler.step(train_loss)
    best_model = torch.load(model_path)
    best_results = trainer.test(best_model, test_loader)
    print('best results:', best_results)


if __name__ == '__main__':
    config = get_args()
    setup_seed(config.seed)
    main(config)








