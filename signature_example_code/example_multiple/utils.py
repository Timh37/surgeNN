import torch
import torch.nn.functional as F
import torch.utils.data as torchdata
import ignite.engine as engine
import ignite.metrics as ignite_metrics
import tqdm
import time
import sys

def count_parameters(model):
    '''
    Count the number of model parameters.
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def generate_torch_batched_data(x_train, y_train, x_test, y_test, train_batch_size, test_batch_size,
                                x_valid=None, y_valid=None, valid_batch_size=None,
                                shuffle=False, num_workers=0):
    '''
    Generate torch dataloaders. (There is no validation dataset by default.)
    Arg:
        - x(y)_train(valid, test): shape can be (batch, seq, feature) or (batch, feature).
        - shuffle: set to True to have the data reshuffled at every epoch (default: False).
        - num_workers: how many subprocesses to use for data loading (default: 0).
    '''
    # 'TensorDataset' 将输入数据和目标数据组合成一个数据集对象
    # 'Dataloader' 用于创建数据加载器， 在迭代训练时按批次提供数据
    train_dataset = torchdata.TensorDataset(torch.tensor(x_train, dtype=torch.float),
                                            torch.tensor(y_train, dtype=torch.float))
    train_dataloader = torchdata.DataLoader(train_dataset, batch_size=train_batch_size,
                                            shuffle=shuffle, num_workers=num_workers)
    test_dataset = torchdata.TensorDataset(torch.tensor(x_test, dtype=torch.float),
                                           torch.tensor(y_test, dtype=torch.float))
    test_dataloader = torchdata.DataLoader(test_dataset, batch_size=test_batch_size,
                                           shuffle=shuffle, num_workers=num_workers)
    # 用于初始化模型
    example_batch_x, example_batch_y = next(iter(train_dataloader))
    if x_valid is not None and y_valid is not None:
        valid_dataset = torchdata.TensorDataset(torch.tensor(x_valid, dtype=torch.float),
                                                torch.tensor(y_valid, dtype=torch.float))
        valid_dataloader = torchdata.DataLoader(valid_dataset, batch_size=valid_batch_size,
                                                shuffle=shuffle, num_workers=num_workers)
        return train_dataloader, valid_dataloader, test_dataloader, example_batch_x, example_batch_y
    else:
        return train_dataloader, test_dataloader, example_batch_x, example_batch_y
def create_model_supervised_trainer(lr, optimizer_fn, loss_fn, max_epochs, train_dataloader, eval_dataloader, example_batch_x):
    '''
    To save and load the trained model:
        # save
        torch.save(model.state_dict(), 'my_model.pth')
        # load
        model.load_state_dict(torch.load('my_model.pth'))
        # switch the model to evaluation mode
        model.eval()
        # prediction
        prediction = model(input_data)
    Args:
        - loss_fn: 可自定的损失函数，不用于训练。
    '''

    def loss_function(y_pre,y):
        '''
        默认使用MSE作为训练的损失函数。
        '''
        return F.mse_loss(y_pre,y)
    def train_model(model, model_name, history, device=None):
        model(example_batch_x)# 初始化模型
        optimizer = optimizer_fn(model.parameters(), lr=lr)
        history[model_name] = {'train_loss': [], 'train_mse': [], 'eval_loss': [], 'eval_mse': []}
        if device not in ('cuda', 'cpu'):
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # 定义训练器和评估器
        trainer = engine.create_supervised_trainer(model, optimizer, loss_function, device=device)
        evaluator = engine.create_supervised_evaluator(model, device=device, metrics={'loss': ignite_metrics.Loss(loss_fn),
                                                                                      'mse': ignite_metrics.MeanSquaredError()})

        desc = "Epoch: {:4}{:12}"
        num_batches = len(train_dataloader)

        @trainer.on(engine.Events.STARTED)
        def logging_results(trainer):
            # training process
            evaluator.run(train_dataloader)
            train_mse = evaluator.state.metrics['mse']
            train_loss = evaluator.state.metrics['loss']
            # evaluating process
            evaluator.run(eval_dataloader)
            eval_mse = evaluator.state.metrics['mse']
            eval_loss = evaluator.state.metrics['loss']
            # logging
            tqdm.tqdm.write("train mse: {:5.4f} --- train loss: {:5.4f} --- eval mse: {:5.4f} --- eval loss: {:5.4f}"
                            .format(train_mse, train_loss, eval_mse, eval_loss), file=sys.stdout)
            model_history = history[model_name]
            model_history['train_mse'].append(train_mse); model_history['train_loss'].append(train_loss)
            model_history['eval_mse'].append(eval_mse); model_history['eval_loss'].append(eval_loss)

        @trainer.on(engine.Events.EPOCH_STARTED)
        def creat_pbar(trainer):
            trainer.state.pbar = tqdm.tqdm(initial=0, total=num_batches, desc=desc.format(trainer.state.epoch, ''),
                                           file=sys.stdout)

        @trainer.on(engine.Events.EPOCH_COMPLETED)
        def logging_results_(trainer):
            trainer.state.pbar.n = num_batches
            trainer.state.pbar.last_print_n = num_batches
            trainer.state.pbar.refresh()
            trainer.state.pbar.close()
            logging_results(trainer)

        start = time.time()
        trainer.run(train_dataloader, max_epochs=max_epochs)
        end = time.time()
        tqdm.tqdm.write("Training took {:.2f} seconds.".format(end - start), file=sys.stdout)
    return train_model

